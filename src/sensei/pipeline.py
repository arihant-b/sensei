import dataclasses
import logging
import re
import subprocess
from pathlib import Path

from xgboost import Booster

from sensei.config import Settings, ensense_pin
from sensei.data.bins import FrozenBins
from sensei.data.loader import Dataset
from sensei.eval.heldout_verify import HeldoutResult, HeldoutVerifier
from sensei.eval.metrics import Metrics
from sensei.eval.plots import ResultsPlotter
from sensei.eval.sweeps import Sweeper, SweepPoint
from sensei.model.leaves import LeafMap
from sensei.model.train import Trainer
from sensei.oracle.sensei.encoding import TreeEncoder
from sensei.oracle.sensei.solve import SenseiOracle
from sensei.oracle.types import Pair, TreeStructure
from sensei.repair.cuts import Cut
from sensei.repair.loop import (
    CegsalLoop,
    ConditionalCertificate,
    Inconclusive,
    ParetoBest,
)
from sensei.repair.pareto import Snapshot
from sensei.spec import Spec, load_spec

log: logging.Logger = logging.getLogger("sensei.pipeline")


def _safe_filename(label: str) -> str:
    """
    Turn a schedule label like 'protected_pair:(sex,race)' into a
    filename-safe stem.

    Args:
        label (str): The label to sanitize.

    Returns:
        str: The filename-safe stem.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


class CertifyPipeline:
    """
    The search -> repair -> verify -> sweep pipeline: load data, train M0,
    run CEGSAL, then held-out verify + theta/eps sweep the result. Shared by
    `experiments/run_certify.py` and the `sensei` CLI -- one implementation,
    two `Settings` sources.
    """

    def run(
        self, dataset: str, feature: str, direction: str, settings: Settings
    ) -> dict:
        """
        Run the full pipeline for one `feature`/`direction`: load data,
        train M0, run CEGSAL, then held-out verify + theta/eps sweep.

        Args:
            dataset (str): Dataset name, matching `dataset/<name>/` and
                `spec/<name>.yaml`.
            feature (str): Feature to repair sensitivity to.
            direction (str): `"protected"` or `"monotone_wrong"`.
            settings (Settings): Every hyperparameter for this run.

        Returns:
            dict: The results JSON payload (also written to disk by the
                caller via `ResultsWriter`).
        """

        spec: Spec = load_spec(dataset)
        ds: Dataset = Dataset(
            dataset,
            eval_holdout=settings.dataset.eval_holdout,
            seed=settings.seed,
        ).load()

        log.info(
            "training M0: %d trees, depth %d",
            settings.model.n_estimators,
            settings.model.max_depth,
        )

        assert ds.X_train is not None and ds.y_train is not None
        assert ds.X_test is not None and ds.y_test is not None
        assert ds.columns is not None and ds.feature_bounds is not None

        booster: Booster = Trainer.train_baseline(
            ds.X_train,
            ds.y_train,
            settings.model.n_estimators,
            settings.model.max_depth,
            settings.seed,
        )

        log.info("--- CEGSAL: repairing '%s' ---", feature)
        loop = CegsalLoop(
            booster,
            ds.X_train,
            ds.y_train,
            ds.X_test,
            ds.y_test,
            ds.columns,
            ds.feature_bounds,
            spec,
            settings,
            ds.categorical_levels,
        )
        result: ConditionalCertificate | ParetoBest | Inconclusive = loop.run(
            flip_set=(feature,), direction=direction
        )

        exit_kind: str = type(result).__name__
        log.info("CEGSAL exit: %s", exit_kind)

        plotter = ResultsPlotter(
            label=f"{dataset}_{feature}_{direction}",
            hyperparams=self._plot_hyperparams(settings),
        )
        self._plot_training_history(plotter, loop.history)

        payload: dict = {
            "git_sha": self._git_sha(),
            "ensense_pin": ensense_pin(),
            "dataset": dataset,
            "feature": feature,
            "direction": direction,
            "plots_dir": str(plotter.run_dir),
            "spec_hash": spec.spec_hash,
            "eps": settings.sensitivity.eps,
            "theta": settings.sensitivity.theta,
            "mu": settings.repair.mu,
            "kap": settings.repair.kap,
            "oracle_type": settings.oracle.type,
            "seed": settings.seed,
            "cegsal_exit": exit_kind,
        }

        if isinstance(result, ConditionalCertificate):
            payload["certificate"] = result.statement()
            snapshot: Snapshot | None = result.snapshot
        elif isinstance(result, ParetoBest):
            payload["pareto_reason"] = result.reason
            snapshot: Snapshot | None = result.snapshot
        else:
            assert isinstance(result, Inconclusive)
            payload["inconclusive_reason"] = result.reason
            payload["inconclusive_iteration"] = result.iteration
            snapshot: Snapshot | None = result.snapshot

        if snapshot is None:
            log.warning("no snapshot met the accuracy floor -- stopping here")
            payload["note"] = "no snapshot met A_min; nothing to certify/verify further"
            return payload

        payload["snapshot"] = {
            "iteration": snapshot.iteration,
            "accuracy": snapshot.accuracy,
            "sensitivity_rate": snapshot.sensitivity_rate,
            "worst_gap": snapshot.worst_gap,
            "slack_mass": snapshot.slack_mass,
            "n_cuts": snapshot.n_cuts,
        }

        leaf_map = LeafMap(booster)
        X_eval, y_eval = ds.X_eval, ds.y_eval
        payload["snapshot"]["eval_accuracy"] = Metrics.accuracy(
            booster, leaf_map, X_eval, y_eval, snapshot.v
        )
        payload["snapshot"]["eval_sensitivity_rate"] = Metrics.sensitivity_rate(
            booster,
            leaf_map,
            X_eval,
            spec.protected,
            snapshot.v,
            seed=settings.seed,
        )

        repaired_booster: Booster = leaf_map.write_leaf_values(booster, snapshot.v)
        structure: TreeStructure = TreeEncoder.extract_tree_structure(
            booster, leaf_map.leaf_index, ds.columns
        )
        bins: FrozenBins = FrozenBins.fit_or_load(
            dataset,
            settings.seed,
            settings.bins.n_quantile_bins,
            ds.X_train,
            ds.columns,
            ds.categorical_levels,
        )
        payload.update(
            self._verify_flip_set(
                repaired_booster,
                booster,
                ds,
                spec,
                bins,
                structure,
                (feature,),
                direction,
                snapshot.v,
                snapshot.cuts,
                settings,
                plotter,
                f"{direction}:{feature}",
            )
        )

        return payload

    def _verify_flip_set(
        self,
        repaired_booster: Booster,
        booster: Booster,
        ds: Dataset,
        spec: Spec,
        bins: FrozenBins,
        structure: TreeStructure,
        flip_set: tuple[str, ...],
        direction: str,
        v,
        cuts: list[Cut],
        settings: Settings,
        plotter: ResultsPlotter,
        stage_label: str,
    ) -> dict:
        """
        Held-out verification (Ensense oracle, fresh), a Sensei oracle
        self-check, and theta/eps sweeps for ONE flip_set/direction against a
        fixed, already-repaired `v`. Called once by `run`.

        `plotter`/`stage_label` are used only to save the theta/eps sweep
        plots (`<plotter.run_dir>/theta_sweep__<stage_label>.png` etc.) --
        every OTHER plot (per-iteration diagnostics, Pareto curves) is saved
        by the caller directly from `CegsalLoop.history`/`.stage_history`,
        which this method has no access to.

        Args:
            repaired_booster (Booster): The repaired model to verify.
            booster (Booster): The unrepaired model, for the self-check/sweeps.
            ds (Dataset): Supplies columns/feature_bounds/X_train/categorical_levels.
            spec (Spec): Dataset spec.
            bins (FrozenBins): Frozen plausibility bins.
            structure (TreeStructure): Pre-extracted tree structure.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            v: The repaired leaf values.
            cuts (list[Cut]): Accumulated cuts, for the region-overlap check.
            settings (Settings): Every hyperparameter for this run.
            plotter (ResultsPlotter): Where to save the sweep plots.
            stage_label (str): This stage's label, for plot filenames.

        Returns:
            dict: Held-out verification, self-check, and sweep results, to
                merge into the run's payload.
        """

        payload: dict = {}
        leaf_map = LeafMap(booster)
        log.info("--- held-out verification (Ensense core) ---")

        assert ds.columns is not None and ds.feature_bounds is not None

        hv: HeldoutResult = HeldoutVerifier.verify(
            repaired_booster,
            ds.columns,
            flip_set,
            spec,
            bins,
            ds.feature_bounds,
            settings,
        )
        payload["heldout_verify"] = {"generalized": hv.generalized, "note": hv.note}

        if hv.fresh_pair is not None:
            payload["heldout_verify"]["fresh_gap"] = hv.fresh_pair.gap
            rate: float = Metrics.overlap_rate(
                booster, leaf_map, [hv.fresh_pair], ds.columns, cuts
            )
            payload["region_overlap_rate"] = rate
            log.info("region overlap rate: %.2f", rate)

        log.info("--- sensei oracle self-check at the repaired v ---")
        # mip_gap forced to 0 for the self-check -- it should confirm the
        # loop's own feasibility result at the TRUE tolerance, not the
        # looser intermediate-round `settings.oracle.mip_gap`.
        self_check_settings: Settings = dataclasses.replace(
            settings, oracle=dataclasses.replace(settings.oracle, mip_gap=0.0)
        )
        sensei_pair: Pair | None = SenseiOracle().worst_valid_pair(
            booster,
            leaf_map,
            ds.columns,
            ds.feature_bounds,
            spec,
            flip_set,
            direction,
            mode="feasibility",
            settings=self_check_settings,
            enforce_validity=True,
            structure=structure,
            v=v,
            bins=bins,
        )
        payload["sensei_vs_ensense"] = {
            "sensei_unsat": sensei_pair is None,
            "ensense_unsat": hv.generalized,
            "note": (
                "Sensei-oracle UNSAT means absence within our Q1+Q2-encoded domain; "
                "Ensense-oracle UNSAT means absence under Ensense core's own search, "
                "which does not enforce our declared validity/plausibility rules "
                "inside its search at all."
            ),
        }

        log.info("--- theta sweep ---")
        thetas: list[float] = [1e-30, 1e-15, 1e-9, 1e-6, 1e-3, 1e-1, 0.5]
        theta_points: list[SweepPoint] = Sweeper.theta_sweep(
            booster,
            ds.columns,
            ds.feature_bounds,
            spec,
            bins,
            flip_set,
            direction,
            thetas=thetas,
            settings=settings,
            v=v,
            structure=structure,
        )
        payload["theta_sweep"] = [
            {
                "theta": p.value,
                "certified": p.certified,
                "gap": p.gap,
                "status": p.status,
            }
            for p in theta_points
        ]
        for p in theta_points:
            log.info("  theta=%-10.0e certified=%s gap=%s", p.value, p.certified, p.gap)

        log.info("--- eps sweep ---")
        epsilons: list[float] = [0.05, 0.1, 0.2, 0.5, 1.0]
        eps_points: list[SweepPoint] = Sweeper.eps_sweep(
            booster,
            ds.columns,
            ds.feature_bounds,
            spec,
            bins,
            flip_set,
            direction,
            epsilons=epsilons,
            settings=settings,
            v=v,
            structure=structure,
        )
        payload["eps_sweep"] = [
            {"eps": p.value, "certified": p.certified, "gap": p.gap, "status": p.status}
            for p in eps_points
        ]
        for p in eps_points:
            log.info("  eps=%-6.2f certified=%s gap=%s", p.value, p.certified, p.gap)

        safe_label: str = _safe_filename(stage_label)
        try:
            theta_path: Path = plotter.theta_sweep(
                theta_points, name=f"theta_sweep__{safe_label}"
            )
            eps_path: Path = plotter.eps_sweep(
                eps_points, name=f"eps_sweep__{safe_label}"
            )
            payload["theta_sweep_plot"] = str(theta_path)
            payload["eps_sweep_plot"] = str(eps_path)
        except Exception:
            log.warning(
                "failed to plot theta/eps sweep for %s -- continuing without it",
                stage_label,
                exc_info=True,
            )

        return payload

    def _plot_hyperparams(self, settings: Settings) -> dict[str, object]:
        """
        The run-identifying settings stamped as a caption on every plot this
        run produces (see `ResultsPlotter.__init__`), read off one
        `Settings` object for the whole run.

        Args:
            settings (Settings): Every hyperparameter for this run.

        Returns:
            dict[str, object]: The subset to caption plots with.
        """

        return {
            "n_estimators": settings.model.n_estimators,
            "max_depth": settings.model.max_depth,
            "eps": settings.sensitivity.eps,
            "theta": settings.sensitivity.theta,
            "oracle_type": settings.oracle.type,
        }

    def _plot_training_history(
        self, plotter: ResultsPlotter, history: list[Snapshot], suffix: str = ""
    ) -> list[Path]:
        """
        Save the standard set of per-iteration training plots (accuracy,
        sensitivity rate, worst gap, slack mass, cut count vs. iteration,
        the Pareto curve, and the combined dashboard) from ONE stage's own
        snapshot history. `worst_gap`/the Pareto curve are only meaningful
        within a single flip_set/direction's own iterations.

        Never raises: a plotting failure is logged as a warning and the
        certify/repair result is still returned -- a missing diagnostic
        plot must not take down an otherwise-successful run.

        `suffix` disambiguates per-stage plots saved into the same
        `plotter.run_dir` (e.g. `"protected_single_sex"`) -- default ""
        for the single-flip-set `run()` case, one stage per plotter.

        Args:
            plotter (ResultsPlotter): Where to save the plots.
            history (list[Snapshot]): One stage's per-iteration snapshots.
            suffix (str): Disambiguates per-stage plot filenames.

        Returns:
            list[Path]: The plots actually produced, empty on any failure
                or if `history` has fewer than 2 snapshots.
        """

        if len(history) < 2:
            log.warning(
                "training history has only %d snapshot(s) -- skipping per-iteration "
                "plots (nothing to show a trend over)",
                len(history),
            )
            return []

        tag: str = f"__{suffix}" if suffix else ""

        try:
            paths: list[Path] = [
                plotter.accuracy_vs_iteration(
                    history, name=f"accuracy_vs_iteration{tag}"
                ),
                plotter.sensitivity_vs_iteration(
                    history, name=f"sensitivity_vs_iteration{tag}"
                ),
                plotter.worst_gap_vs_iteration(
                    history, name=f"worst_gap_vs_iteration{tag}"
                ),
                plotter.slack_mass_vs_iteration(
                    history, name=f"slack_mass_vs_iteration{tag}"
                ),
                plotter.cuts_vs_iteration(history, name=f"cuts_vs_iteration{tag}"),
                plotter.pareto_curve(history, name=f"pareto_curve{tag}"),
                plotter.training_dashboard(history, name=f"training_dashboard{tag}"),
            ]
        except Exception:
            log.warning(
                "failed to plot training history%s -- continuing without it",
                f" for {suffix}" if suffix else "",
                exc_info=True,
            )
            return []

        log.info("wrote %d training plot(s) to %s", len(paths), plotter.run_dir)
        return paths

    def _git_sha(self) -> str:
        """
        Current git HEAD SHA, or "unknown" if it can't be determined.

        Returns:
            str: The current git HEAD SHA, or `"unknown"`.
        """

        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
        except Exception:
            return "unknown"

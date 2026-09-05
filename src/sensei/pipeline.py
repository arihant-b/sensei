import logging
import subprocess
from pathlib import Path

from xgboost import Booster

from sensei.config import Settings, ensense_pin
from sensei.data.bins import FrozenBins
from sensei.data.loader import Dataset
from sensei.eval.heldout_verify import HeldoutResult, HeldoutVerifier
from sensei.eval.region_overlap import RegionOverlapAnalyzer
from sensei.eval.sweeps import Sweeper, SweepPoint
from sensei.model.leaves import LeafMap
from sensei.model.train import Trainer
from sensei.oracle.milp.encoding import TreeEncoder, TreeStructure
from sensei.oracle.milp.solve import TierAOracle
from sensei.oracle.types import Pair
from sensei.repair.loop import (
    CegsalLoop,
    ConditionalCertificate,
    Inconclusive,
    ParetoBest,
)
from sensei.repair.pareto import Snapshot
from sensei.spec import Spec, load_spec

log: logging.Logger = logging.getLogger("sensei.pipeline")

_DATASET_ROOT: Path = Path(__file__).resolve().parents[2] / "dataset"


class CertifyPipeline:
    """
    The CertifyPipeline orchestrates the entire process of loading a dataset, training a
    model, repairing it using the CEGSAL loop, and performing various verification and
    sweep analyses.
    """

    def run(
        self, dataset: str, feature: str, direction: str, settings: Settings
    ) -> dict:
        """
        Run the certification pipeline.

        Args:
            dataset (str): The name of the dataset to load and process.
            feature (str): The name of the feature to analyze.
            direction (str): The direction of the analysis.
            settings (Settings): The settings for the certification pipeline.

        Returns:
            dict: A dictionary containing the results of the certification process,
                  including the exit reason, snapshot details, and results of various
                  verification and sweep analyses.
        """

        spec: Spec = load_spec(dataset)
        ds: Dataset = Dataset(
            dataset,
            eval_holdout=settings.dataset.eval_holdout,
            seed=settings.seeds.data_split,
        ).load()

        log.info(
            "training M0: %d trees, depth %d",
            settings.model.n_estimators,
            settings.model.max_depth,
        )

        assert ds.X_train is not None and ds.y_train is not None
        assert ds.columns is not None and ds.feature_bounds is not None

        booster: Booster = Trainer.train_baseline(
            ds.X_train,
            ds.y_train,
            settings.model.n_estimators,
            settings.model.max_depth,
            settings.seeds.model_train,
        )

        log.info("--- CEGSAL: repairing '%s' ---", feature)
        result: ConditionalCertificate | ParetoBest | Inconclusive = CegsalLoop(
            booster,
            ds.X_train,
            ds.y_train,
            ds.X_test,
            ds.y_test,
            ds.columns,
            ds.feature_bounds,
            spec,
            settings.seeds.model_train,
        ).run(
            flip_set=(feature,),
            direction=direction,
            eps=settings.sensitivity.eps,
            theta=settings.sensitivity.theta,
            mu=settings.repair.mu,
            kap=settings.repair.kap,
            max_iters=settings.loop.max_iters,
            a_min=settings.loop.A_min,
            stall_delta=settings.loop.stall_delta,
            cuts_per_round=settings.loop.cuts_per_round,
            oracle_time_limit_s=settings.oracle.time_limit_s,
            oracle_mip_gap=settings.oracle.mip_gap,
            n_quantile_bins=settings.bins.n_quantile_bins,
        )

        exit_kind: str = type(result).__name__
        log.info("CEGSAL exit: %s", exit_kind)

        payload: dict = {
            "git_sha": self._git_sha(),
            "ensense_pin": ensense_pin(),
            "stage": "stage3_certify",
            "dataset": dataset,
            "feature": feature,
            "direction": direction,
            "spec_hash": spec.spec_hash,
            "eps": settings.sensitivity.eps,
            "theta": settings.sensitivity.theta,
            "mu": settings.repair.mu,
            "kap": settings.repair.kap,
            "seeds": {
                "data_split": settings.seeds.data_split,
                "model_train": settings.seeds.model_train,
            },
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
        repaired_booster: Booster = leaf_map.write_leaf_values(booster, snapshot.v)
        structure: TreeStructure = TreeEncoder.extract_tree_structure(
            booster, leaf_map.leaf_index, ds.columns
        )
        bins: FrozenBins = FrozenBins(settings.bins.n_quantile_bins).fit(
            ds.X_train, ds.columns
        )
        details_csv: Path = _DATASET_ROOT / dataset / "details.csv"

        log.info("--- held-out verification (Ensense core) ---")
        hv: HeldoutResult = HeldoutVerifier.verify(
            repaired_booster,
            ds.columns,
            (feature,),
            details_csv=str(details_csv) if details_csv.exists() else None,
            output_gap=(settings.sensitivity.gap, 1.0 - settings.sensitivity.gap),
            timeout=int(settings.oracle.time_limit_s),
        )
        payload["heldout_verify"] = {"generalized": hv.generalized, "note": hv.note}

        if hv.fresh_pair is not None:
            payload["heldout_verify"]["fresh_gap"] = hv.fresh_pair.gap

        if hv.fresh_pair is not None:
            rate: float = RegionOverlapAnalyzer.overlap_rate(
                booster, leaf_map, [hv.fresh_pair], ds.columns, snapshot.cuts
            )
            payload["region_overlap_rate"] = rate
            log.info("region overlap rate: %.2f", rate)

        log.info("--- Tier A self-check at the repaired v ---")
        tier_a_pair: Pair | None = TierAOracle().worst_valid_pair(
            booster,
            leaf_map,
            ds.columns,
            ds.feature_bounds,
            spec,
            (feature,),
            direction,
            mode="feasibility",
            eps=settings.sensitivity.eps,
            seed=settings.seeds.model_train,
            time_limit_s=settings.oracle.time_limit_s,
            mip_gap=0.0,
            enforce_validity=True,
            structure=structure,
            v=snapshot.v,
            bins=bins,
            theta=settings.sensitivity.theta,
        )
        payload["tier_a_vs_tier_b"] = {
            "tier_a_unsat": tier_a_pair is None,
            "tier_b_unsat": hv.generalized,
            "note": (
                "Tier A UNSAT means absence within our Q1+Q2-encoded domain; Tier B "
                "UNSAT means absence under Ensense core's own search, which does not "
                "enforce our declared validity/plausibility rules inside its search at "
                "all."
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
            (feature,),
            direction,
            eps=settings.sensitivity.eps,
            thetas=thetas,
            seed=settings.seeds.model_train,
            time_limit_s=settings.oracle.time_limit_s,
            mip_gap=settings.oracle.mip_gap,
            v=snapshot.v,
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
            (feature,),
            direction,
            theta=settings.sensitivity.theta,
            epsilons=epsilons,
            seed=settings.seeds.model_train,
            time_limit_s=settings.oracle.time_limit_s,
            mip_gap=settings.oracle.mip_gap,
            v=snapshot.v,
            structure=structure,
        )
        payload["eps_sweep"] = [
            {"eps": p.value, "certified": p.certified, "gap": p.gap, "status": p.status}
            for p in eps_points
        ]
        for p in eps_points:
            log.info("  eps=%-6.2f certified=%s gap=%s", p.value, p.certified, p.gap)

        return payload

    def _git_sha(self) -> str:
        """
        Get the current Git SHA of the repository.

        Returns:
            str: The current Git SHA, or "unknown" if it cannot be determined.
        """

        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
        except Exception:
            return "unknown"

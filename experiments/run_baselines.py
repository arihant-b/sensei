import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from sensei.config import Settings, ensense_pin, load_defaults
from sensei.data.bins import FrozenBins
from sensei.data.loader import Dataset
from sensei.eval.metrics import Metrics
from sensei.eval.plots import ResultsPlotter
from sensei.eval.results_writer import ResultsWriter
from sensei.logging_setup import setup_logging
from sensei.model.baselines import BaselineResult, Baselines
from sensei.model.leaves import LeafMap
from sensei.oracle.ensense.adapter import EnsenseOracle
from sensei.oracle.types import OracleDegenerate, OracleSaturated, Pair
from sensei.spec import Spec, load_spec

log: logging.Logger = logging.getLogger("sensei.experiments.run_baselines")


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run(dataset: str) -> dict:
    settings: Settings = load_defaults()
    spec: Spec = load_spec(dataset)

    ds: Dataset = Dataset(
        dataset,
        eval_holdout=settings.dataset.eval_holdout,
        seed=settings.seed,
    ).load()

    assert ds.X_train is not None and ds.y_train is not None
    assert ds.columns is not None and ds.feature_bounds is not None

    n_estimators, max_depth = settings.model.n_estimators, settings.model.max_depth
    result: dict[str, Any] = {
        "git_sha": _git_sha(),
        "ensense_pin": ensense_pin(),
        "dataset": dataset,
        "spec_hash": spec.spec_hash,
        "metrics_version": Metrics.VERSION,
        "seed": settings.seed,
        "baselines": {},
    }

    def _record(name: str, booster, columns, policy: str | None = None) -> None:
        leaf_map = LeafMap(booster)
        protected_in_cols: tuple[str, ...] = tuple(
            p for p in spec.protected if p in columns
        )

        assert ds.X_test is not None and ds.y_test is not None

        acc: float = Metrics.accuracy(booster, leaf_map, ds.X_test[columns], ds.y_test)
        sens: float = Metrics.sensitivity_rate(
            booster,
            leaf_map,
            ds.X_test[columns],
            protected_in_cols,
            seed=settings.seed,
        )
        entry: dict[str, float | str] = {"accuracy": acc, "sensitivity_rate": sens}

        if policy is not None:
            entry["policy"] = policy

        result["baselines"][name] = entry
        log.info(
            f"{name:28s} acc={acc:.4f} sens={sens:.4f}"
            + (f" policy={policy}" if policy else "")
        )

    plain: BaselineResult = Baselines.fit_plain(
        ds.X_train, ds.y_train, n_estimators, max_depth, settings.seed
    )
    _record("1_plain", plain.booster, plain.columns)

    if spec.protected:
        dropped = Baselines.fit_dropped(
            ds.X_train,
            ds.y_train,
            spec,
            n_estimators,
            max_depth,
            settings.seed,
        )
        _record("2_protected_dropped", dropped.booster, dropped.columns)
    else:
        log.info(
            "2_protected_dropped: skipped, spec.protected is empty for this dataset"
        )

    if spec.monotone:
        monotone = Baselines.fit_monotone(
            ds.X_train,
            ds.y_train,
            spec,
            n_estimators,
            max_depth,
            settings.seed,
        )
        _record("3_monotone_constraints", monotone.booster, monotone.columns)
    else:
        log.info(
            "3_monotone_constraints: skipped, spec.monotone is empty for this dataset"
        )

    if spec.protected:
        m0_leaf_map = LeafMap(plain.booster)
        bins: FrozenBins = FrozenBins.fit_or_load(
            dataset,
            settings.seed,
            settings.bins.n_quantile_bins,
            ds.X_train,
            ds.columns,
            ds.categorical_levels,
        )
        pairs: list[Pair] = []
        t0: float = time.perf_counter()

        for feature in spec.protected:
            try:
                pair: Pair | None = EnsenseOracle().worst_valid_pair(
                    plain.booster,
                    m0_leaf_map,
                    plain.columns,
                    (feature,),
                    spec,
                    bins,
                    ds.feature_bounds,
                    settings,
                )
            except (OracleDegenerate, OracleSaturated) as e:
                log.info(
                    f"baseline 4: ensense oracle rejected/saturated for '{feature}', "
                    f"not used for retraining ({e})"
                )
                continue

            if pair is not None:
                pairs.append(pair)

        elapsed: float = time.perf_counter() - t0
        result["baseline4_pair_search_seconds"] = elapsed
        result["baseline4_pairs_found"] = len(pairs)
        log.info(
            f"baseline 4: found {len(pairs)} pair(s) via the ensense oracle in "
            f"{elapsed:.1f}s"
        )

        if pairs:
            for policy in ("P1", "P2"):
                retrained = Baselines.fit_counterexample_retrained(
                    ds.X_train,
                    ds.y_train,
                    pairs,
                    plain.columns,
                    spec,
                    plain.booster,
                    m0_leaf_map,
                    policy=policy,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    seed=settings.seed,
                )
                _record(
                    f"4_counterexample_retrained_{policy}",
                    retrained.booster,
                    retrained.columns,
                    policy,
                )
        else:
            log.info(
                "4_counterexample_retrained: no pairs found (the ensense oracle "
                "returned None for every protected feature -- this could be a "
                "genuine 'insensitive at this gap' result OR an unlabeled timeout; "
                "not distinguishable as shipped). Skipped, not faked."
            )
    else:
        log.info(
            "4_counterexample_retrained: skipped, spec.protected is empty for this "
            "dataset"
        )

    plotter = ResultsPlotter(label=f"{dataset}_baselines")
    result["plots_dir"] = str(plotter.run_dir)

    try:
        plot_path: Path = plotter.baseline_comparison(result["baselines"])
        result["baseline_comparison_plot"] = str(plot_path)
    except Exception:
        log.warning("failed to plot baseline comparison -- continuing without it",
                     exc_info=True)

    ResultsWriter.write(result, f"baselines_{dataset}")
    return result


def main() -> None:
    setup_logging("run_baselines")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from sensei.config import Settings, ensense_pin, load_defaults
from sensei.data.loader import Dataset
from sensei.eval.baselines import BaselineResult, Baselines
from sensei.eval.metrics import Metrics
from sensei.model.leaves import LeafMap
from sensei.oracle.ensense_adapter import TierBOracle
from sensei.oracle.types import Pair
from sensei.spec import Spec, load_spec

_RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results"
_DATASET_ROOT: Path = Path(__file__).resolve().parents[1] / "dataset"


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
        seed=settings.seeds.data_split,
    ).load()

    assert ds.X_train is not None and ds.y_train is not None

    n_estimators, max_depth = settings.model.n_estimators, settings.model.max_depth
    result: dict[str, Any] = {
        "git_sha": _git_sha(),
        "ensense_pin": ensense_pin(),
        "stage": "stage0_baselines",
        "dataset": dataset,
        "spec_hash": spec.spec_hash,
        "metrics_version": Metrics.VERSION,
        "seeds": {
            "data_split": settings.seeds.data_split,
            "model_train": settings.seeds.model_train,
        },
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
            seed=settings.seeds.model_train,
        )
        entry: dict[str, float | str] = {"accuracy": acc, "sensitivity_rate": sens}

        if policy is not None:
            entry["policy"] = policy

        result["baselines"][name] = entry
        print(
            f"{name:28s} acc={acc:.4f} sens={sens:.4f}"
            + (f" policy={policy}" if policy else "")
        )

    plain: BaselineResult = Baselines.fit_plain(
        ds.X_train, ds.y_train, n_estimators, max_depth, settings.seeds.model_train
    )
    _record("1_plain", plain.booster, plain.columns)

    if spec.protected:
        dropped = Baselines.fit_dropped(
            ds.X_train,
            ds.y_train,
            spec,
            n_estimators,
            max_depth,
            settings.seeds.model_train,
        )
        _record("2_protected_dropped", dropped.booster, dropped.columns)
    else:
        print("2_protected_dropped: skipped, spec.protected is empty for this dataset")

    if spec.monotone:
        monotone = Baselines.fit_monotone(
            ds.X_train,
            ds.y_train,
            spec,
            n_estimators,
            max_depth,
            settings.seeds.model_train,
        )
        _record("3_monotone_constraints", monotone.booster, monotone.columns)
    else:
        print(
            "3_monotone_constraints: skipped, spec.monotone is empty for this dataset"
        )

    if spec.protected:
        m0_leaf_map = LeafMap(plain.booster)
        details_csv: Path = _DATASET_ROOT / dataset / "details.csv"
        pairs: list[Pair] = []
        t0: float = time.perf_counter()

        for feature in spec.protected:
            pair: Pair | None = TierBOracle().worst_valid_pair(
                plain.booster,
                m0_leaf_map,
                plain.columns,
                (feature,),
                method="pb",
                details_csv=str(details_csv) if details_csv.exists() else None,
                output_gap=(settings.sensitivity.gap, 1.0 - settings.sensitivity.gap),
                timeout=int(settings.oracle.time_limit_s),
            )
            if pair is not None:
                pairs.append(pair)

        elapsed: float = time.perf_counter() - t0
        result["baseline4_pair_search_seconds"] = elapsed
        result["baseline4_pairs_found"] = len(pairs)
        print(f"baseline 4: found {len(pairs)} pair(s) via Tier B in {elapsed:.1f}s")

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
                    seed=settings.seeds.baseline4_retrain,
                )
                _record(
                    f"4_counterexample_retrained_{policy}",
                    retrained.booster,
                    retrained.columns,
                    policy,
                )
        else:
            print(
                "4_counterexample_retrained: no pairs found (Tier B returned None for "
                "every protected feature -- per docs/ensense_interface.md this could be"
                " a genuine 'insensitive at this gap' result OR an unlabeled timeout; "
                "not distinguishable as shipped). Skipped, not faked."
            )
    else:
        print(
            "4_counterexample_retrained: skipped, spec.protected is empty for this "
            "dataset"
        )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path: Path = (
        _RESULTS_DIR / f"stage0_baselines_{dataset}_{int(time.time())}.json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()

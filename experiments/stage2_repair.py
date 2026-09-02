"""
Stage 2 -- repair (the main result).

Runs the real SensEI loop across datasets/seeds and reports worst gap,
accuracy, cost, and whether the run certified. Report across at least
three seeds (README).

    python -m experiments.stage2_repair --all --seeds 42,43,44
    python -m experiments.stage2_repair --dataset adult \
        --verify-generalisation --seed 43
"""

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data import Dataset  # noqa: E402
from experiments._common import (  # noqa: E402
    ALL_DATASETS,
    build_config,
    parse_seeds,
    resolve_datasets,
)
from loop import SensEILoop  # noqa: E402
from metrics import accuracy  # noqa: E402
from model import Ensemble  # noqa: E402
from oracle import SensitivityOracle  # noqa: E402
from runlog import RunWriter  # noqa: E402
from sampler import SamplingScreen  # noqa: E402
from spec import SPECS  # noqa: E402
from validity import ValidityChecker  # noqa: E402

log = logging.getLogger("experiments.stage2")


def run_one(
    dataset: str, seed: int,
    n_estimators: int | None = None, max_depth: int | None = None,
    verify_generalisation: bool = False,
) -> dict:
    cfg = build_config(dataset, seed, n_estimators, max_depth)
    spec = SPECS[dataset]

    data = Dataset(cfg)
    data.load()
    assert data.X_train is not None and data.y_train is not None
    assert data.X_test is not None and data.y_test is not None
    bins = data.build_bins()
    model = Ensemble(cfg).fit(data.X_train, data.y_train)
    baseline_acc = accuracy(model, data.X_test, data.y_test)

    validity = ValidityChecker(spec, bins, cfg, data.feature_bounds)
    oracle = SensitivityOracle(validity, spec, cfg)
    sampler = SamplingScreen(spec, validity, cfg)
    loop = SensEILoop(model, data, oracle, sampler, cfg)

    per_feature = {}
    for feature in spec.protected:
        log.info("[%s seed=%d] repairing '%s'", dataset, seed, feature)
        best = loop.run(flip_set=[feature])

        entry = {
            "worst_gap": best.worst_gap if best else None,
            "accuracy": best.accuracy if best else None,
            "cost": (baseline_acc - best.accuracy) if best else None,
            "certified": loop.certified,
        }

        if verify_generalisation:
            ce = loop.verify_generalization(flip_set=[feature], seed=seed + 1)
            entry["generalises"] = ce is None

        per_feature[feature] = entry

    return {
        "dataset": dataset, "seed": seed,
        "baseline_accuracy": baseline_acc,
        "certified": loop.certified,
        "per_feature": per_feature,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 2: repair (main result)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", type=str, choices=ALL_DATASETS)
    p.add_argument("--seeds", type=str, default="42",
                    help="comma-separated seeds, e.g. 42,43,44")
    p.add_argument("--seed", type=int, default=None,
                    help="single-seed shortcut, overrides --seeds")
    p.add_argument("--verify-generalisation", action="store_true")
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets = resolve_datasets(args)
    seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)

    for dataset in datasets:
        for seed in seeds:
            results = run_one(
                dataset, seed, args.n_estimators, args.max_depth,
                verify_generalisation=args.verify_generalisation)
            log.info("[%s seed=%d] certified=%s", dataset, seed, results["certified"])

            writer = RunWriter(tag=f"stage2-{dataset}-seed{seed}")
            writer.write_summary(results)
            writer.write_environment()
            log.info("  -> %s", writer.dir)


if __name__ == "__main__":
    main()

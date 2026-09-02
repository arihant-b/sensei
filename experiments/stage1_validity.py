"""
Stage 1 -- validity ablation.

Measures how many sampling-screen candidates Q1/Q2 (ValidityChecker)
eliminate versus a permissive stand-in that accepts everything -- the
data behind README's "Validity comparison" figure (malformed/implausible
values eliminated).

    python -m experiments.stage1_validity --dataset adult
    python -m experiments.stage1_validity --all
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
    resolve_datasets,
)
from model import Ensemble  # noqa: E402
from runlog import RunWriter  # noqa: E402
from sampler import SamplingScreen  # noqa: E402
from spec import SPECS  # noqa: E402
from validity import ValidityChecker  # noqa: E402

log = logging.getLogger("experiments.stage1")


class _PermissiveValidity:
    """Accepts every candidate -- the 'no Q1/Q2 gating' arm of the ablation."""

    def is_valid_pair(self, x1, x2) -> bool:
        return True


def run_dataset(
    dataset: str, seed: int,
    n_estimators: int | None = None, max_depth: int | None = None,
) -> dict:
    cfg = build_config(dataset, seed, n_estimators, max_depth)
    spec = SPECS[dataset]

    data = Dataset(cfg)
    data.load()
    assert data.X_train is not None and data.y_train is not None
    bins = data.build_bins()
    model = Ensemble(cfg).fit(data.X_train, data.y_train)

    validity = ValidityChecker(spec, bins, cfg, data.feature_bounds)
    gated_sampler = SamplingScreen(spec, validity, cfg)
    ungated_sampler = SamplingScreen(spec, _PermissiveValidity(), cfg)

    results: dict = {"dataset": dataset, "seed": seed, "per_feature": {}}

    for feature in spec.protected:
        gated = gated_sampler.find_violations(
            model, data.X_train, feature, k=cfg.cuts_per_round)
        ungated = ungated_sampler.find_violations(
            model, data.X_train, feature, k=cfg.cuts_per_round)

        results["per_feature"][feature] = {
            "n_found_with_validity": len(gated),
            "n_found_without_validity": len(ungated),
            "eliminated_by_validity": len(ungated) - len(gated),
        }

    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 1: validity ablation")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", type=str, choices=ALL_DATASETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets = resolve_datasets(args)

    for dataset in datasets:
        log.info("=== stage1 validity ablation: %s ===", dataset)
        results = run_dataset(dataset, args.seed, args.n_estimators, args.max_depth)
        for feature, counts in results["per_feature"].items():
            log.info("  %-20s %s", feature, counts)

        writer = RunWriter(tag=f"stage1-{dataset}")
        writer.write_summary(results)
        writer.write_environment()
        log.info("  -> %s", writer.dir)


if __name__ == "__main__":
    main()

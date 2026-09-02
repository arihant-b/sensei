"""
Stage 3 -- train-vs-repair comparison.

Same dataset, same final accuracy metric, two different interventions:
training a model differently from the start (reweighing) versus taking
a plainly-trained model and repairing it after the fact (SensEI). Only
the latter is data-aware and produces a certificate.

    python -m experiments.stage3_training --dataset adult
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
from experiments.stage0_baselines import (  # noqa: E402
    _ClassifierAdapter,
    _fit_reweighed,
)
from loop import SensEILoop  # noqa: E402
from metrics import accuracy, sensitivity_rate  # noqa: E402
from model import Ensemble  # noqa: E402
from oracle import SensitivityOracle  # noqa: E402
from runlog import RunWriter  # noqa: E402
from sampler import SamplingScreen  # noqa: E402
from spec import SPECS  # noqa: E402
from validity import ValidityChecker  # noqa: E402

log = logging.getLogger("experiments.stage3")


def run_dataset(
    dataset: str, seed: int,
    n_estimators: int | None = None, max_depth: int | None = None,
) -> dict:
    cfg = build_config(dataset, seed, n_estimators, max_depth)
    spec = SPECS[dataset]

    if not spec.protected:
        log.info("%s has no protected feature declared; skipping", dataset)
        return {"dataset": dataset, "seed": seed, "skipped": True}

    data = Dataset(cfg)
    data.load()
    assert data.X_train is not None and data.y_train is not None
    assert data.X_test is not None and data.y_test is not None
    bins = data.build_bins()
    validity = ValidityChecker(spec, bins, cfg, data.feature_bounds)
    sampler = SamplingScreen(spec, validity, cfg)
    feature = spec.protected[0]

    # --- train differently: reweighing baked in from the start -----------
    rw_clf = _fit_reweighed(cfg, data.X_train, data.y_train, spec.protected)
    rw_model = _ClassifierAdapter(rw_clf)
    trained_differently = {
        "accuracy": accuracy(rw_model, data.X_test, data.y_test),
        "sensitivity_rate": sensitivity_rate(rw_model, data.X_test, spec, sampler),
        "certified": False,
    }

    # --- repair after the fact: plain model, then SensEI ------------------
    plain = Ensemble(cfg).fit(data.X_train, data.y_train)
    baseline_acc = accuracy(plain, data.X_test, data.y_test)
    oracle = SensitivityOracle(validity, spec, cfg)
    loop = SensEILoop(plain, data, oracle, sampler, cfg)
    best = loop.run(flip_set=[feature])

    repaired_after_fact = {
        "accuracy": best.accuracy if best else baseline_acc,
        "sensitivity_rate": best.sensitivity_rate if best else None,
        "certified": loop.certified,
    }

    return {
        "dataset": dataset, "seed": seed, "feature": feature,
        "trained_differently_reweighing": trained_differently,
        "repaired_after_fact_sensei": repaired_after_fact,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 3: train-vs-repair comparison")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", type=str, choices=ALL_DATASETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets = resolve_datasets(args)

    for dataset in datasets:
        log.info("=== stage3 train-vs-repair: %s ===", dataset)
        results = run_dataset(dataset, args.seed, args.n_estimators, args.max_depth)
        if results.get("skipped"):
            continue

        log.info("  trained differently (reweighing): %s",
                  results["trained_differently_reweighing"])
        log.info("  repaired after the fact (SensEI):  %s",
                  results["repaired_after_fact_sensei"])

        writer = RunWriter(tag=f"stage3-{dataset}")
        writer.write_summary(results)
        writer.write_environment()
        log.info("  -> %s", writer.dir)


if __name__ == "__main__":
    main()

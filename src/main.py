import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from config import Config, load_config
from data import Dataset, FrozenBins
from loop import SensEILoop
from metrics import accuracy
from model import Ensemble
from oracle import SensitivityOracle
from runlog import RunWriter
from sampler import SamplingScreen
from spec import SPECS, SensitivitySpec, load_spec
from validity import ValidityChecker

log: logging.Logger = logging.getLogger("sensei")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="sensei", description="Certified sensitivity repair")
    p.add_argument("--config", type=str, default=None,
                    help="path to a configs/*.yaml file")
    p.add_argument("--smoke", action="store_true",
                    help="run the full loop on a 3-tree synthetic model with a "
                         "planted violation instead of a real dataset")
    p.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--dump-cuts", action="store_true",
                    help="write every cut's leaf indices/coefficients to "
                         "runs/<id>/cuts.jsonl")
    return p.parse_args(argv)


def _run_smoke(cfg: Config, writer: RunWriter, dump_cuts: bool) -> bool:
    """
    Full loop on a 3-tree synthetic model with a known planted violation
    and a FakeOracle (no Gurobi MILP search, no sampling screen) --
    should certify in well under 10s. See tests/fixtures/synthetic.py.
    """
    _tests_dir = Path(__file__).resolve().parents[1] / "tests"
    if str(_tests_dir) not in sys.path:
        sys.path.insert(0, str(_tests_dir))
    from fixtures.synthetic import (
        FakeOracle,
        make_planted_counterexample,
        make_synthetic_dataset,
        make_synthetic_model,
    )

    # independent copy: must not mutate the caller's cfg
    smoke_cfg: Config = replace(cfg, use_sampling_screen=False, min_accuracy=0.0)

    model = make_synthetic_model(seed=smoke_cfg.seed)
    X, y = make_synthetic_dataset(seed=smoke_cfg.seed)
    planted = make_planted_counterexample(model)

    spec = SensitivitySpec(
        protected=["protected"], monotone={}, immutable=[],
        integer_features=[], one_hot_groups={}, ranges={}, functional_deps=[],
    )

    data = Dataset(smoke_cfg)
    data.X_train = X
    data.y_train = y
    data.X_test = X
    data.y_test = y
    data.X_eval = X
    data.y_eval = y
    data.columns = list(X.columns)
    data.feature_bounds = {}

    bins: FrozenBins = data.build_bins()
    validity = ValidityChecker(spec, bins, smoke_cfg, data.feature_bounds)
    oracle = FakeOracle(planted)
    sampler = SamplingScreen(spec, validity, smoke_cfg)
    loop = SensEILoop(model, data, oracle, sampler, smoke_cfg)

    baseline_acc = accuracy(model, data.X_test, data.y_test)
    log.info("smoke: baseline accuracy %.4f", baseline_acc)

    best = loop.run(flip_set=["protected"])
    log.info("smoke: certified=%s", loop.certified)

    writer.write_config(smoke_cfg)
    writer.write_environment()
    if dump_cuts:
        writer.write_cuts(loop.qp.cuts)
    writer.write_snapshots(loop.checkpoint.history)
    if model.v is not None:
        writer.write_leaf_values(model.v)
    writer.write_summary({
        "mode": "smoke",
        "certified": loop.certified,
        "baseline_accuracy": baseline_acc,
        "best_worst_gap": best.worst_gap if best else None,
        "best_accuracy": best.accuracy if best else None,
    })

    return loop.certified


def main(argv: list[str] | None = None) -> None:
    args: argparse.Namespace = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    cfg: Config = load_config(args.config) if args.config else Config()
    writer = RunWriter(tag=cfg.dataset if not args.smoke else "smoke")
    log.info("run directory: %s", writer.dir)

    if args.smoke:
        _run_smoke(cfg, writer, args.dump_cuts)
        return

    spec = load_spec(args.config) if args.config else SPECS[cfg.dataset]

    # --- stage 0: data + baseline model ----------------------------------
    data: Dataset = Dataset(cfg)
    data.load()
    assert data.X_train is not None and data.y_train is not None
    assert data.X_test is not None and data.y_test is not None
    bins: FrozenBins = data.build_bins()                      # frozen for the whole run

    model: Ensemble = Ensemble(cfg).fit(data.X_train, data.y_train)
    baseline_acc: float = accuracy(model, data.X_test, data.y_test)
    log.info("baseline accuracy: %.4f", baseline_acc)

    # --- stage 1: validity ------------------------------------------------
    validity = ValidityChecker(spec, bins, cfg, data.feature_bounds)

    # --- stage 2: repair loop --------------------------------------------
    oracle = SensitivityOracle(validity, spec, cfg)
    sampler = SamplingScreen(spec, validity, cfg)
    loop = SensEILoop(model, data, oracle, sampler, cfg)
    per_feature = {}

    for feature in spec.protected:
        log.info("--- repairing sensitivity to '%s' ---", feature)
        best = loop.run(flip_set=[feature])

        if best:
            log.info("worst gap %.4f | accuracy %.4f | cost %.4f",
                     best.worst_gap, best.accuracy,
                     baseline_acc - best.accuracy)
            per_feature[feature] = {
                "worst_gap": best.worst_gap, "accuracy": best.accuracy,
                "cost": baseline_acc - best.accuracy,
            }

    log.info("certified: %s", loop.certified)

    # --- held-out verification -------------------------------------------
    for feature in spec.protected:
        loop.verify_generalization(flip_set=[feature], seed=cfg.seed + 1)

    # --- runs/ output -------------------------------------------------------
    writer.write_config(cfg)
    writer.write_environment()

    if args.dump_cuts:
        writer.write_cuts(loop.qp.cuts)

    writer.write_snapshots(loop.checkpoint.history)

    if model.v is not None:
        writer.write_leaf_values(model.v)
    
    writer.write_summary({
        "mode": "run",
        "dataset": cfg.dataset,
        "certified": loop.certified,
        "baseline_accuracy": baseline_acc,
        "per_feature": per_feature,
    })


if __name__ == "__main__":
    main()

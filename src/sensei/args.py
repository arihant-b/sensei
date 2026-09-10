import argparse
from dataclasses import dataclass

from sensei.config import (
    BinsConfig,
    DatasetConfig,
    LoopConfig,
    ModelConfig,
    OracleConfig,
    RepairConfig,
    SensitivityConfig,
    Settings,
    load_defaults,
)
from sensei.spec import Spec, load_spec


@dataclass(frozen=True)
class CliArgs:
    """The `sensei` command's fully-parsed and validated CLI arguments."""

    dataset: str
    feature: str
    direction: str
    settings: Settings


def build_parser() -> argparse.ArgumentParser:
    """
    Build the `sensei` CLI parser -- one flag per `config/defaults.yaml`
    field, its `default=` sourced FROM that file (via `load_defaults()`),
    never a hardcoded literal duplicated here -- so a change to
    defaults.yaml is reflected in `sensei --help` and in what a bare
    `sensei --dataset ... --feature ...` call actually runs, without
    needing a matching edit in this file too.

    Returns:
        argparse.ArgumentParser: The fully configured `sensei` CLI parser.
    """

    defaults: Settings = load_defaults()

    parser = argparse.ArgumentParser(
        prog="sensei",
        description=(
            "Repair sensitivity to a protected feature (or a wrong-direction "
            "monotone violation) in a trained XGBoost ensemble via CEGSAL "
            "(search -> cut -> repair -> repeat), independently re-check the "
            "result against the vendored Ensense core, and write one results "
            "JSON."
        ),
    )

    required: argparse._ArgumentGroup = parser.add_argument_group("required")
    required.add_argument(
        "--dataset",
        required=True,
        help="dataset name, matching dataset/<name>/ and sensei/spec/<name>.yaml",
    )
    required.add_argument(
        "--feature",
        required=True,
        help="feature to repair sensitivity to (must be in the spec's protected or "
        "monotone group)",
    )

    parser.add_argument(
        "--direction",
        default="protected",
        choices=["protected", "monotone_wrong"],
        help="'protected' (default) for a protected-feature flip, or 'monotone_wrong' "
        "for a wrong-direction monotone violation",
    )
    parser.add_argument(
        "--results-dir",
        default=defaults.results_dir,
        help="directory results JSON files are written into",
    )

    dataset_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "dataset (config/defaults.yaml: dataset)"
    )
    dataset_grp.add_argument(
        "--test-size",
        type=float,
        default=defaults.dataset.test_size,
        help="NO EFFECT on this command -- dataset/<name>/{train,test}.csv are "
        "already built and this command only reads them. Controls the "
        "train/test split only when BUILDING a dataset, via "
        "`python -m sensei.data.builder --test-size`. Kept here to fill "
        "Settings.dataset's schema, sourced from defaults.yaml otherwise.",
    )
    dataset_grp.add_argument(
        "--eval-holdout",
        type=float,
        default=defaults.dataset.eval_holdout,
        help="D_eval fraction, sealed after the split",
    )

    model_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "model (config/defaults.yaml: model)"
    )
    model_grp.add_argument(
        "--n-estimators", type=int, default=defaults.model.n_estimators
    )
    model_grp.add_argument("--max-depth", type=int, default=defaults.model.max_depth)

    sens_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "sensitivity (config/defaults.yaml: sensitivity)"
    )
    sens_grp.add_argument(
        "--eps",
        type=float,
        default=defaults.sensitivity.eps,
        help="sensitivity budget, margin space",
    )
    sens_grp.add_argument(
        "--theta",
        type=float,
        default=defaults.sensitivity.theta,
        help="plausibility threshold",
    )
    sens_grp.add_argument(
        "--gap",
        type=float,
        default=defaults.sensitivity.gap,
        help="confident-flip margin, probability space (Ensense's --output_gap "
        "convention)",
    )

    repair_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "repair (config/defaults.yaml: repair)"
    )
    repair_grp.add_argument(
        "--mu",
        type=float,
        default=defaults.repair.mu,
        help="proximal weight on ||v - v0||^2",
    )
    repair_grp.add_argument(
        "--kap", type=float, default=defaults.repair.kap, help="slack price, sum(s_i)"
    )
    repair_grp.add_argument(
        "--type",
        choices=["exact", "approx"],
        default=defaults.repair.type,
        help="repair QP type: exact or approx",
    )

    loop_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "loop (config/defaults.yaml: loop)"
    )
    loop_grp.add_argument(
        "--max-iters", type=int, default=defaults.loop.max_iters
    )
    loop_grp.add_argument(
        "--a-min", type=float, default=defaults.loop.A_min, help="accuracy floor"
    )
    loop_grp.add_argument(
        "--stall-delta", type=float, default=defaults.loop.stall_delta
    )
    loop_grp.add_argument(
        "--cuts-per-round", type=int, default=defaults.loop.cuts_per_round
    )

    oracle_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "oracle (config/defaults.yaml: oracle)"
    )
    oracle_grp.add_argument(
        "--oracle-time-limit-s", type=float, default=defaults.oracle.time_limit_s
    )
    oracle_grp.add_argument(
        "--oracle-mip-gap",
        type=float,
        default=defaults.oracle.mip_gap,
        help="sensei oracle intermediate-round tolerance only -- a certifying run "
        "always uses 0 internally",
    )
    oracle_grp.add_argument(
        "--oracle-type",
        choices=["sensei", "ensense"],
        default=defaults.oracle.type,
        help="which oracle drives the CEGSAL loop's own search: 'sensei' (own "
        "MILP -- the preferred architecture) or 'ensense' (Ensense core + "
        "postfilter -- weaker, no no-goods, no --direction monotone_wrong support). "
        "The sensei oracle self-check and held-out verification always run against "
        "Ensense core regardless of this flag.",
    )
    oracle_grp.add_argument(
        "--oracle-method",
        choices=["pb", "milp"],
        default=defaults.oracle.method,
        help="which Ensense core solver family to call whenever Ensense core is "
        "used (--oracle-type ensense, and always for held-out verification): "
        "'pb' (Pseudo-Boolean) or 'milp'",
    )

    bins_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "bins (config/defaults.yaml: bins)"
    )
    bins_grp.add_argument(
        "--n-quantile-bins",
        type=int,
        default=defaults.bins.n_quantile_bins,
        help="quantile bins per numeric feature (one bin per categorical level, "
        "always)",
    )

    seed_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "seed (config/defaults.yaml: seed)"
    )
    seed_grp.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="single global seed -- data split, model training, oracle search "
        "sampling, baseline retraining",
    )

    return parser


def _validate_feature(
    parser: argparse.ArgumentParser, dataset: str, feature: str, direction: str
) -> None:
    """
    Check `--feature` actually belongs to `dataset`'s declared protected/
    monotone group for the given `--direction`, before anything downstream
    runs. Without this, an immutable feature (or a typo'd name) would reach
    `CegsalLoop` unchecked and either crash deep in the oracle or, worse,
    silently run a search over a feature the spec never declared safe to
    flip at all. Raises via `parser.error` (clean usage message + exit code
    2), not a raw traceback.

    Args:
        parser (argparse.ArgumentParser): Used to report a usage error.
        dataset (str): Dataset name, matching `spec/<dataset>.yaml`.
        feature (str): Feature to validate against the spec.
        direction (str): `"protected"` or `"monotone_wrong"`.
    """

    try:
        spec: Spec = load_spec(dataset)
    except FileNotFoundError:
        parser.error(f"--dataset {dataset!r}: no spec/{dataset}.yaml found")
        return

    if direction == "protected":
        if feature not in spec.protected:
            parser.error(
                f"--feature {feature!r} is not declared protected in "
                f"spec/{dataset}.yaml (protected: {list(spec.protected)}) -- pass "
                f"--direction monotone_wrong if it's a monotone feature instead, "
                f"or add it to the spec"
            )
    else:
        assert direction == "monotone_wrong"
        if feature not in spec.monotone:
            parser.error(
                f"--feature {feature!r} is not declared monotone in "
                f"spec/{dataset}.yaml (monotone: {list(spec.monotone)}) -- pass "
                f"--direction protected if it's a protected feature instead, or "
                f"add it to the spec"
            )


def parse_args(argv: list[str] | None = None) -> CliArgs:
    """
    Parse and validate `argv` (`sys.argv` if None) into `CliArgs`.

    Args:
        argv (list[str] | None): Argument list to parse; `sys.argv` if None.

    Returns:
        CliArgs: The fully-parsed and validated CLI arguments.
    """

    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    _validate_feature(parser, args.dataset, args.feature, args.direction)

    if args.oracle_type == "ensense" and args.direction == "monotone_wrong":
        parser.error(
            "--oracle-type ensense does not support --direction monotone_wrong -- "
            "Ensense core's search has no way to pin which side of the pair is "
            "the lower/higher monotone side, and always returns "
            "direction='protected' pairs. Use --oracle-type sensei for a monotone "
            "feature instead."
        )

    settings = Settings(
        dataset=DatasetConfig(test_size=args.test_size, eval_holdout=args.eval_holdout),
        model=ModelConfig(n_estimators=args.n_estimators, max_depth=args.max_depth),
        sensitivity=SensitivityConfig(eps=args.eps, theta=args.theta, gap=args.gap),
        repair=RepairConfig(mu=args.mu, kap=args.kap, type=args.type),
        loop=LoopConfig(
            max_iters=args.max_iters,
            A_min=args.a_min,
            stall_delta=args.stall_delta,
            cuts_per_round=args.cuts_per_round,
        ),
        oracle=OracleConfig(
            time_limit_s=args.oracle_time_limit_s,
            mip_gap=args.oracle_mip_gap,
            type=args.oracle_type,
            method=args.oracle_method,
        ),
        bins=BinsConfig(n_quantile_bins=args.n_quantile_bins),
        results_dir=args.results_dir,
        seed=args.seed,
    )

    return CliArgs(
        dataset=args.dataset,
        feature=args.feature,
        direction=args.direction,
        settings=settings,
    )

import argparse
from dataclasses import dataclass

from sensei.config import (
    BinsConfig,
    DatasetConfig,
    LoopConfig,
    ModelConfig,
    OracleConfig,
    RepairConfig,
    Seeds,
    SensitivityConfig,
    Settings,
)


@dataclass(frozen=True)
class CliArgs:
    """
    The parsed command-line arguments for the Sensei repair process.
    """

    dataset: str
    feature: str
    direction: str
    settings: Settings


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns:
        argparse.ArgumentParser: The built argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="sensei",
        description=(
            "Repair sensitivity to a protected feature (or a wrong-direction "
            "monotone violation) in a trained XGBoost ensemble via CEGSAL "
            "(search -> cut -> repair -> repeat), independently re-check the "
            "result against the vendored Ensense core, and write one results "
            "JSON. See README.md's CLI section."
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
        default="results/",
        help="directory results JSON files are written into",
    )

    dataset_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "dataset (config/defaults.yaml: dataset)"
    )
    dataset_grp.add_argument("--test-size", type=float, default=0.2)
    dataset_grp.add_argument(
        "--eval-holdout",
        type=float,
        default=0.2,
        help="D_eval fraction, sealed after the split",
    )

    model_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "model (config/defaults.yaml: model)"
    )
    model_grp.add_argument("--n-estimators", type=int, default=200)
    model_grp.add_argument("--max-depth", type=int, default=5)

    sens_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "sensitivity (config/defaults.yaml: sensitivity)"
    )
    sens_grp.add_argument(
        "--eps", type=float, default=0.10, help="sensitivity budget, margin space"
    )
    sens_grp.add_argument(
        "--theta", type=float, default=1e-6, help="plausibility threshold"
    )
    sens_grp.add_argument(
        "--gap",
        type=float,
        default=0.50,
        help="confident-flip margin, probability space (Ensense's --output_gap "
        "convention)",
    )

    repair_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "repair (config/defaults.yaml: repair)"
    )
    repair_grp.add_argument(
        "--mu", type=float, default=0.01, help="proximal weight on ||v - v0||^2"
    )
    repair_grp.add_argument(
        "--kap", type=float, default=100.0, help="slack price, sum(s_i)"
    )

    loop_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "loop (config/defaults.yaml: loop)"
    )
    loop_grp.add_argument("--max-iters", type=int, default=50)
    loop_grp.add_argument(
        "--a-min", type=float, default=0.82, help="accuracy floor"
    )
    loop_grp.add_argument("--stall-delta", type=float, default=1e-3)
    loop_grp.add_argument("--cuts-per-round", type=int, default=5)

    oracle_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "oracle (config/defaults.yaml: oracle)"
    )
    oracle_grp.add_argument("--oracle-time-limit-s", type=float, default=300.0)
    oracle_grp.add_argument(
        "--oracle-mip-gap",
        type=float,
        default=0.05,
        help="Tier A intermediate-round tolerance only -- a certifying run always uses "
        "0 internally",
    )

    bins_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "bins (config/defaults.yaml: bins)"
    )
    bins_grp.add_argument(
        "--n-quantile-bins",
        type=int,
        default=10,
        help="quantile bins per numeric feature (one bin per categorical level, "
        "always)",
    )

    seeds_grp: argparse._ArgumentGroup = parser.add_argument_group(
        "seeds (config/defaults.yaml: seeds)"
    )
    seeds_grp.add_argument("--data-split-seed", type=int, default=42)
    seeds_grp.add_argument("--model-train-seed", type=int, default=42)
    seeds_grp.add_argument("--baseline4-retrain-seed", type=int, default=42)

    return parser


def parse_args(argv: list[str] | None = None) -> CliArgs:
    """
    Parse command-line arguments into a CliArgs object.

    Args:
        argv (list[str] | None, optional): The command-line arguments to parse. Defaults
                                           to None.

    Returns:
        CliArgs: The parsed command-line arguments as a CliArgs object.
    """

    args: argparse.Namespace = build_parser().parse_args(argv)

    settings = Settings(
        dataset=DatasetConfig(test_size=args.test_size, eval_holdout=args.eval_holdout),
        model=ModelConfig(n_estimators=args.n_estimators, max_depth=args.max_depth),
        sensitivity=SensitivityConfig(eps=args.eps, theta=args.theta, gap=args.gap),
        repair=RepairConfig(mu=args.mu, kap=args.kap),
        loop=LoopConfig(
            max_iters=args.max_iters,
            A_min=args.a_min,
            stall_delta=args.stall_delta,
            cuts_per_round=args.cuts_per_round,
        ),
        oracle=OracleConfig(
            time_limit_s=args.oracle_time_limit_s, mip_gap=args.oracle_mip_gap
        ),
        bins=BinsConfig(n_quantile_bins=args.n_quantile_bins),
        results_dir=args.results_dir,
        seeds=Seeds(
            data_split=args.data_split_seed,
            model_train=args.model_train_seed,
            baseline4_retrain=args.baseline4_retrain_seed,
        ),
    )

    return CliArgs(
        dataset=args.dataset,
        feature=args.feature,
        direction=args.direction,
        settings=settings,
    )

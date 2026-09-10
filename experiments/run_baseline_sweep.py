import argparse
import logging
from pathlib import Path

from run_baselines import run as run_baselines

from sensei.logging_setup import setup_logging
from sensei.spec import ALL_DATASETS

log: logging.Logger = logging.getLogger("sensei.experiments.run_baseline_sweep")

_DATASET_ROOT: Path = Path(__file__).resolve().parents[1] / "dataset"


def run(datasets: list[str] | None = None) -> dict[str, dict]:
    targets: list[str] = datasets if datasets is not None else _built_datasets()

    if not targets:
        log.warning(
            "no dataset under %s has a built train.csv -- see sensei/data/builder.py",
            _DATASET_ROOT,
        )
        return {}

    results: dict[str, dict] = {}

    for dataset in targets:
        log.info("=== baselines: %s ===", dataset)
        results[dataset] = run_baselines(dataset)

    return results


def _built_datasets() -> list[str]:
    return [d for d in ALL_DATASETS if (_DATASET_ROOT / d / "train.csv").is_file()]


def main() -> None:
    setup_logging("run_baseline_sweep")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="dataset names to run (default: every dataset under dataset/ with a built "
        "train.csv)",
    )
    args: argparse.Namespace = parser.parse_args()
    run(args.datasets)


if __name__ == "__main__":
    main()

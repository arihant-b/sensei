import json
import logging
import time
from pathlib import Path

from sensei.args import CliArgs, parse_args
from sensei.pipeline import CertifyPipeline

log: logging.Logger = logging.getLogger("sensei.cli")


def main(argv: list[str] | None = None) -> None:
    """
    Main entry point for the Sensei CLI.

    Args:
        argv (list[str] | None, optional): The command-line arguments to parse. Defaults
                                           to None.
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli_args: CliArgs = parse_args(argv)

    payload: dict = CertifyPipeline().run(
        cli_args.dataset, cli_args.feature, cli_args.direction, cli_args.settings
    )
    out_path: Path = _write(
        payload, cli_args.dataset, cli_args.feature, cli_args.settings.results_dir
    )
    print(f"wrote {out_path}")


def _write(payload: dict, dataset: str, feature: str, results_dir: str) -> Path:
    """
    Write the payload to a JSON file.

    Args:
        payload (dict): The payload to write to the JSON file.
        dataset (str): The name of the dataset.
        feature (str): The name of the feature.
        results_dir (str): The directory to write the results to.

    Returns:
        Path: The path to the written JSON file.
    """

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = (
        out_dir / f"stage3_certify_{dataset}_{feature}_{int(time.time())}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


if __name__ == "__main__":
    main()

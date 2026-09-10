import logging
from pathlib import Path

from sensei.args import CliArgs, parse_args
from sensei.eval.results_writer import ResultsWriter
from sensei.logging_setup import setup_logging
from sensei.pipeline import CertifyPipeline

log: logging.Logger = logging.getLogger("sensei.cli")


def main(argv: list[str] | None = None) -> None:
    """
    `sensei` console entry point: parse args -> run the pipeline -> write results.

    Args:
        argv (list[str] | None): Argument list to parse; `sys.argv` if None.
    """

    setup_logging("cli")
    cli_args: CliArgs = parse_args(argv)

    payload: dict = CertifyPipeline().run(
        cli_args.dataset, cli_args.feature, cli_args.direction, cli_args.settings
    )
    out_path: Path = ResultsWriter.write(
        payload,
        f"certify_{cli_args.dataset}_{cli_args.feature}",
        Path(cli_args.settings.results_dir),
    )
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()

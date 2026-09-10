import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO

_LOGS_DIR: Path = Path(__file__).resolve().parents[2] / "logs"


def setup_logging(name: str, level: int = logging.INFO) -> Path:
    """
    Configure the root logger with a console handler (bare message, same as
    every entry point's previous `logging.basicConfig` call) and a file
    handler under `logs/<name>_<timestamp>.log` (timestamped and leveled,
    since a log file is read back later rather than watched live).

    Args:
        name (str): This entry point's identifying name (e.g. "cli",
            "run_certify"), used as the log file's name prefix.
        level (int): Root logger level passed straight to
            `logging.basicConfig`.

    Returns:
        Path: The log file's path.
    """

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Path = _LOGS_DIR / f"{name}_{stamp}.log"

    console: logging.StreamHandler[TextIO] = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    logging.basicConfig(level=level, handlers=[console, file_handler], force=True)
    return log_path

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

log: logging.Logger = logging.getLogger("sensei.eval.results_writer")

_RESULTS_DIR: Path = Path(__file__).resolve().parents[3] / "results"


class ResultsWriter:
    """
    Writes a JSON-serializable results payload to
    `results/<name>_<YYYYMMDD_HHMMSS>.json`.
    """

    @staticmethod
    def write(
        payload: dict[str, Any], name: str, results_dir: Path | None = None
    ) -> Path:
        """
        Write a results payload to a JSON file, never overwritten (a fresh
        timestamp per call).

        Args:
            payload (dict[str, Any]): The results payload to write.
            name (str): Filename stem; the final name is
                `<name>_<YYYYMMDD_HHMMSS>.json`.
            results_dir (Path | None): Directory to write into; `results/`
                if None.

        Returns:
            Path: The path to the written JSON file.
        """

        out_dir: Path = results_dir if results_dir is not None else _RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path: Path = out_dir / f"{name}_{stamp}.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        log.info("wrote %s", out_path)
        return out_path

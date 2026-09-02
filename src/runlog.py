import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return str(obj)


class RunWriter:
    """
    One directory per run: runs/<timestamp>-<git-sha>[-<tag>]/, holding
    the resolved config, cut log, per-iteration snapshots, final leaf
    values, and an environment capture -- exactly what README documents
    every run should leave behind.
    """

    def __init__(self, tag: str | None = None, root: Path | None = None) -> None:
        root = root or _RUNS_ROOT
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}-{_git_sha()}"
        if tag:
            name += f"-{tag}"

        self.dir = root / name
        self.dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, cfg) -> None:
        self._write_json("config.json", asdict(cfg))

    def write_environment(self) -> None:
        packages = {}
        for module_name, attr in [
            ("numpy", "__version__"), ("pandas", "__version__"),
            ("sklearn", "__version__"), ("xgboost", "__version__"),
            ("scipy", "__version__"),
        ]:
            try:
                mod = __import__(module_name)
                packages[module_name] = getattr(mod, attr)
            except Exception:
                packages[module_name] = "unknown"

        try:
            import gurobipy as gp
            packages["gurobipy"] = ".".join(str(v) for v in gp.gurobi.version())
        except Exception:
            packages["gurobipy"] = "unknown"

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        }
        self._write_json("environment.json", env)

    def write_cuts(self, cuts: list[dict]) -> None:
        """--dump-cuts: every cut's leaf indices and coefficients, one per line."""
        with open(self.dir / "cuts.jsonl", "w") as f:
            for cut in cuts:
                f.write(json.dumps(cut, default=_json_default) + "\n")

    def write_snapshots(self, history: list) -> None:
        self._write_json("snapshots.json", [asdict(s) for s in history])

    def write_leaf_values(self, v: np.ndarray) -> None:
        np.save(self.dir / "leaf_values.npy", v)

    def write_summary(self, summary: dict) -> None:
        self._write_json("summary.json", summary)

    def _write_json(self, name: str, obj: Any) -> None:
        with open(self.dir / name, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)

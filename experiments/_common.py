"""Shared bootstrap for experiment scripts: src/ on sys.path, common CLI bits."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config  # noqa: E402
from spec import SPECS  # noqa: E402

ALL_DATASETS = list(SPECS.keys())


def parse_seeds(raw: str) -> list[int]:
    return [int(s) for s in raw.split(",") if s.strip()]


def build_config(
    dataset: str, seed: int,
    n_estimators: int | None = None, max_depth: int | None = None,
) -> Config:
    cfg = Config(dataset=dataset, seed=seed)
    if n_estimators is not None:
        cfg.n_estimators = n_estimators
    if max_depth is not None:
        cfg.max_depth = max_depth
    return cfg


def resolve_datasets(args) -> list[str]:
    if getattr(args, "all", False):
        return ALL_DATASETS
    if getattr(args, "dataset", None):
        return [args.dataset]
    raise SystemExit("pass --all or --dataset <name>")

from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_DIR: Path = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    test_size: float
    eval_holdout: float


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    max_depth: int


@dataclass(frozen=True)
class SensitivityConfig:
    eps: float
    theta: float
    gap: float


@dataclass(frozen=True)
class RepairConfig:
    mu: float
    kap: float


@dataclass(frozen=True)
class LoopConfig:
    max_iters: int
    A_min: float
    stall_delta: float
    cuts_per_round: int


@dataclass(frozen=True)
class OracleConfig:
    time_limit_s: float
    mip_gap: float


@dataclass(frozen=True)
class BinsConfig:
    n_quantile_bins: int


@dataclass(frozen=True)
class Seeds:
    data_split: int
    model_train: int
    baseline4_retrain: int


@dataclass(frozen=True)
class Settings:
    dataset: DatasetConfig
    model: ModelConfig
    sensitivity: SensitivityConfig
    repair: RepairConfig
    loop: LoopConfig
    oracle: OracleConfig
    bins: BinsConfig
    results_dir: str
    seeds: Seeds


def load_defaults(path: Path | str | None = None) -> Settings:
    """Load sensei/config/defaults.yaml (or an override path with the same shape)."""

    raw = yaml.safe_load(Path(path or _CONFIG_DIR / "defaults.yaml").read_text())

    return Settings(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        sensitivity=SensitivityConfig(**raw["sensitivity"]),
        repair=RepairConfig(**raw["repair"]),
        loop=LoopConfig(**raw["loop"]),
        oracle=OracleConfig(**raw["oracle"]),
        bins=BinsConfig(**raw["bins"]),
        results_dir=raw["results_dir"],
        seeds=Seeds(**raw["seeds"]),
    )


def ensense_pin() -> str:
    """The pinned Ensense core commit SHA. Every results JSON must carry this."""

    return (_CONFIG_DIR / "ensense_pin.txt").read_text().strip()

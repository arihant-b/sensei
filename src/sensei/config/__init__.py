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
    type: str


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
    type: str  # "sensei" | "ensense"
    method: str  # "pb" | "milp" -- which Ensense core solver family to call

    def __post_init__(self) -> None:
        if self.type not in ("sensei", "ensense"):
            raise ValueError(
                f"oracle.type must be 'sensei' or 'ensense', got {self.type!r}"
            )
        if self.method not in ("pb", "milp"):
            raise ValueError(
                f"oracle.method must be 'pb' or 'milp', got {self.method!r}"
            )


@dataclass(frozen=True)
class BinsConfig:
    n_quantile_bins: int


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
    seed: int


def load_defaults(path: Path | str | None = None) -> Settings:
    """
    Load sensei/config/defaults.yaml (or an override path with the same shape).

    Args:
        path (Path | str | None): Override path to load instead of
            `sensei/config/defaults.yaml`. Must have the same YAML shape
            (one section per `Settings` field).

    Returns:
        Settings: The fully populated settings object.
    """

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
        seed=raw["seed"],
    )


def ensense_pin() -> str:
    """
    The pinned Ensense core commit SHA. Every results JSON must carry this.

    Returns:
        str: The commit SHA recorded in `config/ensense_pin.txt`.
    """

    return (_CONFIG_DIR / "ensense_pin.txt").read_text().strip()

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # --- data -------------------------------------------------------------
    dataset: str = 'adult'
    test_size: float = 0.2
    eval_holdout: float = 0.2
    seed: int = 42

    # --- model ------------------------------------------------------------
    n_estimators: int = 200
    max_depth: int = 5

    # --- sensitivity ------------------------------------------------------
    epsilon: float = 0.10          # sensitivity budget (cut tightness)
    gap: float = 0.50              # confident-flip margin
    theta: float = 1e-6            # plausibility threshold

    # --- repair QP --------------------------------------------------------
    mu: float = 0.01               # weight on leaf-drift term

    # --- loop -------------------------------------------------------------
    max_iters: int = 50
    min_accuracy: float = 0.82     # floor; stop if we drop below
    stall_delta: float = 1e-3      # stop if improvement smaller than this
    cuts_per_round: int = 5        # batch several cuts per MILP call

    # --- oracle cost control ---------------------------------------------
    use_sampling_screen: bool = True
    n_sample_rows: int = 1000
    mip_gap: float = 0.05
    mip_time_limit: float = 300.0

    # --- density bins -----------------------------------------------------
    n_quantile_bins: int = 10

    artifacts_dir: str = 'runs/'
    tags: list[str] = field(default_factory=list)


def load_config(path: str | Path) -> Config:
    """
    Overlay a YAML file's top-level keys onto Config's defaults. The
    'spec:' block is a separate document handled by spec.load_spec, not
    a Config field, so it is ignored here. Unrecognised top-level keys
    are ignored too, rather than raising -- keeps a config file usable
    even if it also carries notes/metadata Config doesn't model.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    field_names = {f.name for f in fields(Config)}
    overrides = {k: v for k, v in raw.items() if k in field_names}
    return Config(**overrides)

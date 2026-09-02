from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

_CONFIGS_ROOT = Path(__file__).resolve().parents[1] / "configs"


class Direction(Enum):
    UP = "up"                       # increasing the feature must not decrease the score
    DOWN = "down"


@dataclass
class SensitivitySpec:
    """
    Declared by a human, before anything runs. This is Q3 from the notes:
    'should this change flip the answer?'. It is never inferred from data.
    """
    protected: list                 # flipping these must NOT change the label
    monotone: dict                  # feature -> Direction
    immutable: list                 # cannot be flipped at all
    integer_features: list
    one_hot_groups: dict            # group name -> list of column names
    ranges: dict                    # feature -> (lo, hi)
    functional_deps: list           # list of (if_col, if_val, then_col, then_val)

    @property
    def free(self):
        raise NotImplementedError("TODO: all columns minus protected/monotone")


def load_spec(path: str | Path) -> SensitivitySpec:
    """
    Parse the 'spec:' block of a configs/*.yaml file. Values are taken
    exactly as declared (units, encoding) -- this loader does not
    convert or infer anything, same posture as ValidityChecker.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    block = raw.get("spec", {})

    monotone = {
        feature: Direction(direction)
        for feature, direction in block.get("monotone", {}).items()
    }
    ranges = {
        feature: (float(lo), float(hi))
        for feature, (lo, hi) in block.get("ranges", {}).items()
    }
    functional_deps = [tuple(dep) for dep in block.get("functional_deps", [])]

    return SensitivitySpec(
        protected=list(block.get("protected", [])),
        monotone=monotone,
        immutable=list(block.get("immutable", [])),
        integer_features=list(block.get("integer_features", [])),
        one_hot_groups=dict(block.get("one_hot_groups", {})),
        ranges=ranges,
        functional_deps=functional_deps,
    )


# dataset name (matches Config.dataset / dataset/<name>/) -> config filename.
# Filenames follow README's convention; 'pima' the dataset folder is
# actually named 'pimadiabetes' (see configs/pima.yaml's own comment).
_REGISTERED_CONFIGS = {
    "adult": "adult.yaml",
    "german_credit": "german_credit.yaml",
    "pimadiabetes": "pimadiabetes.yaml",
}

SPECS = {
    dataset: load_spec(_CONFIGS_ROOT / filename)
    for dataset, filename in _REGISTERED_CONFIGS.items()
}

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

_SPEC_DIR: Path = Path(__file__).resolve().parent


class Direction(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"


@dataclass(frozen=True)
class Preprocessing:
    drop_fd_redundant_columns: tuple[str, ...] = ()
    integer_code_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainRule:
    """
    A linear inequality over the features, for type-validity (Q1) and Tier B1
    preprocessing. The coefficients are in the feature space, not the leaf space.
    """

    coefficients: dict[str, float]
    sense: str  # "<=" | ">=" | "=="
    rhs: float


@dataclass(frozen=True)
class Spec:
    """
    The parsed sensei/spec/<dataset>.yaml file, frozen after Stage 0.
    """

    dataset: str
    protected: tuple[str, ...]
    monotone: dict[str, Direction]
    immutable: tuple[str, ...]

    integer_features: tuple[str, ...]
    one_hot_groups: dict[str, tuple[str, ...]]
    ranges: dict[str, tuple[float, float]]
    functional_deps: tuple[tuple[str, float, str, float], ...]
    domain_rules: tuple[DomainRule, ...]

    preprocessing: Preprocessing

    spec_hash: str = field(compare=False)


def _spec_hash(raw: dict) -> str:
    """
    Compute a stable hash over the raw YAML content.

    Args:
        raw (dict): The raw dictionary loaded from the YAML file.

    Returns:
        str: The SHA256 hash of the canonical JSON representation of the raw dictionary.
    """

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_spec(dataset: str, path: Path | str | None = None) -> Spec:
    """
    Load a spec from a YAML file.

    Args:
        dataset (str): The dataset name, used to locate the spec file.
        path (Path | str | None, optional): The path to the spec file. If None, the
                                            default path is used. Defaults to None.

    Returns:
        Spec: The loaded spec object.
    """

    spec_path = Path(path or _SPEC_DIR / f"{dataset}.yaml")
    raw = yaml.safe_load(spec_path.read_text())

    monotone: dict[Any, Direction] = {
        feature: Direction(direction)
        for feature, direction in raw.get("monotone", {}).items()
    }
    ranges: dict[Any, tuple[float, float]] = {
        feature: (float(lo), float(hi))
        for feature, (lo, hi) in raw.get("ranges", {}).items()
    }
    one_hot_groups: dict[Any, tuple[Any, ...]] = {
        group: tuple(cols) for group, cols in raw.get("one_hot_groups", {}).items()
    }
    functional_deps: tuple[tuple[Any, float, Any, float], ...] = tuple(
        tuple(dep) for dep in raw.get("functional_deps", [])
    )
    domain_rules: tuple[DomainRule, ...] = tuple(
        DomainRule(
            coefficients={k: float(v) for k, v in rule["coefficients"].items()},
            sense=rule["sense"],
            rhs=float(rule["rhs"]),
        )
        for rule in raw.get("domain_rules", [])
    )
    preprocessing_raw = raw.get("preprocessing", {})

    return Spec(
        dataset=raw["dataset"],
        protected=tuple(raw.get("protected", [])),
        monotone=monotone,
        immutable=tuple(raw.get("immutable", [])),
        integer_features=tuple(raw.get("integer_features", [])),
        one_hot_groups=one_hot_groups,
        ranges=ranges,
        functional_deps=functional_deps,
        domain_rules=domain_rules,
        preprocessing=Preprocessing(
            drop_fd_redundant_columns=tuple(
                preprocessing_raw.get("drop_fd_redundant_columns", [])
            ),
            integer_code_features=tuple(
                preprocessing_raw.get("integer_code_features", [])
            ),
        ),
        spec_hash=_spec_hash(raw),
    )


ALL_DATASETS: tuple[str, ...] = ("adult", "german_credit", "pimadiabetes", "churn")

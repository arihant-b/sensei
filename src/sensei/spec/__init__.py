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
class DomainRule:
    """
    A linear inequality over the features, for type-validity (Q1) checks. The
    coefficients are in the feature space, not the leaf space.
    """

    coefficients: dict[str, float]
    sense: str  # "<=" | ">=" | "=="
    rhs: float


@dataclass(frozen=True)
class Spec:
    """
    One dataset's parsed spec/<dataset>.yaml: which features are protected,
    monotone, or immutable, plus every validity rule (integer/one-hot/range,
    functional dependency, linear domain rule) used to tell a real point
    apart from a malformed one.
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

    spec_hash: str = field(compare=False)


def _spec_hash(raw: dict[str, Any]) -> str:
    """
    SHA256 of the raw YAML dict, keys sorted so the hash doesn't depend on
    field order in the file. Used to detect when a spec changed under a
    result that was certified against the old version.

    Args:
        raw (dict[str, Any]): The parsed YAML spec.

    Returns:
        str: The hex-encoded SHA256 digest.
    """

    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_spec(dataset: str, path: Path | str | None = None) -> Spec:
    """
    Parse spec/<dataset>.yaml (or `path`, if given) into a `Spec`.

    Args:
        dataset (str): Dataset name, matching `spec/<dataset>.yaml`.
        path (Path | str | None): Override path to parse instead.

    Returns:
        Spec: The parsed spec.
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
        spec_hash=_spec_hash(raw),
    )


ALL_DATASETS: tuple[str, ...] = ("adult", "german_credit", "pimadiabetes", "churn")

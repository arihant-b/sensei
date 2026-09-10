from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pair:
    """
    A pair of points (x1, x2) with their active leaves (ell1, ell2) and the objective
    gap (gap = E_v(x1) - E_v(x2)) between them. Both `SenseiOracle` and `EnsenseOracle`
    return these.
    """

    x1: NDArray[np.float64]
    x2: NDArray[np.float64]
    ell1: NDArray[np.int64]
    ell2: NDArray[np.int64]
    gap: float  # E_v(x1) - E_v(x2)
    flip_set: tuple[str, ...]
    direction: str  # "protected" | "monotone_wrong"
    solver_status: str  # "OPTIMAL" | "FEASIBLE" | "TIMEOUT" | "UNSAT"


class OracleTimeout(Exception):
    """
    The oracle did not resolve within time_limit_s. Unknown, not None. Both
    `SenseiOracle` and `EnsenseOracle` raise this.
    """


class OracleSaturated(Exception):
    """
    `EnsenseOracle`'s postfilter rejection budget was exhausted. Not a certificate.
    """


class OracleDegenerate(Exception):
    """
    An oracle returned a pair that fails basic sanity checks -- x1 and x2 are
    identical, differ outside the declared flip set, or have zero gap. Not a
    valid counterexample, and NOT the same thing as None (which means the
    oracle searched and legitimately found nothing). Collapsing this into
    None would silently turn "we got garbage back" into "verified absent".
    """


@dataclass
class TreeStructure:
    """
    Represents the structure of a tree ensemble, including unique thresholds for each
    feature, the mapping of tree IDs to global leaf indices, and the paths to each leaf.
    Built by `SenseiOracle`'s own `oracle/sensei/encoding.py::TreeEncoder`, but a plain
    data type with no Gurobi dependency -- `EnsenseOracle` (`oracle/ensense/adapter.py`)
    accepts one too, purely for signature compatibility with `SenseiOracle`, so it
    belongs here rather than in either oracle's own package.
    """

    thresholds: dict[str, list[float]]  # feature -> sorted unique thresholds
    trees_leaves: dict[int, list[int]]  # tree_id -> global leaf indices
    leaf_paths: dict[int, list[tuple[str, float, str]]]  # global leaf idx -> path


@dataclass(frozen=True)
class NoGood:
    """
    Forbids one specific joint (ell1, ell2) leaf pattern from being returned
    again by the oracle -- see `oracle/sensei/nogoods.py::NoGoodBuilder` for
    how these are built and injected into `SenseiOracle`'s MILP. A plain data
    type with no Gurobi dependency; `EnsenseOracle` accepts a list of these
    too (and ignores them -- it has no mechanism to exclude a
    previously-found pattern), purely for signature compatibility with
    `SenseiOracle`, so it belongs here rather than in either oracle's own
    package.
    """

    idx1: NDArray[np.int64]  # N1*: the T global leaf indices active for x1
    idx2: NDArray[np.int64]  # N2*: the T global leaf indices active for x2
    rhs: int  # 2*n_trees - 1

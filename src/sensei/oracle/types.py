from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pair:
    """
    A pair of points (x1, x2) with their active leaves (ell1, ell2) and the objective
    gap (gap = E_v(x1) - E_v(x2)) between them. Both Tier A and Tier B return these.
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
    The oracle did not resolve within time_limit_s. Unknown, not None. Both Tier A and
    Tier B return these.
    """


class OracleSaturated(Exception):
    """
    Tier B's rejection budget was exhausted. Not a certificate. Both Tier A and Tier B
    return these.
    """

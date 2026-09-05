from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sensei.oracle.types import Pair


@dataclass(frozen=True)
class Cut:
    """
    A sparse leaf-difference cut constructed from a counterexample pair.
    """

    idx: NDArray[np.int64]  # nonzero global leaf indices of d
    coef: NDArray[
        np.float64
    ]  # +1.0 where only ell1 has it, -1.0 where only ell2 has it
    eps: float


class CutBuilder:
    """
    Builds sparse leaf-difference cuts from counterexample pairs. Each cut forbids the
    joint pattern of active leaves for the two points, ensuring that at least one tree's
    active leaf differs between the two points in future solutions.
    """

    @staticmethod
    def make_cut(pair: Pair, eps: float) -> Cut:
        """
        Construct a sparse leaf-difference cut from a counterexample pair.

        Args:
            pair (Pair): The pair of points for which to construct the cut.
            eps (float): The epsilon value used in the cut construction.

        Returns:
            Cut: The constructed sparse leaf-difference cut representing the difference
                 between the two points.
        """

        set1: set[int] = {int(n) for n in pair.ell1}
        set2: set[int] = {int(n) for n in pair.ell2}
        only1: list[int] = sorted(set1 - set2)
        only2: list[int] = sorted(set2 - set1)

        assert only1 or only2, "identical leaf paths cannot produce a gap"

        idx: NDArray[np.int64] = np.array(only1 + only2, dtype=np.int64)
        coef: NDArray[np.float64] = np.array(
            [1.0] * len(only1) + [-1.0] * len(only2), dtype=np.float64
        )
        return Cut(idx=idx, coef=coef, eps=eps)

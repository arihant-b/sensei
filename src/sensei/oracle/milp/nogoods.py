from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from numpy.typing import NDArray

from sensei.oracle.types import Pair


@dataclass(frozen=True)
class NoGood:
    idx1: NDArray[np.int64]  # N1*: the T global leaf indices active for x1
    idx2: NDArray[np.int64]  # N2*: the T global leaf indices active for x2
    rhs: int  # 2*n_trees - 1


class NoGoodBuilder:
    """
    Builds no-good constraints from pairs of points and the number of trees in
    the ensemble. Each no-good constraint forbids the joint pattern of active
    leaves for the two points, ensuring that at least one tree's active leaf
    differs between the two points in future solutions.
    """

    @staticmethod
    def make_nogood(pair: Pair, n_trees: int) -> NoGood:
        """
        Creates a no-good constraint from a pair of points and the number of trees.

        Args:
            pair (Pair): The pair of points for which to create the no-good constraint.
            n_trees (int): The number of trees in the ensemble, used to compute the
                           right-hand side of the constraint.

        Returns:
            NoGood: The constructed no-good constraint representing the joint pattern of
                    the two points.
        """

        assert len(pair.ell1) == n_trees and len(pair.ell2) == n_trees, (
            f"expected {n_trees} active leaves per point (one per tree), "
            f"got {len(pair.ell1)} and {len(pair.ell2)}"
        )
        return NoGood(
            idx1=np.asarray(pair.ell1, dtype=np.int64),
            idx2=np.asarray(pair.ell2, dtype=np.int64),
            rhs=2 * n_trees - 1,
        )

    @staticmethod
    def add_nogood_constraint(
        m: gp.Model,
        ell1_vars: dict[int, gp.Var],
        ell2_vars: dict[int, gp.Var],
        nogood: NoGood,
    ) -> None:
        """
        Adds a no-good constraint to the Gurobi model, ensuring that at least one
        tree's active leaf differs between the two points represented by the no-good.

        Args:
            m (gp.Model): The Gurobi model to which the no-good constraint will be
                          added.
            ell1_vars (dict[int, gp.Var]): A dictionary mapping global leaf indices to
                                           Gurobi variables for the first point.
            ell2_vars (dict[int, gp.Var]): A dictionary mapping global leaf indices to
                                           Gurobi variables for the second point.
            nogood (NoGood): The no-good constraint to be added.
        """

        lhs: gp.LinExpr = gp.quicksum(
            ell1_vars[int(n)] for n in nogood.idx1
        ) + gp.quicksum(ell2_vars[int(n)] for n in nogood.idx2)
        m.addConstr(lhs <= nogood.rhs)

import gurobipy as gp
import numpy as np

from sensei.oracle.types import NoGood, Pair


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
        Build the NoGood forbidding `pair`'s exact joint leaf pattern.

        Args:
            pair (Pair): The pair whose joint (ell1, ell2) leaf pattern
                should never be returned again.
            n_trees (int): Number of trees in the ensemble -- both
                `pair.ell1` and `pair.ell2` must have exactly this length.

        Returns:
            NoGood: `rhs = 2*n_trees - 1`, so proposing this exact joint
                pattern again sums to `2*n_trees`, violating the constraint.
        """

        assert len(pair.ell1) == n_trees and len(pair.ell2) == n_trees, (
            f"expected {n_trees} active leaves per point (one per tree), "
            f"got {len(pair.ell1)} and {len(pair.ell2)}"
        )

        # rhs is 2*n_trees - 1 because if solver tries to propose the same combination
        # of ell1 and ell2, then it would result in sum being exactly 2*n which is 1
        # more than rhs resulting in false, hence the solver is forbidden from
        # returning the same joint pair of leaf choices again
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
        Inject `sum(ell1_vars[nogood.idx1]) + sum(ell2_vars[nogood.idx2]) <=
        2T - 1` into `m`: at least one of those 2T leaf choices must differ
        next time, forbidding exactly this joint pattern and nothing else.

        Args:
            m (gp.Model): The Gurobi model to add the constraint to.
            ell1_vars (dict[int, gp.Var]): Leaf-indicator variables for x1,
                keyed by global leaf index.
            ell2_vars (dict[int, gp.Var]): Leaf-indicator variables for x2,
                keyed by global leaf index.
            nogood (NoGood): The joint leaf pattern to forbid.
        """

        lhs: gp.LinExpr = gp.quicksum(
            ell1_vars[int(n)] for n in nogood.idx1
        ) + gp.quicksum(ell2_vars[int(n)] for n in nogood.idx2)
        m.addConstr(lhs <= nogood.rhs)

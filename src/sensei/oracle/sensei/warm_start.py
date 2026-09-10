import gurobipy as gp

from sensei.oracle.types import Pair


class WarmStartApplier:
    """Seeds a fresh MILP with the previous iteration's pair as a candidate solution."""

    @staticmethod
    def apply_warm_start(
        x1_vars: dict[str, gp.Var],
        x2_only_vars: dict[str, gp.Var],
        ell1: dict[int, gp.Var],
        ell2: dict[int, gp.Var],
        warm_start: Pair,
        columns: list[str],
        flip_set: tuple[str, ...],
    ) -> None:
        """
        Set Gurobi's `.Start` attribute on every relevant variable (both
        points' feature values, both leaf-indicator sets) from the previous
        `warm_start` pair. Since tree structure never changes, that pair's
        leaf assignment is still structurally valid to try -- just not
        guaranteed feasible under the current leaf values, so this is a MIP
        start/candidate, not a guaranteed feasible warm start.

        Args:
            x1_vars (dict[str, gp.Var]): x1's per-feature decision variables.
            x2_only_vars (dict[str, gp.Var]): x2's own decision variables,
                one per feature in `flip_set`.
            ell1 (dict[int, gp.Var]): x1's leaf-indicator variables, keyed
                by global leaf index.
            ell2 (dict[int, gp.Var]): x2's leaf-indicator variables, keyed
                by global leaf index.
            warm_start (Pair): The previous iteration's pair to seed from.
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
        """

        for i, f in enumerate(columns):
            x1_vars[f].Start = float(warm_start.x1[i])

        for f in flip_set:
            if f in x2_only_vars:
                x2_only_vars[f].Start = float(warm_start.x2[columns.index(f)])

        ws_leaves1: set[int] = {int(n) for n in warm_start.ell1}
        ws_leaves2: set[int] = {int(n) for n in warm_start.ell2}

        for g, var in ell1.items():
            var.Start = 1.0 if g in ws_leaves1 else 0.0

        for g, var in ell2.items():
            var.Start = 1.0 if g in ws_leaves2 else 0.0

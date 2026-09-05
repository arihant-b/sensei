import gurobipy as gp

from sensei.oracle.types import Pair


class WarmStartApplier:
    """
    Apply a warm start to a Gurobi MILP model, given the previous pair and the current
    model's variables.
    """

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
        Apply a warm start to the Gurobi MILP model by setting the Start attribute of
        the relevant variables based on the previous pair's values.

        Args:
            x1_vars (dict[str, gp.Var]): The Gurobi variables corresponding to the first
                                         point's features.
            x2_only_vars (dict[str, gp.Var]): The Gurobi variables corresponding to the
                                              second point's features that are not
                                              shared with the first point.
            ell1 (dict[int, gp.Var]): The Gurobi variables corresponding to the first
                                      point's active leaves in the tree ensemble.
            ell2 (dict[int, gp.Var]): The Gurobi variables corresponding to the second
                                      point's active leaves in the tree ensemble.
            warm_start (Pair): The previous pair of points to use for the warm start.
            columns (list[str]): The list of feature columns in the model.
            flip_set (tuple[str, ...]): The set of features that are allowed to differ
                                        between the two points.
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

from dataclasses import dataclass

import gurobipy as gp
import pandas as pd
import xgboost as xgb
from gurobipy import GRB

_SPLIT_EPS = 1e-4


@dataclass
class TreeStructure:
    """
    Represents the structure of a tree ensemble, including unique thresholds for each
    feature, the mapping of tree IDs to global leaf indices, and the paths to each leaf.
    """

    thresholds: dict[str, list[float]]  # feature -> sorted unique thresholds
    trees_leaves: dict[int, list[int]]  # tree_id -> global leaf indices
    leaf_paths: dict[int, list[tuple[str, float, str]]]  # global leaf idx -> path


class TreeEncoder:
    """
    Encodes a tree ensemble into a Gurobi MILP model, including order variables for
    thresholds, leaf indicators, and integrality constraints for integer features.
    """

    @staticmethod
    def extract_tree_structure(
        booster: xgb.Booster, leaf_index: dict[tuple[int, int], int], columns: list[str]
    ) -> TreeStructure:
        """
        Extracts the structure of a tree ensemble from an XGBoost booster, including
        unique thresholds for each feature, the mapping of tree IDs to global leaf
        indices, and the paths to each leaf.

        Args:
            booster (xgb.Booster): The XGBoost booster containing the tree ensemble.
            leaf_index (dict[tuple[int, int], int]): The mapping of tree IDs and leaf
                                                     IDs to global leaf indices.
            columns (list[str]): The list of column names.

        Raises:
            ValueError: If a feature identifier in the booster is unrecognized (not in
                        columns or not a valid "f{index}" format).

        Returns:
            TreeStructure: The extracted tree structure, including thresholds,
                           trees_leaves, and leaf_paths.
        """

        df: pd.DataFrame = booster.trees_to_dataframe()

        def _real_name(raw: str) -> str:
            if raw in columns:
                return raw

            if raw.startswith("f") and raw[1:].isdigit():
                return columns[int(raw[1:])]

            raise ValueError(
                f"unrecognized feature identifier {raw!r} in trees_to_dataframe()"
            )

        by_tree: dict[int, dict[str, object]] = {}

        for row in df.itertuples():
            by_tree.setdefault(int(str(row.Tree)), {})[str(row.ID)] = row

        thresholds: dict[str, set[float]] = {}
        trees_leaves: dict[int, list[int]] = {}
        leaf_paths: dict[int, list[tuple[str, float, str]]] = {}

        for tree_id, nodes in by_tree.items():
            trees_leaves[tree_id] = []
            root = f"{tree_id}-0"
            stack: list[tuple[str, list[tuple[str, float, str]]]] = [(root, [])]

            while stack:
                node_id, path = stack.pop()
                row: object = nodes[node_id]

                if row.Feature == "Leaf":  # type: ignore[attr-defined]
                    leaf_num = int(node_id.split("-")[1])
                    g: int = leaf_index[(tree_id, leaf_num)]
                    trees_leaves[tree_id].append(g)
                    leaf_paths[g] = path
                else:
                    feat: str = _real_name(str(row.Feature))  # type: ignore[attr-defined]
                    thresh = float(row.Split)  # type: ignore[attr-defined]
                    thresholds.setdefault(feat, set()).add(thresh)
                    stack.append((str(row.Yes), [*path, (feat, thresh, "yes")]))  # type: ignore[attr-defined]
                    stack.append((str(row.No), [*path, (feat, thresh, "no")]))  # type: ignore[attr-defined]

        return TreeStructure(
            thresholds={f: sorted(ts) for f, ts in thresholds.items()},
            trees_leaves=trees_leaves,
            leaf_paths=leaf_paths,
        )

    @staticmethod
    def add_order_variables(
        m: gp.Model,
        thresholds: dict[str, list[float]],
        columns: list[str],
        name_prefix: str,
    ) -> dict[tuple[str, float], gp.Var]:
        """
        Add order variables to the Gurobi model for each feature and threshold.

        Args:
            m (gp.Model): The Gurobi model to which the order variables will be added.
            thresholds (dict[str, list[float]]): A dictionary mapping feature names to
                                                 their sorted unique thresholds.
            columns (list[str]): The list of feature names to consider for adding order
                                 variables.
            name_prefix (str): A prefix to use for naming the order variables in the
                               Gurobi model.

        Returns:
            dict[tuple[str, float], gp.Var]: A dictionary mapping (feature, threshold)
                                             pairs to their corresponding Gurobi order
                                             variables.
        """

        order_vars: dict[tuple[str, float], gp.Var] = {}

        for f in columns:
            ts: list[float] = thresholds.get(f, [])

            for t in ts:
                order_vars[(f, t)] = m.addVar(
                    vtype=GRB.BINARY, name=f"{name_prefix}_u_{f}_{t:.6g}"
                )

            for t_lo, t_hi in zip(ts, ts[1:], strict=False):
                m.addConstr(order_vars[(f, t_lo)] <= order_vars[(f, t_hi)])

        return order_vars

    @staticmethod
    def link_order_variables(
        m: gp.Model,
        order_vars: dict[tuple[str, float], gp.Var],
        x_vars: dict[str, gp.Var],
        thresholds: dict[str, list[float]],
    ) -> None:
        """
        Link the order variables u[f, t] to the corresponding feature variables x_f
        in the Gurobi model. The order variables are defined such that:
            u[f, t] = 1 <=> x_f <= t - eps,
            u[f, t] = 0 <=> x_f >= t.
        This method adds the necessary constraints to enforce this relationship.

        Args:
            m (gp.Model): The Gurobi model to which the constraints will be added.
            order_vars (dict[tuple[str, float], gp.Var]): A dictionary mapping (feature,
                                                          threshold) pairs to their
                                                          corresponding Gurobi order
                                                          variables.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            thresholds (dict[str, list[float]]): A dictionary mapping feature names to
                                                 their sorted unique thresholds.
        """

        le, ge = GRB.LESS_EQUAL, GRB.GREATER_EQUAL

        for f, ts in thresholds.items():
            for t in ts:
                u: gp.Var = order_vars[(f, t)]
                m.addGenConstrIndicator(u, True, x_vars[f], le, t - _SPLIT_EPS)
                m.addGenConstrIndicator(u, False, x_vars[f], ge, t)

    @staticmethod
    def add_leaf_indicators(
        m: gp.Model,
        structure: TreeStructure,
        order_vars: dict[tuple[str, float], gp.Var],
        name_prefix: str,
    ) -> dict[int, gp.Var]:
        """
        Add leaf indicator variables to the Gurobi model for each global leaf index in
        the tree structure. Each leaf indicator variable is linked to the order
        variables corresponding to the path leading to that leaf.

        Args:
            m (gp.Model): The Gurobi model to which the leaf indicator variables will be
                          added.
            structure (TreeStructure): The tree structure containing the paths to each
                                       leaf.
            order_vars (dict[tuple[str, float], gp.Var]): A dictionary mapping (feature,
                                                          threshold) pairs to their
                                                          corresponding Gurobi order
                                                          variables.
            name_prefix (str): A prefix to use for naming the leaf indicator variables
                               in the Gurobi model.

        Returns:
            dict[int, gp.Var]: A dictionary mapping global leaf indices to their
                               corresponding Gurobi leaf indicator variables.
        """

        ell: dict[int, gp.Var] = {}

        for g, path in structure.leaf_paths.items():
            ell[g] = m.addVar(vtype=GRB.BINARY, name=f"{name_prefix}_ell_{g}")

            literals: list[gp.Var | gp.LinExpr] = [
                order_vars[(f, t)] if direction == "yes" else (1 - order_vars[(f, t)])
                for f, t, direction in path
            ]

            if not literals:
                # stump tree: no splits, only one leaf -- always active, no indicator
                # variable needed
                m.addConstr(ell[g] == 1)
                continue

            for lit in literals:
                m.addConstr(ell[g] <= lit)

            m.addConstr(ell[g] >= gp.quicksum(literals) - (len(literals) - 1))

        for leaves in structure.trees_leaves.values():
            m.addConstr(gp.quicksum(ell[g] for g in leaves) == 1)

        return ell

    @staticmethod
    def add_integrality_constraints(
        m: gp.Model,
        x_vars: dict[str, gp.Var],
        integer_features: tuple[str, ...],
        feature_bounds: dict[str, tuple[float, float]],
        name_prefix: str,
    ) -> None:
        """
        Add integrality constraints to the Gurobi model.

        Args:
            m (gp.Model): The Gurobi model to which the integrality constraints will be
                          added.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            integer_features (tuple[str, ...]): A tuple of feature names that are
                                                required to be integer-valued.
            feature_bounds (dict[str, tuple[float, float]]): A dictionary mapping
                                                             feature names to their
                                                             (min, max) bounds.
            name_prefix (str): A prefix to use for naming the auxiliary integer
                               variables in the Gurobi model.
        """

        for f in integer_features:
            if f not in x_vars or f not in feature_bounds:
                continue

            lo, hi = feature_bounds[f]
            span: float = hi - lo

            if span <= 0:
                continue

            n_levels = int(round(span))
            aux: gp.Var = m.addVar(
                vtype=GRB.INTEGER, lb=0, ub=n_levels, name=f"{name_prefix}_int_{f}"
            )
            m.addConstr(x_vars[f] * span == aux)

import gurobipy as gp
import pandas as pd
import xgboost as xgb
from gurobipy import GRB

from sensei.oracle.types import TreeStructure

SPLIT_EPS = 1e-4


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
        Walk every tree's dataframe once to build a `TreeStructure`: each
        feature's sorted unique split thresholds, which global leaves belong
        to which tree, and the root-to-leaf path (as a list of (feature,
        threshold, "yes"/"no") literals) for every leaf.

        Args:
            booster (xgb.Booster): The trained ensemble to walk.
            leaf_index (dict[tuple[int, int], int]): `(tree_id, leaf_num) ->`
                flat global leaf index map.
            columns (list[str]): Feature names, used to resolve `"f<index>"`
                placeholders back to real column names.

        Returns:
            TreeStructure: Thresholds, tree-to-leaves, and leaf-to-path maps.

        Raises:
            ValueError: A split's feature identifier is neither a real
                column name nor a valid "f<index>" placeholder.
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
        One binary order variable u[f, t] per (feature, threshold), plus the
        chain constraint u[f, t_lo] <= u[f, t_hi] for consecutive thresholds
        of the same feature (see `link_order_variables` for what u means).

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            thresholds (dict[str, list[float]]): Feature -> sorted unique
                split thresholds.
            columns (list[str]): Feature names to build order variables for.
            name_prefix (str): Prefix for generated variable names (keeps
                x1's and x2's order variables distinct in the same model).

        Returns:
            dict[tuple[str, float], gp.Var]: One binary variable per
                (feature, threshold) pair.
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
        Link each order variable to its feature: `u[f,t]=1 <=> x_f <= t -
        eps`, `u[f,t]=0 <=> x_f >= t` -- so u[f,t] mirrors which side of
        split (f, t) the point falls on, matching how the tree itself
        branches.

        Args:
            m (gp.Model): The Gurobi model to add constraints to.
            order_vars (dict[tuple[str, float], gp.Var]): Order variables
                from `add_order_variables`, keyed by (feature, threshold).
            x_vars (dict[str, gp.Var]): The point's per-feature decision
                variables.
            thresholds (dict[str, list[float]]): Feature -> sorted unique
                split thresholds (must match `order_vars`'s keys).

        Raises:
            ValueError: Two thresholds for the same feature are closer than
                `SPLIT_EPS` -- see `_assert_thresholds_separated`.
        """

        le, ge = GRB.LESS_EQUAL, GRB.GREATER_EQUAL

        TreeEncoder._assert_thresholds_separated(thresholds)

        for f, ts in thresholds.items():
            for t in ts:
                u: gp.Var = order_vars[(f, t)]
                m.addGenConstrIndicator(u, True, x_vars[f], le, t - SPLIT_EPS)
                m.addGenConstrIndicator(u, False, x_vars[f], ge, t)

    @staticmethod
    def _assert_thresholds_separated(thresholds: dict[str, list[float]]) -> None:
        """
        `u[f, t] == 1 <=> x_f <= t - SPLIT_EPS` structurally excludes the
        interval `[t - SPLIT_EPS, t)` from the search domain for every
        threshold `t` -- unavoidable in any MILP encoding of a strict `<`
        split using non-strict Gurobi inequalities. This is provably harmless
        for the objectives this encoding is used for (tree margins and
        plausibility bin membership are both piecewise-CONSTANT between
        thresholds, so the achievable objective value at `x_f = t -
        SPLIT_EPS` is identical to any point in the excluded interval -- no
        completeness is lost with respect to the worst gap or UNSAT/SAT
        verdict) PROVIDED `SPLIT_EPS` is smaller than the gap between any two
        thresholds for the same feature.

        If it isn't, the encoding stops being merely incomplete and becomes
        actively WRONG: two order variables' linking constraints can overlap
        or leave a gap neither `u=0` nor `u=1` satisfies, which can silently
        corrupt leaf routing or plausibility bin membership -- a false UNSAT
        or a wrong worst gap, not just a narrower search. This has not been
        observed on any model built so far (confirmed empirically: the
        smallest real threshold gap seen, on a 200-tree/depth-5 Adult model,
        was 3e-4 against `SPLIT_EPS=1e-4`) but nothing upstream guarantees
        it -- more trees, deeper trees, or a higher-cardinality feature could
        produce closer thresholds. Failing loudly here, once, at encoding
        time is the alternative to silently trusting that gap holds.

        Args:
            thresholds (dict[str, list[float]]): Feature -> sorted unique
                split thresholds to check for closeness.

        Raises:
            ValueError: Two thresholds for the same feature are `SPLIT_EPS`
                apart or closer.
        """

        for f, ts in thresholds.items():
            for prev, curr in zip(ts, ts[1:], strict=False):
                gap = curr - prev

                if gap <= SPLIT_EPS:
                    raise ValueError(
                        f"thresholds {prev} and {curr} for feature '{f}' are only "
                        f"{gap:.2e} apart, at or below SPLIT_EPS ({SPLIT_EPS:.2e}) "
                        f"-- the MILP encoding's split-epsilon trick is no longer "
                        f"sound for this feature (see _assert_thresholds_separated's "
                        f"docstring). Do not silently shrink SPLIT_EPS to make this "
                        f"pass -- it interacts with Gurobi's own IntFeasTol/"
                        f"FeasibilityTol (1e-9); investigate why thresholds landed "
                        f"this close together instead (more trees/depth than "
                        f"expected, a near-duplicate raw feature value, or too many "
                        f"quantile bins for this feature's cardinality)."
                    )

    @staticmethod
    def add_leaf_indicators(
        m: gp.Model,
        structure: TreeStructure,
        order_vars: dict[tuple[str, float], gp.Var],
        name_prefix: str,
    ) -> dict[int, gp.Var]:
        """
        One binary `ell[g]` per global leaf g, forced to 1 exactly when
        every order-variable literal on g's root-to-leaf path holds (an AND
        of literals, encoded the standard MILP way: `ell <= each literal`
        and `ell >= sum(literals) - (n - 1)`), plus one exactly-one-per-tree
        constraint so each tree contributes exactly one active leaf.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            structure (TreeStructure): Supplies `leaf_paths` (root-to-leaf
                literals per global leaf) and `trees_leaves` (which leaves
                belong to which tree).
            order_vars (dict[tuple[str, float], gp.Var]): Order variables
                keyed by (feature, threshold), used to build each leaf's
                path literals.
            name_prefix (str): Prefix for generated variable names.

        Returns:
            dict[int, gp.Var]: One binary leaf-indicator variable per
                global leaf index.
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
        For each declared integer feature, add an auxiliary integer variable
        `aux` spanning the feature's `n_levels = round(hi - lo)` raw values
        and constrain `x_f * (hi - lo) == aux` -- forcing the scaled `x_f`
        to land exactly on one of those `n_levels + 1` evenly-spaced points,
        never a fractional value in between.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            x_vars (dict[str, gp.Var]): The point's per-feature decision
                variables.
            integer_features (tuple[str, ...]): Features that must land on
                an integer raw value.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature, used to compute each feature's level count.
            name_prefix (str): Prefix for generated variable names.
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

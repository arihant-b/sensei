from typing import Literal

import gurobipy as gp
import numpy as np
import xgboost as xgb
from gurobipy import GRB
from numpy.typing import NDArray

from sensei.config import Settings
from sensei.data.bins import FrozenBins
from sensei.model.leaves import LeafMap
from sensei.oracle.sensei.encoding import TreeEncoder
from sensei.oracle.sensei.nogoods import NoGoodBuilder
from sensei.oracle.sensei.warm_start import WarmStartApplier
from sensei.oracle.types import NoGood, OracleTimeout, Pair, TreeStructure
from sensei.spec import Direction, Spec
from sensei.validity.encode_milp import MilpValidityEncoder


class SenseiOracle:
    """Our own tree-ensemble MILP oracle."""

    def worst_valid_pair(
        self,
        booster: xgb.Booster,
        leaf_map: LeafMap,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec | None,
        flip_set: tuple[str, ...],
        direction: str,  # "protected" | "monotone_wrong"
        mode: str,  # "feasibility" | "optimality"
        settings: Settings,
        nogoods: list[NoGood] | None = None,
        warm_start: Pair | None = None,
        enforce_validity: bool = True,
        structure: TreeStructure | None = None,
        v: NDArray[np.float64] | None = None,
        bins: FrozenBins | None = None,
    ) -> Pair | None:
        """
        Build one fresh Gurobi model encoding the tree ensemble, optionally
        every validity/plausibility rule, and any no-goods/warm start, then
        maximize the signed gap E_v(x1) - E_v(x2) over `flip_set`.
        `mode="feasibility"` adds `gap >= settings.sensitivity.eps` as a hard
        constraint and stops at the first feasible solution;
        `mode="optimality"` (`eps` unused) solves to the true worst gap.

        Args:
            booster (xgb.Booster): The trained ensemble (structure only;
                leaf values come from `v`/`leaf_map.v0`).
            leaf_map (LeafMap): Global leaf map for `booster`.
            columns (list[str]): All feature names, in model column order.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            spec (Spec | None): Dataset spec; required when
                `enforce_validity` or `direction="monotone_wrong"`.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            mode (str): `"feasibility"` or `"optimality"`.
            settings (Settings): Supplies `sensitivity.eps`/`.theta`, `seed`,
                `oracle.time_limit_s`/`.mip_gap`.
            nogoods (list[NoGood] | None): Joint leaf patterns to forbid.
            warm_start (Pair | None): A candidate MIP start.
            enforce_validity (bool): Whether to encode Q1/Q2 at all.
            structure (TreeStructure | None): Pre-extracted tree structure;
                extracted fresh from `booster` if None.
            v (NDArray[np.float64] | None): Leaf values to score with;
                `leaf_map.v0` if None.
            bins (FrozenBins | None): Frozen plausibility bins, needed only
                when `enforce_validity` is True.

        Returns:
            Pair | None: The found pair, or None on a proven-infeasible
                model (UNSAT within the encoded domain).

        Raises:
            OracleTimeout: The solver stopped with no solution and no
                infeasibility proof -- never collapse this into None, which
                means something different.
        """

        assert mode in ("feasibility", "optimality")
        assert direction in ("protected", "monotone_wrong")

        eps: float = settings.sensitivity.eps
        theta: float = settings.sensitivity.theta

        if structure is None:
            structure = TreeEncoder.extract_tree_structure(
                booster, leaf_map.leaf_index, columns
            )

        model: gp.Model = self._build_model(
            mode, settings.seed, settings.oracle.time_limit_s, settings.oracle.mip_gap
        )
        x1_vars, x2_only_vars, x2_vars = self._add_point_variables(
            model, columns, flip_set
        )

        if enforce_validity:
            assert spec is not None, "enforce_validity=True requires a spec"
            self._enforce_validity(
                model,
                x1_vars,
                x2_vars,
                flip_set,
                spec,
                feature_bounds,
                bins,
                theta,
                structure,
            )

        if direction == "monotone_wrong":
            assert spec is not None, (
                "direction='monotone_wrong' requires a spec, to know whether the "
                "flip feature is declared increasing or decreasing"
            )
            self._pin_monotone_direction(model, x1_vars, x2_vars, flip_set, spec)

        ell1, ell2 = self._encode_leaf_indicators(
            model, structure, columns, flip_set, x1_vars, x2_only_vars
        )

        for nogood in nogoods or []:
            NoGoodBuilder.add_nogood_constraint(model, ell1, ell2, nogood)

        if warm_start is not None:
            WarmStartApplier.apply_warm_start(
                x1_vars, x2_only_vars, ell1, ell2, warm_start, columns, flip_set
            )

        self._set_gap_objective(model, leaf_map, structure, ell1, ell2, v, mode, eps)
        model.optimize()

        return self._extract_result(
            model,
            x1_vars,
            x2_vars,
            ell1,
            ell2,
            structure,
            columns,
            flip_set,
            direction,
            settings.oracle.time_limit_s,
        )

    def _build_model(
        self, mode: str, seed: int, time_limit_s: float, mip_gap: float
    ) -> gp.Model:
        """
        A fresh Gurobi model+env with tight tolerances (`FeasibilityTol`/
        `IntFeasTol` = 1e-9, needed for `SPLIT_EPS`'s soundness margin --
        see `encoding.py::_assert_thresholds_separated`). `mip_gap` is
        forced to 0 in optimality mode: reporting/certification needs the
        TRUE worst gap, not an approximation.

        Args:
            mode (str): `"feasibility"` or `"optimality"`.
            seed (int): Gurobi's own `Seed` parameter.
            time_limit_s (float): Gurobi's own `TimeLimit` parameter.
            mip_gap (float): Relative MIP gap; ignored (forced to 0) in
                optimality mode.

        Returns:
            gp.Model: A fresh, empty model with these parameters set.
        """

        env = gp.Env(
            params={
                "DualReductions": 0,
                "FeasibilityTol": 1e-9,
                "IntFeasTol": 1e-9,
                "MIPGap": 0.0 if mode == "optimality" else mip_gap,
                "OutputFlag": 0,
                "Seed": seed,
                "TimeLimit": time_limit_s,
            }
        )
        return gp.Model("sensei_oracle", env=env)

    def _add_point_variables(
        self, m: gp.Model, columns: list[str], flip_set: tuple[str, ...]
    ) -> tuple[dict[str, gp.Var], dict[str, gp.Var], dict[str, gp.Var]]:
        """
        x1 gets one [0,1] variable per feature. x2 shares x1's variable for
        every feature OUTSIDE `flip_set` (they're literally the same
        variable, not just constrained equal -- x1/x2 can only differ on
        `flip_set` by construction) and gets its own fresh variable for
        each feature inside it (`x2_only_vars`).

        Args:
            m (gp.Model): The Gurobi model to add variables to.
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.

        Returns:
            tuple[dict[str, gp.Var], dict[str, gp.Var], dict[str, gp.Var]]:
                `(x1_vars, x2_only_vars, x2_vars)` -- `x2_vars` already
                merges `x1_vars` and `x2_only_vars` per-feature.
        """

        x1_vars: dict[str, gp.Var] = {
            f: m.addVar(lb=0.0, ub=1.0, name=f"x1_{f}") for f in columns
        }
        x2_only_vars: dict[str, gp.Var] = {
            f: m.addVar(lb=0.0, ub=1.0, name=f"x2_{f}") for f in flip_set
        }
        x2_vars: dict[str, gp.Var] = {
            f: (x2_only_vars[f] if f in flip_set else x1_vars[f]) for f in columns
        }
        return x1_vars, x2_only_vars, x2_vars

    def _enforce_validity(
        self,
        m: gp.Model,
        x1_vars: dict[str, gp.Var],
        x2_vars: dict[str, gp.Var],
        flip_set: tuple[str, ...],
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
        bins: FrozenBins | None,
        theta: float,
        structure: TreeStructure,
    ) -> None:
        """
        Every validity/plausibility rule (Q1 integrality + Q2), on both
        points. Q2 is emitted once for the pair (`emit_plausibility_pair`),
        not once per point -- see that method's docstring for why emitting
        it independently for x1 and x2 is a correctness bug, not just
        redundant.

        Args:
            m (gp.Model): The Gurobi model to add constraints to.
            x1_vars (dict[str, gp.Var]): x1's per-feature decision variables.
            x2_vars (dict[str, gp.Var]): x2's per-feature decision variables.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            spec (Spec): Declares integer features and every other Q1/Q2 rule.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            bins (FrozenBins | None): Frozen plausibility bins; Q2 is
                skipped entirely if None.
            theta (float): Plausibility threshold for Q2.
            structure (TreeStructure): Tree structure to snap/validate
                thresholds against before encoding.

        Raises:
            ValueError: `assert_no_cross_threshold_collisions` found a tree
                threshold trapping an achievable bin edge or integrality
                grid point that float32/float64 snapping didn't resolve --
                see its docstring.
        """

        MilpValidityEncoder.snap_thresholds_to_achievable_points(
            structure, spec, feature_bounds, bins
        )
        MilpValidityEncoder.assert_no_cross_threshold_collisions(
            structure, spec, feature_bounds, bins
        )

        TreeEncoder.add_integrality_constraints(
            m, x1_vars, spec.integer_features, feature_bounds, "x1"
        )
        TreeEncoder.add_integrality_constraints(
            m, x2_vars, spec.integer_features, feature_bounds, "x2"
        )
        MilpValidityEncoder.emit_all(m, x1_vars, spec, feature_bounds)
        MilpValidityEncoder.emit_all(m, x2_vars, spec, feature_bounds)

        if bins is not None:
            MilpValidityEncoder.emit_plausibility_pair(
                m, x1_vars, x2_vars, flip_set, bins, theta, "q2"
            )

    def _pin_monotone_direction(
        self,
        m: gp.Model,
        x1_vars: dict[str, gp.Var],
        x2_vars: dict[str, gp.Var],
        flip_set: tuple[str, ...],
        spec: Spec,
    ) -> None:
        """
        Pins x1/x2's order on the monotone feature so that maximizing
        E(x1) - E(x2) searches for a WRONG-direction violation, whichever way
        the spec declares this feature. The objective is always
        E(x1) - E(x2) (never changes); what changes is which point is pinned
        to be the "should score no higher" side:

        - increasing: x1 <= x2 pinned. A violation is a LOWER feature value
          (x1) scoring HIGHER than a higher value (x2), i.e. E(x1) > E(x2).
        - decreasing: x2 <= x1 pinned (the pin is flipped, not the objective).
          A violation is a HIGHER feature value (x1) scoring HIGHER than a
          lower value (x2), i.e. E(x1) > E(x2) -- which for a decreasing
          feature means raising it increased the score, the wrong way.

        Pinning x1 <= x2 unconditionally (the previous bug) is only correct
        for increasing features; for decreasing features it searches for the
        CORRECT behavior and calls it a violation, never the actual one.

        Args:
            m (gp.Model): The Gurobi model to add the pin constraint to.
            x1_vars (dict[str, gp.Var]): x1's per-feature decision variables.
            x2_vars (dict[str, gp.Var]): x2's per-feature decision variables.
            flip_set (tuple[str, ...]): Must contain exactly one feature --
                the monotone feature being checked.
            spec (Spec): Declares `monotone[mono_feature]`'s direction.

        Raises:
            ValueError: `spec.monotone[mono_feature]` is neither
                `Direction.INCREASING` nor `Direction.DECREASING`.
        """

        assert len(flip_set) == 1, (
            "monotone_wrong direction expects exactly one feature"
        )
        (mono_feature,) = flip_set
        assert mono_feature in spec.monotone, (
            f"'{mono_feature}' is not declared in spec.monotone -- "
            f"direction='monotone_wrong' only makes sense for a declared monotone "
            f"feature"
        )

        mono_direction: Direction = spec.monotone[mono_feature]

        if mono_direction == Direction.INCREASING:
            m.addConstr(
                x1_vars[mono_feature] <= x2_vars[mono_feature], name="monotone_pin"
            )
        elif mono_direction == Direction.DECREASING:
            m.addConstr(
                x2_vars[mono_feature] <= x1_vars[mono_feature], name="monotone_pin"
            )
        else:
            raise ValueError(
                f"unknown monotone direction {mono_direction!r} for '{mono_feature}'"
            )

    def _encode_leaf_indicators(
        self,
        m: gp.Model,
        structure: TreeStructure,
        columns: list[str],
        flip_set: tuple[str, ...],
        x1_vars: dict[str, gp.Var],
        x2_only_vars: dict[str, gp.Var],
    ) -> tuple[dict[int, gp.Var], dict[int, gp.Var]]:
        """
        Full order-variable + leaf-indicator encoding for x1 (every
        threshold), then the SAME for x2 -- except x2 only needs its own
        fresh order variables for thresholds on `flip_set` features (prefix
        "b"); everywhere else it reuses x1's order variables directly
        (prefix "a"), since x1 and x2 share those feature values by
        construction.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            structure (TreeStructure): Supplies thresholds, leaf paths, and
                tree-to-leaves maps.
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            x1_vars (dict[str, gp.Var]): x1's per-feature decision variables.
            x2_only_vars (dict[str, gp.Var]): x2's own decision variables,
                one per feature in `flip_set`.

        Returns:
            tuple[dict[int, gp.Var], dict[int, gp.Var]]: `(ell1, ell2)`,
                each a global-leaf-index -> indicator-variable map.
        """

        order_vars1: dict[tuple[str, float], gp.Var] = TreeEncoder.add_order_variables(
            m, structure.thresholds, columns, "a"
        )
        TreeEncoder.link_order_variables(m, order_vars1, x1_vars, structure.thresholds)
        ell1: dict[int, gp.Var] = TreeEncoder.add_leaf_indicators(
            m, structure, order_vars1, "a"
        )

        flip_thresholds: dict[str, list[float]] = {
            f: t for f, t in structure.thresholds.items() if f in flip_set
        }
        order_vars2_only: dict[tuple[str, float], gp.Var] = (
            TreeEncoder.add_order_variables(m, flip_thresholds, list(flip_set), "b")
        )
        TreeEncoder.link_order_variables(
            m, order_vars2_only, x2_only_vars, flip_thresholds
        )
        order_vars2: dict[tuple[str, float], gp.Var] = {
            (f, t): (order_vars2_only[(f, t)] if f in flip_set else order_vars1[(f, t)])
            for f, ts in structure.thresholds.items()
            for t in ts
        }
        ell2: dict[int, gp.Var] = TreeEncoder.add_leaf_indicators(
            m, structure, order_vars2, "b"
        )
        return ell1, ell2

    def _set_gap_objective(
        self,
        m: gp.Model,
        leaf_map: LeafMap,
        structure: TreeStructure,
        ell1: dict[int, gp.Var],
        ell2: dict[int, gp.Var],
        v: NDArray[np.float64] | None,
        mode: str,
        eps: float,
    ) -> None:
        """
        Maximize the signed gap E_v(x1) - E_v(x2), where E_v(x) = base_score
        + sum of the leaf values `v` (or `v0` if `v` is None) reached by
        each active leaf. In feasibility mode this also adds `gap >= eps +
        1e-9` as a hard constraint, so the solve stops the moment any
        feasible violation is found rather than searching for the worst one.
        `eps` is otherwise unused (mode="optimality" solves to the true
        worst gap instead).

        Args:
            m (gp.Model): The Gurobi model to set the objective on.
            leaf_map (LeafMap): Supplies `v0`/`base_score`.
            structure (TreeStructure): Supplies `leaf_paths` (which global
                leaves exist).
            ell1 (dict[int, gp.Var]): x1's leaf-indicator variables.
            ell2 (dict[int, gp.Var]): x2's leaf-indicator variables.
            v (NDArray[np.float64] | None): Leaf values to score with;
                `leaf_map.v0` if None.
            mode (str): `"feasibility"` or `"optimality"`.
            eps (float): Sensitivity budget; used only in feasibility mode.
        """

        # objective: maximize the signed gap E_v(x1) - E_v(x2)
        v_active = leaf_map.v0 if v is None else v
        E1: gp.LinExpr = leaf_map.base_score + gp.quicksum(
            float(v_active[g]) * ell1[g] for g in structure.leaf_paths
        )
        E2: gp.LinExpr = leaf_map.base_score + gp.quicksum(
            float(v_active[g]) * ell2[g] for g in structure.leaf_paths
        )
        gap_expr: gp.LinExpr = E1 - E2

        if mode == "feasibility":
            m.addConstr(gap_expr >= eps + 1e-9, name="gap_violation")

        m.setObjective(gap_expr, GRB.MAXIMIZE)

    def _extract_result(
        self,
        m: gp.Model,
        x1_vars: dict[str, gp.Var],
        x2_vars: dict[str, gp.Var],
        ell1: dict[int, gp.Var],
        ell2: dict[int, gp.Var],
        structure: TreeStructure,
        columns: list[str],
        flip_set: tuple[str, ...],
        direction: str,
        time_limit_s: float,
    ) -> Pair | None:
        """
        None on a proven-infeasible model. `OracleTimeout` if the solver has
        neither a solution nor an infeasibility proof (unknown, never
        treated as None). Otherwise reads x1/x2's values and both leaf sets
        (`ell[g].X > 0.5`) off the solved model into a `Pair`, asserting
        `gap >= 0` for `direction="protected"` -- the whole point of that
        direction's symmetric search is that a negative gap here means the
        symmetry assumption itself broke.

        Args:
            m (gp.Model): The solved Gurobi model.
            x1_vars (dict[str, gp.Var]): x1's per-feature decision variables.
            x2_vars (dict[str, gp.Var]): x2's per-feature decision variables.
            ell1 (dict[int, gp.Var]): x1's leaf-indicator variables.
            ell2 (dict[int, gp.Var]): x2's leaf-indicator variables.
            structure (TreeStructure): Supplies `leaf_paths` (which global
                leaves to read `.X` off).
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 were allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            time_limit_s (float): Only used in the timeout error message.

        Returns:
            Pair | None: The solved pair, or None on proven infeasibility.

        Raises:
            OracleTimeout: The solver stopped with no best-so-far solution
                and no infeasibility proof within `time_limit_s`.
        """

        if m.Status == GRB.INFEASIBLE:
            return None

        if m.SolCount == 0:
            raise OracleTimeout(
                f"oracle stopped (status={m.Status}) with no best-so-far solution and "
                f"no infeasibility proof within {time_limit_s}s -- unknown, not None"
            )

        x1_val: NDArray[np.float64] = np.array(
            [x1_vars[f].X for f in columns], dtype=float
        )
        x2_val: NDArray[np.float64] = np.array(
            [x2_vars[f].X for f in columns], dtype=float
        )
        ell1_val: NDArray[np.int64] = np.array(
            [g for g in structure.leaf_paths if ell1[g].X > 0.5], dtype=np.int64
        )
        ell2_val: NDArray[np.int64] = np.array(
            [g for g in structure.leaf_paths if ell2[g].X > 0.5], dtype=np.int64
        )
        gap_val = float(m.ObjVal)

        if direction == "protected":
            assert gap_val >= -1e-6, (
                f"direction='protected' assumes symmetry (maximizing the signed gap "
                f"covers both directions) -- a negative gap ({gap_val}) means that "
                f"assumption broke"
            )

        solver_status: Literal["OPTIMAL"] | Literal["FEASIBLE"] = (
            "OPTIMAL" if m.Status == GRB.OPTIMAL else "FEASIBLE"
        )

        return Pair(
            x1=x1_val,
            x2=x2_val,
            ell1=ell1_val,
            ell2=ell2_val,
            gap=gap_val,
            flip_set=flip_set,
            direction=direction,
            solver_status=solver_status,
        )

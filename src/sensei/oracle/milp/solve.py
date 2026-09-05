from typing import Literal

import gurobipy as gp
import numpy as np
import xgboost as xgb
from gurobipy import GRB
from numpy.typing import NDArray

from sensei.data.bins import FrozenBins
from sensei.model.leaves import LeafMap
from sensei.oracle.milp.encoding import TreeEncoder, TreeStructure
from sensei.oracle.milp.nogoods import NoGood, NoGoodBuilder
from sensei.oracle.milp.warm_start import WarmStartApplier
from sensei.oracle.types import OracleTimeout, Pair
from sensei.spec import Spec
from sensei.validity.encode_milp import MilpValidityEncoder


class TierAOracle:
    """
    A sensitivity oracle that finds the worst valid pair of points in a given context.
    """

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
        eps: float,
        seed: int,
        time_limit_s: float,
        mip_gap: float,
        nogoods: list[NoGood] | None = None,
        warm_start: Pair | None = None,
        enforce_validity: bool = True,
        structure: TreeStructure | None = None,
        v: NDArray[np.float64] | None = None,
        bins: FrozenBins | None = None,
        theta: float | None = None,
    ) -> Pair | None:
        """
        Finds the worst valid pair of points in the given context using a mixed-integer
        linear programming (MILP) formulation. The oracle constructs a Gurobi model to
        maximize the signed gap between the predictions of the two points while
        enforcing validity constraints and other specified conditions.

        Args:
            booster (xgb.Booster): The trained XGBoost model used for predictions.
            leaf_map (LeafMap): The leaf map to use for mapping points to leaves.
            columns (list[str]): The list of column names in the dataset.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.
            spec (Spec | None): The specification to enforce.
            flip_set (tuple[str, ...]): The set of features to flip.
            direction (str): The direction of the gap to maximize ("protected" or
                             "monotone_wrong").
            seed (int): The random seed for reproducibility.
            time_limit_s (float): The time limit for the optimization in seconds.
            mip_gap (float): The MIP gap tolerance for the optimization.
            nogoods (list[NoGood] | None, optional): The list of no-good constraints to
                                                     include. Defaults to None.
            warm_start (Pair | None, optional): The initial solution to start the
                                                optimization from. Defaults to None.
            enforce_validity (bool, optional): Whether to enforce validity constraints.
                                               Defaults to True.
            structure (TreeStructure | None, optional): The tree structure to use.
                                                        Defaults to None.
            v (NDArray[np.float64] | None, optional): The array of values. Defaults to
                                                      None.
            bins (FrozenBins | None, optional): The frozen bins to use. Defaults to
                                                None.
            theta (float | None, optional): The threshold value. Defaults to None.

        Returns:
            Pair | None: The worst valid pair of points found, or None if no feasible
                         solution exists.
        """

        assert mode in ("feasibility", "optimality")
        assert mode != "feasibility" or eps is not None, (
            "eps is required when mode='feasibility'"
        )
        assert direction in ("protected", "monotone_wrong")

        if structure is None:
            structure = TreeEncoder.extract_tree_structure(
                booster, leaf_map.leaf_index, columns
            )

        model: gp.Model = self._build_model(mode, seed, time_limit_s, mip_gap)
        x1_vars, x2_only_vars, x2_vars = self._add_point_variables(
            model, columns, flip_set
        )

        if enforce_validity:
            assert spec is not None, "enforce_validity=True requires a spec"
            self._enforce_validity(
                model, x1_vars, x2_vars, spec, feature_bounds, bins, theta
            )

        if direction == "monotone_wrong":
            self._pin_monotone_direction(model, x1_vars, x2_vars, flip_set)

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
            time_limit_s,
        )

    def _build_model(
        self, mode: str, seed: int, time_limit_s: float, mip_gap: float
    ) -> gp.Model:
        """
        Builds a Gurobi model with the specified parameters for the optimization.

        Args:
            mode (str): The mode of the optimization, either "feasibility" or
                        "optimality".
            seed (int): The random seed for reproducibility.
            time_limit_s (float): The time limit for the optimization in seconds.
            mip_gap (float): The MIP gap for the optimization.

        Returns:
            gp.Model: The constructed Gurobi model ready for optimization.
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
        return gp.Model("sensei_tier_a", env=env)

    def _add_point_variables(
        self, m: gp.Model, columns: list[str], flip_set: tuple[str, ...]
    ) -> tuple[dict[str, gp.Var], dict[str, gp.Var], dict[str, gp.Var]]:
        """
        Adds point variables to the model.

        Args:
            m (gp.Model): The Gurobi model.
            columns (list[str]): The list of column names.
            flip_set (tuple[str, ...]): The set of features to flip.

        Returns:
            tuple[dict[str, gp.Var], dict[str, gp.Var], dict[str, gp.Var]]:
                A tuple containing the x1, x2_only, and x2 variables.
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
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
        bins: FrozenBins | None,
        theta: float | None,
    ) -> None:
        """
        Enforces validity constraints on the model for both points.

        Args:
            m (gp.Model): The Gurobi model to which the validity constraints will be
                          added.
            x1_vars (dict[str, gp.Var]): A dictionary mapping feature names to Gurobi
                                         variables for the first point.
            x2_vars (dict[str, gp.Var]): A dictionary mapping feature names to Gurobi
                                         variables for the second point.
            spec (Spec): The dataset specification containing the validity constraints.
            feature_bounds (dict[str, tuple[float, float]]): A dictionary mapping
                                                             feature names to their
                                                             (min, max) bounds.
            bins (FrozenBins | None): The frozen bins to use for encoding the validity
                                      constraints. If None, no binning is applied.
            theta (float | None): The threshold value for the validity constraints. If
                                  None, no thresholding is applied.
        """

        TreeEncoder.add_integrality_constraints(
            m, x1_vars, spec.integer_features, feature_bounds, "x1"
        )
        TreeEncoder.add_integrality_constraints(
            m, x2_vars, spec.integer_features, feature_bounds, "x2"
        )
        MilpValidityEncoder.emit_all(m, x1_vars, spec, feature_bounds, bins, theta)
        MilpValidityEncoder.emit_all(m, x2_vars, spec, feature_bounds, bins, theta)

    def _pin_monotone_direction(
        self,
        m: gp.Model,
        x1_vars: dict[str, gp.Var],
        x2_vars: dict[str, gp.Var],
        flip_set: tuple[str, ...],
    ) -> None:
        """
        Pins the monotone direction for the specified feature in the model.

        Args:
            m (gp.Model): The Gurobi model to which the monotone direction constraint
                          will be added.
            x1_vars (dict[str, gp.Var]): A dictionary mapping feature names to Gurobi
                                         variables for the first point.
            x2_vars (dict[str, gp.Var]): A dictionary mapping feature names to Gurobi
                                         variables for the second point.
            flip_set (tuple[str, ...]): A tuple of feature names for which the monotone
                                        direction will be pinned.
        """

        assert len(flip_set) == 1, (
            "monotone_wrong direction expects exactly one feature"
        )
        (mono_feature,) = flip_set
        m.addConstr(x1_vars[mono_feature] <= x2_vars[mono_feature], name="monotone_pin")

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
        Encodes the leaf indicators for both points in the model.

        Args:
            m (gp.Model): The Gurobi model to which the leaf indicator constraints will
                          be added.
            structure (TreeStructure): The tree structure for which to encode leaf
                                       indicators.
            columns (list[str]): The list of column names in the dataset.
            flip_set (tuple[str, ...]): The set of features for which to flip the
                                        monotone direction.
            x1_vars (dict[str, gp.Var]): A dictionary mapping feature names to Gurobi
                                         variables for the first point.
            x2_only_vars (dict[str, gp.Var]): A dictionary mapping feature names to
                                              Gurobi variables for the second point.

        Returns:
            tuple[dict[int, gp.Var], dict[int, gp.Var]]: A tuple containing two
                                                         dictionaries mapping global
                                                         leaf indices to Gurobi
                                                         variables for the first and
                                                         second points, respectively.
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
        Sets the objective function for the gap optimization problem.

        Args:
            m (gp.Model): The Gurobi model.
            leaf_map (LeafMap): The leaf map.
            structure (TreeStructure): The tree structure.
            ell1 (dict[int, gp.Var]): A dictionary mapping global leaf indices to Gurobi
                                      variables for the first point.
            ell2 (dict[int, gp.Var]): A dictionary mapping global leaf indices to Gurobi
                                      variables for the second point.
            v (NDArray[np.float64] | None): The feature values.
            mode (str): The optimization mode.
            eps (float): The epsilon value.
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
        Extracts the result from the Gurobi model after optimization.

        Args:
            m (gp.Model): The Gurobi model after optimization.
            x1_vars (dict[str, gp.Var]): A dictionary mapping feature names to Gurobi
                                         variables for the first point.
            x2_vars (dict[str, gp.Var]): A dictionary mapping feature names to Gurobi
                                         variables for the second point.
            ell1 (dict[int, gp.Var]): A dictionary mapping global leaf indices to Gurobi
                                      variables for the first point.
            ell2 (dict[int, gp.Var]): A dictionary mapping global leaf indices to Gurobi
                                      variables for the second point.
            structure (TreeStructure): The tree structure used in the optimization.
            columns (list[str]): The list of column names in the dataset.
            flip_set (tuple[str, ...]): The set of features that were allowed to flip.
            direction (str): The direction of the gap that was maximized ("protected" or
                             "monotone_wrong").
            time_limit_s (float): The time limit for the optimization in seconds.

        Raises:
            OracleTimeout: If the oracle stops with no best-so-far solution and no
                           infeasibility proof within the specified time limit.

        Returns:
            Pair | None: The worst valid pair of points found, or None if no feasible
                         solution exists.
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

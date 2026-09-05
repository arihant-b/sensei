import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from sensei.data.bins import FrozenBins
from sensei.oracle.milp.encoding import TreeEncoder
from sensei.spec import Spec


class MilpValidityEncoder:
    """
    Emit Q1/Q2 validity constraints directly into Gurobi, over a feature-name -> gp.Var
    mapping, so validity lives inside the search.
    """

    @staticmethod
    def emit_all(
        m: gp.Model,
        x_vars: dict[str, gp.Var],
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
        bins: FrozenBins | None = None,
        theta: float | None = None,
    ) -> None:
        """
        Emit all validity constraints into the Gurobi model.

        Args:
            m (gp.Model): The Gurobi model to which the validity constraints will be
                          added.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            spec (Spec): The dataset specification containing the validity rules.
            feature_bounds (dict[str, tuple[float, float]]): A dictionary mapping
                                                             feature names to their
                                                             lower and upper bounds.
            bins (FrozenBins | None, optional): The frozen bins for plausibility
                                                constraints. Defaults to None.
            theta (float | None, optional): The threshold for plausibility constraints.
                                            Defaults to None.
        """

        name_prefix: str = f"v{id(x_vars) % 100000}"
        MilpValidityEncoder.emit_ranges(m, x_vars, spec)
        MilpValidityEncoder.emit_one_hot_groups(m, x_vars, spec)
        MilpValidityEncoder.emit_domain_rules(m, x_vars, spec)
        MilpValidityEncoder.emit_functional_deps(
            m, x_vars, spec, feature_bounds, name_prefix
        )

        if bins is not None and theta is not None:
            MilpValidityEncoder.emit_plausibility(
                m, x_vars, bins, theta, f"{name_prefix}_q2"
            )

    @staticmethod
    def emit_ranges(m: gp.Model, x_vars: dict[str, gp.Var], spec: Spec) -> None:
        """
        Emit range constraints for each feature in the Gurobi model.

        Args:
            m (gp.Model): The Gurobi model to which the range constraints will be added.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            spec (Spec): The dataset specification containing the feature ranges.
        """

        for f, (lo, hi) in spec.ranges.items():
            if f in x_vars:
                m.addConstr(x_vars[f] >= lo)
                m.addConstr(x_vars[f] <= hi)

    @staticmethod
    def emit_one_hot_groups(m: gp.Model, x_vars: dict[str, gp.Var], spec: Spec) -> None:
        """
        Emit one-hot encoding constraints for each group of features in the Gurobi
        model.

        Args:
            m (gp.Model): The Gurobi model to which the one-hot encoding constraints
                          will be added.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            spec (Spec): The dataset specification containing the one-hot group
                         information.
        """

        for columns in spec.one_hot_groups.values():
            present: list[str] = list(filter(lambda c: c in x_vars, columns))

            if len(present) != len(columns):
                continue

            for c in present:
                x_vars[c].VType = GRB.BINARY

            m.addConstr(gp.quicksum(x_vars[c] for c in present) == 1)

    @staticmethod
    def emit_domain_rules(m: gp.Model, x_vars: dict[str, gp.Var], spec: Spec) -> None:
        """
        Emit domain rule constraints for each domain rule in the Gurobi model.

        Args:
            m (gp.Model): The Gurobi model to which the domain rule constraints will be
                          added.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            spec (Spec): The dataset specification containing the domain rule
                         information.
        """

        sense_map: dict[str, str] = {
            "<=": GRB.LESS_EQUAL,
            ">=": GRB.GREATER_EQUAL,
            "==": GRB.EQUAL,
        }

        for rule in spec.domain_rules:
            if not all(f in x_vars for f in rule.coefficients):
                continue

            expr: gp.LinExpr = gp.quicksum(
                coef * x_vars[f] for f, coef in rule.coefficients.items()
            )
            sense: str = sense_map[rule.sense]

            if sense == GRB.LESS_EQUAL:
                m.addConstr(expr <= rule.rhs)
            elif sense == GRB.GREATER_EQUAL:
                m.addConstr(expr >= rule.rhs)
            else:
                m.addConstr(expr == rule.rhs)

    @staticmethod
    def emit_functional_deps(
        m: gp.Model,
        x_vars: dict[str, gp.Var],
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
        name_prefix: str,
    ) -> None:
        """
        Emit functional dependency constraints for each functional dependency in the
        Gurobi model.

        Args:
            m (gp.Model): The Gurobi model to which the functional dependency
                          constraints will be added.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            spec (Spec): The dataset specification containing the functional
                         dependency information.
            feature_bounds (dict[str, tuple[float, float]]): A dictionary mapping
                                                             feature names to their
                                                             corresponding value
                                                             bounds.
            name_prefix (str): A prefix to be used for naming the generated
                               variables and constraints.
        """

        for i, (if_col, if_val, then_col, then_val) in enumerate(spec.functional_deps):
            if (
                if_col not in x_vars
                or then_col not in x_vars
                or if_col not in feature_bounds
            ):
                continue

            lo, hi = feature_bounds[if_col]
            span = int(round(hi - lo))

            if span <= 0:
                continue

            aux: gp.Var = m.addVar(
                vtype=GRB.INTEGER, lb=0, ub=span, name=f"{name_prefix}_fd{i}_aux"
            )
            m.addConstr(x_vars[if_col] * span == aux)

            target = int(round(if_val * span))
            b: gp.Var = MilpValidityEncoder._exact_integer_indicator(
                m, aux, target, 0, span, f"{name_prefix}_fd{i}"
            )
            m.addGenConstrIndicator(b, True, x_vars[then_col], GRB.EQUAL, then_val)

    @staticmethod
    def emit_plausibility(
        m: gp.Model,
        x_vars: dict[str, gp.Var],
        bins: FrozenBins,
        theta: float,
        name_prefix: str,
    ) -> None:
        """
        Emit plausibility constraints based on frozen bins and a threshold theta.

        Args:
            m (gp.Model): The Gurobi model to which the plausibility constraints will be
                          added.
            x_vars (dict[str, gp.Var]): A dictionary mapping feature names to their
                                        corresponding Gurobi variables.
            bins (FrozenBins): The frozen bins to use for generating the plausibility
                               constraints.
            theta (float): The threshold value for the plausibility constraints.
            name_prefix (str): A prefix to be used for naming the generated variables
                               and constraints.
        """

        bin_thresholds: dict[str, list[Any]] = {
            f: list(edges[1:-1]) for f, edges in bins.edges.items() if f in x_vars
        }

        order_vars: dict[tuple[str, float], gp.Var] = TreeEncoder.add_order_variables(
            m, bin_thresholds, list(bin_thresholds.keys()), name_prefix
        )
        TreeEncoder.link_order_variables(m, order_vars, x_vars, bin_thresholds)

        log_terms: list[gp.LinExpr] = []

        for f, interior in bin_thresholds.items():
            n_bins: int = len(interior) + 1

            for i in range(n_bins):
                literals = []

                if i > 0:
                    literals.append(1 - order_vars[(f, interior[i - 1])])
                if i < len(interior):
                    literals.append(order_vars[(f, interior[i])])

                if not literals:
                    # only one bin total for this feature (constant column) --
                    # always active, no indicator variable needed
                    log_terms.append(bins.log_p[f][i])
                    continue

                active: gp.Var = m.addVar(
                    vtype=GRB.BINARY, name=f"{name_prefix}_bin_{f}_{i}"
                )

                for lit in literals:
                    m.addConstr(active <= lit)

                m.addConstr(active >= gp.quicksum(literals) - (len(literals) - 1))

                log_terms.append(float(bins.log_p[f][i]) * active)

        m.addConstr(
            gp.quicksum(log_terms) >= math.log(theta), name=f"{name_prefix}_plaus"
        )

    @staticmethod
    def _exact_integer_indicator(
        m: gp.Model,
        aux: gp.Var,
        target: int,
        lo_bound: int,
        hi_bound: int,
        name: str,
    ) -> gp.Var:
        """
        Create a binary indicator variable that is 1 if and only if the auxiliary
        variable equals the target integer value.

        Args:
            m (gp.Model): The Gurobi model to which the indicator variable and
                          constraints will be added.
            aux (gp.Var): The auxiliary variable to be compared against the target.
            target (int): The target integer value.
            lo_bound (int): The lower bound of the auxiliary variable.
            hi_bound (int): The upper bound of the auxiliary variable.
            name (str): The name prefix for the generated variables and constraints.

        Returns:
            gp.Var: The binary indicator variable.
        """

        below: gp.Var = m.addVar(vtype=GRB.BINARY, name=f"{name}_below")
        above: gp.Var = m.addVar(vtype=GRB.BINARY, name=f"{name}_above")

        if target - 1 >= lo_bound:
            m.addGenConstrIndicator(below, True, aux, GRB.LESS_EQUAL, target - 1)
            m.addGenConstrIndicator(below, False, aux, GRB.GREATER_EQUAL, target)
        else:
            m.addConstr(below == 0)

        if target + 1 <= hi_bound:
            m.addGenConstrIndicator(above, True, aux, GRB.GREATER_EQUAL, target + 1)
            m.addGenConstrIndicator(above, False, aux, GRB.LESS_EQUAL, target)
        else:
            m.addConstr(above == 0)

        not_b: gp.Var = m.addVar(vtype=GRB.BINARY, name=f"{name}_not_eq")
        m.addGenConstrOr(not_b, [below, above])
        b: gp.Var = m.addVar(vtype=GRB.BINARY, name=f"{name}_eq")
        m.addConstr(b == 1 - not_b)
        return b

import math
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from sensei.data.bins import FrozenBins
from sensei.oracle.sensei.encoding import SPLIT_EPS, TreeEncoder
from sensei.oracle.types import TreeStructure
from sensei.spec import Spec

_COINCIDENCE_TOL = 1e-6


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
    ) -> None:
        """
        Emit ranges, one-hot groups, domain rules, and FDs for ONE point's
        variables (Q1 only -- plausibility is per-pair, see
        `emit_plausibility_pair`, since it must not be built independently
        for x1 and x2 on features they share).

        Args:
            m (gp.Model): The Gurobi model to add constraints to.
            x_vars (dict[str, gp.Var]): One point's per-feature decision
                variables.
            spec (Spec): Declares every rule to emit.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
        """

        name_prefix: str = f"v{id(x_vars) % 100000}"
        MilpValidityEncoder.emit_ranges(m, x_vars, spec)
        MilpValidityEncoder.emit_one_hot_groups(m, x_vars, spec)
        MilpValidityEncoder.emit_domain_rules(m, x_vars, spec)
        MilpValidityEncoder.emit_functional_deps(
            m, x_vars, spec, feature_bounds, name_prefix
        )

    @staticmethod
    def emit_ranges(m: gp.Model, x_vars: dict[str, gp.Var], spec: Spec) -> None:
        """
        Emit `spec.ranges[f] = (lo, hi)` as `lo <= x_vars[f] <= hi` for each f.

        Args:
            m (gp.Model): The Gurobi model to add constraints to.
            x_vars (dict[str, gp.Var]): One point's per-feature decision
                variables.
            spec (Spec): Declares `ranges`.
        """

        for f, (lo, hi) in spec.ranges.items():
            if f in x_vars:
                m.addConstr(x_vars[f] >= lo)
                m.addConstr(x_vars[f] <= hi)

    @staticmethod
    def emit_one_hot_groups(m: gp.Model, x_vars: dict[str, gp.Var], spec: Spec) -> None:
        """
        Force each `spec.one_hot_groups` column to BINARY, group summing to 1.

        Args:
            m (gp.Model): The Gurobi model to add constraints to.
            x_vars (dict[str, gp.Var]): One point's per-feature decision
                variables.
            spec (Spec): Declares `one_hot_groups`.
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
        Emit each `spec.domain_rules` linear inequality/equality directly.

        Args:
            m (gp.Model): The Gurobi model to add constraints to.
            x_vars (dict[str, gp.Var]): One point's per-feature decision
                variables.
            spec (Spec): Declares `domain_rules`.
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
        For each `spec.functional_deps` (IF `if_col`=`if_val` THEN `then_col`=
        `then_val`), build an exact-integer indicator on `if_col`'s rescaled
        integer code and use it to force `then_col` via a Gurobi indicator
        constraint.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            x_vars (dict[str, gp.Var]): One point's per-feature decision
                variables.
            spec (Spec): Declares `functional_deps`.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            name_prefix (str): Prefix for generated variable names.
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
    def snap_thresholds_to_achievable_points(
        structure: TreeStructure,
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
        bins: FrozenBins | None,
    ) -> None:
        """
        Mutates `structure.thresholds` IN PLACE: any tree split threshold
        within `_COINCIDENCE_TOL` of an achievable plausibility bin edge or
        integrality grid point for that feature is replaced by the EXACT
        achievable value (module docstring above explains why this
        noise exists). This must run before `structure.thresholds` is
        used to build order variables (`TreeEncoder.add_order_variables`/
        `link_order_variables`) -- both here in `_enforce_validity` and
        later in `_encode_leaf_indicators`, which reads the SAME
        `structure` object -- since it corrects the very values those
        constraints are built from, not just a diagnostic reading of them.

        This resolves float32/float64 representation noise at the source,
        restoring the achievable point's true reachability in the MILP,
        rather than merely silencing `assert_no_cross_threshold_collisions`
        (which still runs afterward, and will still raise on any remaining
        gap between `_COINCIDENCE_TOL` and `SPLIT_EPS` -- a genuinely
        suspicious near-miss, not routine float precision noise).

        A no-op for a feature with no achievable points to snap toward
        (not in `bins.edges` and not in `spec.integer_features`).

        Args:
            structure (TreeStructure): Mutated in place -- its `thresholds`
                are snapped where within tolerance of an achievable point.
            spec (Spec): Declares `integer_features`.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            bins (FrozenBins | None): Frozen plausibility bins; skipped if None.
        """

        for f, ts in structure.thresholds.items():
            achievable: list[float] = []

            if bins is not None and f in bins.edges:
                achievable.extend(bins.edges[f][1:-1])

            if f in spec.integer_features and f in feature_bounds:
                lo, hi = feature_bounds[f]
                span = hi - lo

                if span > 0:
                    n_levels = int(round(span))
                    achievable.extend(k / n_levels for k in range(n_levels + 1))

            if not achievable:
                continue

            snapped: set[float] = set()

            for t in ts:
                nearest: float = min(achievable, key=lambda p: abs(t - p))
                snapped.add(nearest if abs(t - nearest) < _COINCIDENCE_TOL else t)

            structure.thresholds[f] = sorted(snapped)

    @staticmethod
    def assert_no_cross_threshold_collisions(
        structure: TreeStructure,
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
        bins: FrozenBins | None,
    ) -> None:
        """
        For every feature, cross-check the tree's own split thresholds
        (`structure.thresholds`) against that feature's plausibility bin
        edges (`bins.edges`, if given) and its integrality grid points (if
        declared in `spec.integer_features`) -- the ONE thing
        `encoding.py::TreeEncoder._assert_thresholds_separated` cannot see,
        since it only checks separation WITHIN one threshold source's own
        list, never across the tree's thresholds and a completely different
        source built by a separate code path.

        A tree threshold `t` that lands strictly inside `(point - SPLIT_EPS,
        point)` for some achievable `point` (a bin edge or integrality grid
        point) makes that point structurally UNREACHABLE: `u[f,t]==1`
        requires `x <= t - SPLIT_EPS` (point is above that), and
        `u[f,t]==0` requires `x >= t` (point is below that) -- neither
        branch admits it. This is directional: `t <= point` is always safe
        (point stays reachable via `x >= t`); only `t` landing just ABOVE
        `point` traps it.

        Confirmed empirically: on a 200-tree/depth-4 Adult model with 10
        quantile bins, several features (age, fnlwgt, education-num,
        hours-per-week) had a tree threshold within ~1e-8 of a plausibility
        bin edge -- far closer than SPLIT_EPS=1e-4 -- and the resulting
        MILP was reported INFEASIBLE (a false EMPTY_DOMAIN) even though the
        Q1+Q2 domain and the tree encoding are each independently feasible.
        Both XGBoost's histogram splits and FrozenBins' quantile edges are
        derived from similar quantile statistics over the same training
        data, so they coincide more often as tree count grows (more
        candidate splits tried against the same handful of quantile edges).

        Raising here, before the expensive tree/leaf encoding is built,
        turns that silent false EMPTY_DOMAIN into an explicit, catchable
        error instead -- it does not change SPLIT_EPS, the plausibility
        model, or the binning, any of which would need to be a deliberate,
        declared decision, not a silent fix.

        Args:
            structure (TreeStructure): Supplies the tree's own split thresholds.
            spec (Spec): Declares `integer_features`.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            bins (FrozenBins | None): Frozen plausibility bins; skipped if None.

        Raises:
            ValueError: A tree threshold for some feature traps an
                achievable bin edge or integrality grid point as described
                above.
        """

        for f, tree_ts in structure.thresholds.items():
            if not tree_ts:
                continue

            check_points: list[tuple[float, str]] = []

            if bins is not None and f in bins.edges:
                check_points.extend(
                    (edge, "plausibility bin edge") for edge in bins.edges[f][1:-1]
                )

            if f in spec.integer_features and f in feature_bounds:
                lo, hi = feature_bounds[f]
                span = hi - lo

                if span > 0:
                    n_levels = int(round(span))
                    check_points.extend(
                        (k / n_levels, "integrality grid point")
                        for k in range(n_levels + 1)
                    )

            for t in tree_ts:
                for point, kind in check_points:
                    gap = t - point

                    if 0 < gap < SPLIT_EPS:
                        raise ValueError(
                            f"tree threshold {t} for feature '{f}' is only "
                            f"{gap:.2e} above {kind} {point:.10g} -- below "
                            f"SPLIT_EPS ({SPLIT_EPS:.2e}), which traps that point "
                            f"as structurally unreachable (see "
                            f"assert_no_cross_threshold_collisions's docstring). "
                            f"Do not widen SPLIT_EPS to force this to pass -- "
                            f"investigate why the tree's own split and this "
                            f"boundary landed this close together (more trees, a "
                            f"different n_quantile_bins, or a different model)."
                        )

    @staticmethod
    def emit_plausibility_pair(
        m: gp.Model,
        x1_vars: dict[str, gp.Var],
        x2_vars: dict[str, gp.Var],
        flip_set: tuple[str, ...],
        bins: FrozenBins,
        theta: float,
        name_prefix: str,
    ) -> None:
        """
        Q2 for x1 AND x2 together, in one call. For every feature OUTSIDE
        `flip_set`, `x1_vars[f] is x2_vars[f]` (the same Gurobi variable, by
        construction -- see `SenseiOracle._add_point_variables`), so its
        bin-membership encoding is built exactly ONCE here and reused in
        both points' plausibility sums. Only `flip_set` features get their
        own independent x1/x2 encoding, since those genuinely differ.

        Calling the single-point `_feature_log_terms` independently for x1
        and x2 (the previous approach) built TWO numerically near-duplicate
        copies of the same `SPLIT_EPS`-based threshold-linking constraints
        on the SAME shared variable -- harmless in principle, but at
        `SPLIT_EPS`/Gurobi's `FeasibilityTol`/`IntFeasTol` (1e-4/1e-9)
        scale, confirmed to tip large models (e.g. 200 trees, 10 quantile
        bins) into a false EMPTY_DOMAIN that a smaller model or fewer bins
        did not produce, even though the underlying validity domain and
        tree encoding are each independently feasible.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            x1_vars (dict[str, gp.Var]): x1's per-feature decision variables.
            x2_vars (dict[str, gp.Var]): x2's per-feature decision variables.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            bins (FrozenBins): Frozen plausibility bins.
            theta (float): Plausibility threshold.
            name_prefix (str): Prefix for generated variable names.
        """

        shared: list[str] = [f for f in x1_vars if f not in flip_set]
        shared_terms: list[gp.LinExpr] = MilpValidityEncoder._feature_log_terms(
            m, x1_vars, bins, shared, f"{name_prefix}_shared"
        )

        flip_features: list[str] = [f for f in flip_set if f in x1_vars]
        x1_flip_terms: list[gp.LinExpr] = MilpValidityEncoder._feature_log_terms(
            m, x1_vars, bins, flip_features, f"{name_prefix}_x1"
        )
        x2_flip_terms: list[gp.LinExpr] = MilpValidityEncoder._feature_log_terms(
            m, x2_vars, bins, flip_features, f"{name_prefix}_x2"
        )

        if theta > 0.0:
            # theta<=0.0 means "no plausibility floor" -- log_plaus(x) is always
            # finite (a sum of Laplace-smoothed log-probabilities, never -inf),
            # so `>= log(0.0)` would be a vacuously-true constraint anyway; skip
            # emitting it rather than crash on `math.log(0.0)` (ValueError:
            # math domain error).
            log_theta: float = math.log(theta)
            m.addConstr(
                gp.quicksum(shared_terms + x1_flip_terms) >= log_theta,
                name=f"{name_prefix}_plaus_x1",
            )
            m.addConstr(
                gp.quicksum(shared_terms + x2_flip_terms) >= log_theta,
                name=f"{name_prefix}_plaus_x2",
            )

    @staticmethod
    def _feature_log_terms(
        m: gp.Model,
        x_vars: dict[str, gp.Var],
        bins: FrozenBins,
        features: list[str],
        name_prefix: str,
    ) -> list[gp.LinExpr]:
        """
        One bin-membership indicator per (feature, bin) for `features` only,
        linked to `x_vars` via `TreeEncoder`'s order-variable machinery,
        returned as the log-weighted terms of `sum_f log p_f(bin)`. Factored
        out of the old single-point `emit_plausibility` so
        `emit_plausibility_pair` can call it once per underlying variable
        instead of once per point.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            x_vars (dict[str, gp.Var]): The point's per-feature decision
                variables to link bin membership to.
            bins (FrozenBins): Frozen plausibility bins.
            features (list[str]): Features to build bin-membership terms for.
            name_prefix (str): Prefix for generated variable names.

        Returns:
            list[gp.LinExpr]: The log-weighted bin-membership terms.
        """

        bin_thresholds: dict[str, list[Any]] = {
            f: list(bins.edges[f][1:-1]) for f in features if f in bins.edges
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

        return log_terms

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
        Binary `b == 1 <=> aux == target`, built from `below`/`above`
        indicators (each hardwired to 0 when `target` is already at
        `aux`'s bound).

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            aux (gp.Var): The integer variable to compare against `target`.
            target (int): The value `aux` must equal for `b` to be 1.
            lo_bound (int): `aux`'s lower bound.
            hi_bound (int): `aux`'s upper bound.
            name (str): Prefix for generated variable names.

        Returns:
            gp.Var: Binary indicator, 1 iff `aux == target`.
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

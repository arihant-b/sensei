from dataclasses import dataclass
from typing import Protocol

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from config import Config
from model import Ensemble
from spec import SensitivitySpec
from validity import ValidityChecker

# Strict tree-split inequalities (x_f < threshold) aren't representable in a
# MILP, which only has <=/>=/==. Following standard practice (Kantchelian et
# al. 2016), the 'yes' (true) branch is encoded as x_f <= threshold - EPS.
# EPS must clear Gurobi's own FeasibilityTol (default 1e-6) with real margin,
# or the solver can treat a point as satisfying both a leaf's path AND a
# sibling leaf's path near the boundary, and misreport which leaf is active.
_SPLIT_EPS = 1e-4


@dataclass
class Counterexample:
    x1: np.ndarray
    x2: np.ndarray
    gap: float
    leaves1: np.ndarray      # global leaf indices reached by x1
    leaves2: np.ndarray


class OracleLike(Protocol):
    """
    Structural type for SensEILoop's oracle param: SensitivityOracle's
    public surface, satisfied by any stand-in with the same shape (e.g.
    tests/fixtures/synthetic.FakeOracle -- CLAUDE.md's own recommended
    way to exercise repair.py/loop.py without a real solver).
    """

    def find_worst(
        self, model, flip_set: list[str], certifying: bool = False
    ) -> Counterexample | None: ...

    def find_batch(
        self, model, flip_set: list[str], k: int, certifying: bool = False
    ) -> list[Counterexample]: ...

    def add_nogood(self, ce: Counterexample) -> None: ...
    def clear_nogoods(self) -> None: ...
    def set_warm_start(self, ce: Counterexample) -> None: ...


class SensitivityOracle:
    """
    Wraps Ensense / the MILP encoding.

    This is the only component that can return UNSAT, i.e. PROVE that no
    valid undesirable counterexample exists. That proof is the whole point.
    """

    def __init__(
        self, validity: ValidityChecker, spec: SensitivitySpec, cfg: Config
    ) -> None:
        self.validity = validity
        self.spec = spec
        self.cfg = cfg
        self._nogoods: list[np.ndarray] = []      # forbidden leaf patterns
        self._warm_start: Counterexample | None = None

    def find_worst(
        self, model: Ensemble, flip_set: list[str], certifying: bool = False
    ) -> Counterexample | None:
        """
        max  E_v(x1) - E_v(x2)
        s.t. tree encoding (per-tree leaf-indicator variables),
             x1, x2 agree outside flip_set (shared variables, not equality
                constraints -- they literally cannot disagree),
             Gap-bin (x1, x2 confidently on opposite sides of the decision
                boundary: raw_score(x1) >= cfg.gap, raw_score(x2) <= -cfg.gap),
             linear validity constraints (self.validity.milp_constraints()),
             accumulated no-good cuts.

        Returns None when UNSAT -> certificate. `certifying=True` forces
        MIPGap=0 regardless of cfg.mip_gap (CLAUDE.md invariant 3): a
        nonzero-gap solve that times out with no incumbent is NOT proof of
        UNSAT, so that case raises instead of silently returning None.
        """
        assert model.booster is not None and model.v is not None, "model must be fit"
        assert not (set(flip_set) & set(self.spec.immutable)), (
            f"flip_set {flip_set} includes an immutable feature"
        )

        columns = list(self.validity.bins.edges.keys())
        bounds = {f: (float(self.validity.bins.edges[f][0]),
                      float(self.validity.bins.edges[f][-1])) for f in columns}
        trees_leaves, leaf_paths = self._tree_leaf_paths(model)

        m = gp.Model("sensitivity_oracle")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = self.cfg.mip_time_limit
        m.Params.MIPGap = 0.0 if certifying else self.cfg.mip_gap
        m.Params.DualReductions = 0     # get a clean INFEASIBLE, not INF_OR_UNBD
        m.Params.IntFeasTol = 1e-9      # keep well clear of _SPLIT_EPS's margin
        m.Params.FeasibilityTol = 1e-9

        # --- point variables: shared outside flip_set, so agreement is
        # structural (one Var object), never an equality constraint.
        vars_x1 = {f: m.addVar(lb=bounds[f][0], ub=bounds[f][1], name=f"x1_{f}")
                   for f in columns}
        vars_x2_only = {f: m.addVar(lb=bounds[f][0], ub=bounds[f][1], name=f"x2_{f}")
                        for f in flip_set}
        vars_x2 = {f: (vars_x2_only[f] if f in flip_set else vars_x1[f])
                   for f in columns}

        self._add_validity_constraints(m, vars_x1)
        self._add_validity_constraints(m, vars_x2)

        # --- tree encoding: one leaf indicator per point per leaf, exactly
        # one active per tree, path conditions enforced via indicators so
        # the leaf var can only be 1 when x actually reaches that leaf.
        z1: dict[int, gp.Var] = {}
        z2: dict[int, gp.Var] = {}

        for _tree_id, leaves in trees_leaves.items():
            for g in leaves:
                z1[g] = m.addVar(vtype=GRB.BINARY, name=f"z1_{g}")
                z2[g] = m.addVar(vtype=GRB.BINARY, name=f"z2_{g}")
            m.addConstr(gp.quicksum(z1[g] for g in leaves) == 1)
            m.addConstr(gp.quicksum(z2[g] for g in leaves) == 1)

        le, ge = GRB.LESS_EQUAL, GRB.GREATER_EQUAL
        for g, path in leaf_paths.items():
            for feat, thresh, direction in path:
                if direction == "yes":
                    hi = thresh - _SPLIT_EPS
                    m.addGenConstrIndicator(z1[g], True, vars_x1[feat], le, hi)
                    m.addGenConstrIndicator(z2[g], True, vars_x2[feat], le, hi)
                else:
                    m.addGenConstrIndicator(z1[g], True, vars_x1[feat], ge, thresh)
                    m.addGenConstrIndicator(z2[g], True, vars_x2[feat], ge, thresh)

        # --- Gap-bin: x1, x2 must land confidently on opposite sides of the
        # decision boundary, not just anywhere with a nonzero score gap.
        E1 = gp.quicksum(float(model.v[g]) * z1[g] for g in leaf_paths)
        E2 = gp.quicksum(float(model.v[g]) * z2[g] for g in leaf_paths)
        m.addConstr(self.cfg.gap <= E1)
        m.addConstr(-self.cfg.gap >= E2)

        # --- accumulated no-goods: forbid re-finding a previously reported
        # leaf pattern. l_n = z1[n] + z2[n], per the formula in add_nogood.
        for pattern in self._nogoods:
            l_n = gp.quicksum(z1[int(n)] + z2[int(n)] for n in pattern)
            m.addConstr(l_n <= len(pattern) - 1)

        self._apply_warm_start(
            vars_x1, vars_x2_only, z1, z2, leaf_paths, flip_set, columns)

        m.setObjective(E1 - E2, GRB.MAXIMIZE)
        m.optimize()

        if m.Status == GRB.INFEASIBLE:
            return None

        if m.SolCount == 0:
            if certifying:
                raise RuntimeError(
                    f"oracle stopped (status={m.Status}) with no incumbent and no "
                    "infeasibility proof -- cannot certify UNSAT from this run"
                )
            return None

        x1_val = np.array([vars_x1[f].X for f in columns], dtype=float)
        x2_val = np.array([vars_x2[f].X for f in columns], dtype=float)
        leaves1 = np.array([g for g in leaf_paths if z1[g].X > 0.5], dtype=int)
        leaves2 = np.array([g for g in leaf_paths if z2[g].X > 0.5], dtype=int)

        return Counterexample(
            x1=x1_val, x2=x2_val, gap=float(m.ObjVal), leaves1=leaves1, leaves2=leaves2)

    def find_batch(
        self, model: Ensemble, flip_set: list[str], k: int, certifying: bool = False
    ) -> list[Counterexample]:
        """k distinct counterexamples via repeated no-good cuts."""
        found: list[Counterexample] = []

        for _ in range(k):
            ce = self.find_worst(model, flip_set, certifying=certifying)

            if ce is None:
                break

            found.append(ce)
            self.add_nogood(ce)

        return found

    def add_nogood(self, ce: Counterexample) -> None:
        """sum_{n in pattern} l_n <= |pattern| - 1"""
        self._nogoods.append(np.union1d(ce.leaves1, ce.leaves2))

    def clear_nogoods(self) -> None:
        """Call after leaf values change: old regions may be fine now."""
        self._nogoods = []

    def set_warm_start(self, ce: Counterexample) -> None:
        self._warm_start = ce

    # ------------------------------------------------------------ helpers
    def _tree_leaf_paths(
        self, model: Ensemble
    ) -> tuple[dict[int, list[int]], dict[int, list[tuple[str, float, str]]]]:
        """
        Walk each tree from its root, collecting the split conditions on
        the path to every leaf. Global leaf indices come from
        model.leaf_index -- the SAME dict model.phi() uses, so this can
        never drift out of sync with it (see CLAUDE.md, "Leaf indexing").
        """
        assert model.booster is not None
        df = model.booster.trees_to_dataframe()
        by_tree: dict[int, dict[str, object]] = {}

        for row in df.itertuples():
            by_tree.setdefault(int(row.Tree), {})[str(row.ID)] = row

        trees_leaves: dict[int, list[int]] = {}
        leaf_paths: dict[int, list[tuple[str, float, str]]] = {}

        for tree_id, nodes in by_tree.items():
            trees_leaves[tree_id] = []
            root: str = f"{tree_id}-0"
            stack: list[tuple[str, list[tuple[str, float, str]]]] = [(root, [])]

            while stack:
                node_id, path = stack.pop()
                row = nodes[node_id]

                if row.Feature == "Leaf":                       # type: ignore[attr-defined]
                    leaf_num = int(node_id.split("-")[1])
                    g = model.leaf_index[(tree_id, leaf_num)]
                    trees_leaves[tree_id].append(g)
                    leaf_paths[g] = path
                else:
                    feat = str(row.Feature)                      # type: ignore[attr-defined]
                    thresh = float(row.Split)                    # type: ignore[attr-defined]
                    stack.append((str(row.Yes), [*path, (feat, thresh, "yes")]))  # type: ignore[attr-defined]
                    stack.append((str(row.No), [*path, (feat, thresh, "no")]))    # type: ignore[attr-defined]

        return trees_leaves, leaf_paths

    def _add_validity_constraints(
        self, m: gp.Model, var_dict: dict[str, gp.Var]
    ) -> None:
        """
        Applies self.validity.milp_constraints() to one point's variables.
        Assumed contract (validity.py owns the real implementation): an
        iterable of (coeffs, sense, rhs) triples meaning
        sum_f coeffs[f] * var_dict[f]  <sense>  rhs, sense in {"<=", ">=", "=="}.
        """
        for coeffs, sense, rhs in self.validity.milp_constraints():
            expr = gp.quicksum(coef * var_dict[f] for f, coef in coeffs.items())
            if sense == "<=":
                m.addConstr(expr <= rhs)
            elif sense == ">=":
                m.addConstr(expr >= rhs)
            else:
                m.addConstr(expr == rhs)

    def _apply_warm_start(
        self,
        vars_x1: dict[str, gp.Var],
        vars_x2_only: dict[str, gp.Var],
        z1: dict[int, gp.Var],
        z2: dict[int, gp.Var],
        leaf_paths: dict[int, list[tuple[str, float, str]]],
        flip_set: list[str],
        columns: list[str],
    ) -> None:
        ws = self._warm_start
        if ws is None:
            return

        ws_x1 = dict(zip(columns, ws.x1, strict=True))
        ws_x2 = dict(zip(columns, ws.x2, strict=True))

        for f in columns:
            vars_x1[f].Start = ws_x1[f]
        for f in flip_set:
            if f in vars_x2_only:
                vars_x2_only[f].Start = ws_x2[f]

        ws_leaves1 = set(int(g) for g in ws.leaves1)
        ws_leaves2 = set(int(g) for g in ws.leaves2)
        for g in leaf_paths:
            z1[g].Start = 1.0 if g in ws_leaves1 else 0.0
            z2[g].Start = 1.0 if g in ws_leaves2 else 0.0

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB


class LeafRepairQP:
    """
    Structure is frozen, so E_v(x) = phi(x) . v is LINEAR in v.
    Every sensitivity cut is therefore a linear constraint, and the whole
    problem is a convex QP:

        min  (v - v0)^T (L + mu I) (v - v0)
        s.t. -eps <= (phi(x1) - phi(x2)) . v <= eps   for each cut

    L = Phi^T Phi is built ONCE. The dataset then disappears from the
    problem: the QP size depends only on the number of leaves.
    """

    def __init__(self, model, X_train, cfg) -> None:
        self.model = model
        self.cfg = cfg
        self.v0 = model.v0.copy()
        self.n = model.n_leaves

        Phi = model.phi(X_train)
        self.L: sp.csc_matrix = (Phi.T @ Phi).tocsc()          # sparse, leaves x leaves

        self._build()

    def _build(self) -> None:
        self.m = gp.Model("leaf_repair")
        self.m.Params.OutputFlag = 0
        self.d = self.m.addVars(self.n, lb=-GRB.INFINITY, name="d")  # d = v - v0

        d = np.array([self.d[i] for i in range(self.n)])
        Lc = self.L.tocoo()
        entries = zip(Lc.row, Lc.col, Lc.data, strict=True)
        quad = gp.quicksum(val * d[i] * d[j] for i, j, val in entries)
        quad += self.cfg.mu * gp.quicksum(d[i] * d[i] for i in range(self.n))
        self.m.setObjective(quad, GRB.MINIMIZE)
        self.n_cuts = 0
        self.cuts: list[dict] = []          # log for --dump-cuts; 1:1 with n_cuts

    # --------------------------------------------------------------- cuts
    def add_cut(self, ce) -> None:
        """
        -eps <= sum_n (phi_n(x1) - phi_n(x2)) * v_n <= eps

        Shared leaves get coefficient 0 and drop out automatically.
        Note v = v0 + d, so the constant part shifts the bounds.
        """
        n_cuts_before = self.n_cuts

        coef = np.zeros(self.n)
        np.add.at(coef, ce.leaves1, 1.0)
        np.add.at(coef, ce.leaves2, -1.0)

        nz = np.nonzero(coef)[0]
        expr: gp.LinExpr = gp.quicksum(coef[i] * self.d[i] for i in nz)
        const = float(coef @ self.v0)
        eps = self.cfg.epsilon

        self.m.addConstr(expr + const <= eps,  name=f"cut_hi_{self.n_cuts}")
        self.m.addConstr(expr + const >= -eps, name=f"cut_lo_{self.n_cuts}")

        self.cuts.append({
            "leaves": nz.tolist(),
            "coefficients": coef[nz].tolist(),
            "const": const,
            "epsilon": eps,
            "gap": float(ce.gap),
        })
        self.n_cuts += 1

        # invariant: cuts are never removed (CLAUDE.md #2) -- n_cuts only grows
        assert self.n_cuts == n_cuts_before + 1
        assert self.n_cuts == len(self.cuts)

    # -------------------------------------------------------------- solve
    def solve(self) -> np.ndarray:
        """Warm-started: same model object, one more row, resume from optimum."""
        self.m.optimize()

        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"QP status {self.m.Status}")

        d = np.array([self.d[i].X for i in range(self.n)])
        return self.v0 + d

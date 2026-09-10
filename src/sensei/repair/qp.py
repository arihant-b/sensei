import logging
from typing import cast

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB, nlfunc
from numpy.typing import NDArray

from sensei.repair.cuts import Cut

log = logging.getLogger("sensei.repair.qp")


class RepairQP:
    """
    The convex repair problem: min LogLoss(D_train, v) + mu*||v-v0||^2 +
    kap*sum(s_i), s.t. soft leaf-difference cuts. `solve_qp` (log-loss, PWL
    approximation) is the reported solver; `solve_qp_fast` (squared error) is
    for fast dev iteration; `solve_qp_exact_small` is a small-scale sanity
    check against a true global optimum, never a real-run solver.
    """

    @staticmethod
    def solve_qp_fast(
        v0: NDArray[np.float64],
        gram: sp.coo_matrix,
        cuts: list[Cut],
        mu: float,
        kap: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Squared-error variant, for fast dev iteration: Loss(D_train, v) =
        sum_x (phi(x) @ (v - v0))^2 = d^T (Phi^T Phi) d for d = v - v0.
        `gram` (Phi_train^T @ Phi_train) should be computed ONCE per stage
        and reused across iterations -- it doesn't change within a stage.

        Args:
            v0 (NDArray[np.float64]): Original leaf values, before this repair.
            gram (sp.coo_matrix): `Phi_train^T @ Phi_train`, computed once
                per stage.
            cuts (list[Cut]): Sensitivity cuts to satisfy (softly).
            mu (float): Proximal weight on `||v - v0||^2`.
            kap (float): Slack price on cut violations.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: `(v_new, slack)`
                -- the repaired leaf values and each cut's slack value.

        Raises:
            RuntimeError: Gurobi didn't return GRB.OPTIMAL.
        """

        n: int = len(v0)

        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model("repair_qp_fast", env=env)

        d_vars: gp.tupledict = m.addVars(n, lb=-GRB.INFINITY, name="d")
        d: list[gp.Var] = [d_vars[i] for i in range(n)]

        quad: gp.QuadExpr = RepairQP._squared_error_objective(d, gram, mu)
        slack_vars: list[gp.Var] = RepairQP._add_soft_cut_constraints(m, d, v0, cuts)

        # changing the objective from quad to quad + kap * sum(s_i)
        # adding a penalty to slack otherwise, the solver will always return a solution
        # with slack > 0. default penalty is 100.
        m.setObjective(quad + kap * gp.quicksum(slack_vars), GRB.MINIMIZE)
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"repair QP status {m.Status}")

        return RepairQP._extract_solution(d_vars, v0, slack_vars, n)

    @staticmethod
    def solve_qp(
        v0: NDArray[np.float64],
        X_train_phi: sp.csr_matrix,
        y_train: NDArray[np.int64],
        base_score: float,
        cuts: list[Cut],
        mu: float,
        kap: float,
        n_pwl_breakpoints: int = 41,
        margin_bound: float = 20.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Reported-run solver: log-loss with a piecewise-linear approximation.
        Loss(D_train, v) = sum_x log(1 + exp(-(2y-1) * E_v(x))), a standard
        convex logistic loss in the margin, linearized over `n_pwl_breakpoints`
        points spanning `[-margin_bound, margin_bound]`.

        Args:
            v0 (NDArray[np.float64]): Original leaf values, before this repair.
            X_train_phi (sp.csr_matrix): Training rows' leaf-indicator matrix.
            y_train (NDArray[np.int64]): Training labels, `{0, 1}`.
            base_score (float): The ensemble's constant base score.
            cuts (list[Cut]): Sensitivity cuts to satisfy (softly).
            mu (float): Proximal weight on `||v - v0||^2`.
            kap (float): Slack price on cut violations.
            n_pwl_breakpoints (int): Number of piecewise-linear breakpoints
                per row's loss term.
            margin_bound (float): The PWL approximation's margin range,
                `[-margin_bound, margin_bound]`.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: `(v_new, slack)`
                -- the repaired leaf values and each cut's slack value.

        Raises:
            RuntimeError: Gurobi didn't return GRB.OPTIMAL.
        """

        n = len(v0)

        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model("repair_qp_logloss", env=env)

        d_vars: gp.tupledict = m.addVars(n, lb=-GRB.INFINITY, name="d")
        d: list[gp.Var] = [d_vars[i] for i in range(n)]

        loss_terms: list[gp.Var] = RepairQP._add_logloss_terms(
            m, d, v0, X_train_phi, y_train, base_score, n_pwl_breakpoints, margin_bound
        )
        quad: gp.QuadExpr = mu * gp.quicksum(d[i] * d[i] for i in range(n))
        slack_vars: list[gp.Var] = RepairQP._add_soft_cut_constraints(m, d, v0, cuts)

        m.setObjective(
            gp.quicksum(loss_terms) + quad + kap * gp.quicksum(slack_vars), GRB.MINIMIZE
        )
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"repair QP (log-loss) status {m.Status}")

        return RepairQP._extract_solution(d_vars, v0, slack_vars, n)

    @staticmethod
    def solve_qp_exact_small(
        v0: NDArray[np.float64],
        X_train_phi: sp.csr_matrix,
        y_train: NDArray[np.int64],
        base_score: float,
        cuts: list[Cut],
        mu: float,
        kap: float,
        time_limit_s: float = 60.0,
        max_rows: int = 50,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve the repair problem using an EXACT log-loss objective via Gurobi's
        native nonlinear general expressions (gurobipy.nlfunc), instead of
        `solve_qp`'s piecewise-linear approximation. Each row's
        log(1 + exp(-margin)) term is modeled directly with
        `nlfunc.log`/`nlfunc.exp` -- no breakpoints, no `margin_bound` truncation.

        NOT FOR REPORTED RUNS. Benchmarked on this repo's own adult-dataset leaf
        map (30 trees): 10 rows solve to PROVEN optimality in ~0.3s, but 20 rows
        already fails to prove optimality within a 45s limit (~0.01% gap left),
        and 50 rows leaves a ~1% gap. Each row adds its own general nonlinear
        constraint, and Gurobi's solver for these does not exploit the fact that
        softplus is convex -- cost grows steeply with row count, and this does
        NOT scale to real training set sizes (thousands of rows). Its only
        legitimate use is sanity-checking `solve_qp`'s piecewise-linear
        approximation against a true global optimum on a small subset of
        D_train -- never as the actual repair solver for a real run.

        `max_rows` (default 50) hard-caps `X_train_phi`'s rows -- raise it
        explicitly if you understand the cost. A non-OPTIMAL status after
        `time_limit_s` is logged, not raised: the caller gets the best
        incumbent found, which may not be a proven optimum (see above).

        Args:
            v0 (NDArray[np.float64]): Original leaf values, before this repair.
            X_train_phi (sp.csr_matrix): Training rows' leaf-indicator matrix.
            y_train (NDArray[np.int64]): Training labels, `{0, 1}`.
            base_score (float): The ensemble's constant base score.
            cuts (list[Cut]): Sensitivity cuts to satisfy (softly).
            mu (float): Proximal weight on `||v - v0||^2`.
            kap (float): Slack price on cut violations.
            time_limit_s (float): Gurobi's own `TimeLimit` parameter.
            max_rows (int): Hard cap on `X_train_phi`'s row count.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: `(v_new, slack)`
                -- the repaired leaf values and each cut's slack value.

        Raises:
            ValueError: `X_train_phi` has more than `max_rows` rows.
            RuntimeError: Gurobi found no incumbent at all within the time limit.
        """

        assert X_train_phi.shape is not None

        n_rows: int = X_train_phi.shape[0]

        if n_rows > max_rows:
            raise ValueError(
                f"solve_qp_exact_small got {n_rows} rows > max_rows={max_rows} -- "
                "this solver does not scale past a few dozen rows (see docstring); "
                "raise max_rows explicitly if you understand the cost."
            )

        n: int = len(v0)

        env = gp.Env(params={"OutputFlag": 0, "TimeLimit": time_limit_s})
        m = gp.Model("repair_qp_logloss_exact_small", env=env)

        d_vars: gp.tupledict = m.addVars(n, lb=-GRB.INFINITY, name="d")
        d: list[gp.Var] = [d_vars[i] for i in range(n)]

        loss_terms: list[gp.Var] = RepairQP._add_logloss_terms_nl(
            m, d, v0, X_train_phi, y_train, base_score
        )
        quad: gp.QuadExpr = mu * gp.quicksum(d[i] * d[i] for i in range(n))
        slack_vars: list[gp.Var] = RepairQP._add_soft_cut_constraints(m, d, v0, cuts)

        m.setObjective(
            gp.quicksum(loss_terms) + quad + kap * gp.quicksum(slack_vars), GRB.MINIMIZE
        )
        m.optimize()

        if m.SolCount == 0:
            raise RuntimeError(
                f"repair QP (exact log-loss) status {m.Status}, no incumbent found "
                f"within time_limit_s={time_limit_s}"
            )
        if m.Status != GRB.OPTIMAL:
            log.warning(
                "solve_qp_exact_small did not prove optimality (status=%s) within "
                "time_limit_s=%.1f -- returning best incumbent found, not a proven "
                "optimum",
                m.Status,
                time_limit_s,
            )

        return RepairQP._extract_solution(d_vars, v0, slack_vars, n)

    @staticmethod
    def _squared_error_objective(
        d: list[gp.Var], gram: sp.coo_matrix, mu: float
    ) -> gp.QuadExpr:
        """
        d^T (Phi^T Phi) d -- squared-error data term -- plus the mu proximal term.

        Args:
            d (list[gp.Var]): Decision variables for `v - v0`.
            gram (sp.coo_matrix): `Phi_train^T @ Phi_train`.
            mu (float): Proximal weight on `||v - v0||^2`.

        Returns:
            gp.QuadExpr: The squared-error data term plus the proximal term.
        """

        # d = v - v0
        # sum_{x in D} (E_v(x) - E_v0(x))^2 + mu * sum_n (v - v0)^2
        # E_v(x) = base_score + phi(x)@v
        # E_v(x) - E_v0(x) = phi(x)@v - phi(x)@v0 = phi(x)@(v - v0) = phi(x)@d
        # sum_x (phi(x)@d)^2 = d^T (Phi^T Phi) d (linear algebra identity)
        quad: gp.QuadExpr = cast(
            gp.QuadExpr,
            gp.quicksum(
                d[i] * d[j] * float(val)
                for i, j, val in zip(gram.row, gram.col, gram.data, strict=True)
            ),
        )
        quad += mu * gp.quicksum(d[i] * d[i] for i in range(len(d)))
        return quad

    @staticmethod
    def _add_logloss_terms(
        m: gp.Model,
        d: list[gp.Var],
        v0: NDArray[np.float64],
        X_train_phi: sp.csr_matrix,
        y_train: NDArray[np.int64],
        base_score: float,
        n_pwl_breakpoints: int,
        margin_bound: float,
    ) -> list[gp.Var]:
        """
        One PWL-approximated `log(1 + exp(-margin))` loss var per training row.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            d (list[gp.Var]): Decision variables for `v - v0`.
            v0 (NDArray[np.float64]): Original leaf values, before this repair.
            X_train_phi (sp.csr_matrix): Training rows' leaf-indicator matrix.
            y_train (NDArray[np.int64]): Training labels, `{0, 1}`.
            base_score (float): The ensemble's constant base score.
            n_pwl_breakpoints (int): Number of piecewise-linear breakpoints
                per row's loss term.
            margin_bound (float): The PWL approximation's margin range,
                `[-margin_bound, margin_bound]`.

        Returns:
            list[gp.Var]: One loss variable per training row.
        """

        assert X_train_phi.shape is not None
        n_rows: int = X_train_phi.shape[0]

        z_pts: NDArray[np.float64] = np.linspace(
            -margin_bound, margin_bound, n_pwl_breakpoints
        )
        loss_pts: NDArray[np.float64] = np.log1p(np.exp(-z_pts))

        signed: NDArray[np.float64] = np.where(y_train > 0, 1.0, -1.0)
        phi_csr: sp.csr_matrix = X_train_phi.tocsr()

        loss_terms: list[gp.Var] = []

        for r in range(n_rows):
            row: sp.csr_matrix = phi_csr.getrow(r)
            margin_expr: gp.LinExpr = signed[r] * float(base_score)
            margin_expr += signed[r] * float((row @ v0).item())
            margin_expr += signed[r] * gp.quicksum(
                float(val) * d[int(idx)]
                for idx, val in zip(row.indices, row.data, strict=True)
            )

            z_var: gp.Var = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"z_{r}")
            m.addConstr(z_var == margin_expr)

            loss_var: gp.Var = m.addVar(lb=0.0, name=f"loss_{r}")
            m.addGenConstrPWL(z_var, loss_var, list(z_pts), list(loss_pts))
            loss_terms.append(loss_var)

        return loss_terms

    @staticmethod
    def _add_logloss_terms_nl(
        m: gp.Model,
        d: list[gp.Var],
        v0: NDArray[np.float64],
        X_train_phi: sp.csr_matrix,
        y_train: NDArray[np.int64],
        base_score: float,
    ) -> list[gp.Var]:
        """
        Same as `_add_logloss_terms` but EXACT: `gurobipy.nlfunc.log`/`.exp`
        model `log(1 + exp(-margin))` directly, no PWL breakpoints. See
        `solve_qp_exact_small`'s docstring for why this does not scale past
        a few dozen rows.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            d (list[gp.Var]): Decision variables for `v - v0`.
            v0 (NDArray[np.float64]): Original leaf values, before this repair.
            X_train_phi (sp.csr_matrix): Training rows' leaf-indicator matrix.
            y_train (NDArray[np.int64]): Training labels, `{0, 1}`.
            base_score (float): The ensemble's constant base score.

        Returns:
            list[gp.Var]: One loss variable per training row.
        """

        assert X_train_phi.shape is not None
        n_rows: int = X_train_phi.shape[0]

        signed: NDArray[np.float64] = np.where(y_train > 0, 1.0, -1.0)
        phi_csr: sp.csr_matrix = X_train_phi.tocsr()

        loss_terms: list[gp.Var] = []

        for r in range(n_rows):
            row: sp.csr_matrix = phi_csr.getrow(r)
            margin_expr: gp.LinExpr = signed[r] * float(base_score)
            margin_expr += signed[r] * float((row @ v0).item())
            margin_expr += signed[r] * gp.quicksum(
                float(val) * d[int(idx)]
                for idx, val in zip(row.indices, row.data, strict=True)
            )

            loss_var: gp.Var = m.addVar(lb=0.0, name=f"loss_{r}")
            m.addConstr(loss_var == nlfunc.log(1 + nlfunc.exp(-margin_expr)))
            loss_terms.append(loss_var)

        return loss_terms

    @staticmethod
    def _add_soft_cut_constraints(
        m: gp.Model, d: list[gp.Var], v0: NDArray[np.float64], cuts: list[Cut]
    ) -> list[gp.Var]:
        """
        Per cut: `-eps - s <= d_i @ v <= eps + s`, `s >= 0`. Cuts are always
        soft -- around 30-50 cuts in, some leaf gets pulled two ways
        and a hard-constrained problem returns INFEASIBLE; slack turns that
        into a measurement (`sum(s) > 0`) instead of a crash. `eps` is
        global; slack is per cut.

        Args:
            m (gp.Model): The Gurobi model to add variables/constraints to.
            d (list[gp.Var]): Decision variables for `v - v0`.
            v0 (NDArray[np.float64]): Original leaf values, before this repair.
            cuts (list[Cut]): Sensitivity cuts to satisfy (softly).

        Returns:
            list[gp.Var]: One slack variable per cut, in `cuts`' order.
        """

        slack_vars: list[gp.Var] = []

        for ci, cut in enumerate(cuts):
            s: gp.Var = m.addVar(lb=0.0, name=f"s_{ci}")
            slack_vars.append(s)

            expr: gp.LinExpr = gp.quicksum(
                float(c) * d[int(i)] for i, c in zip(cut.idx, cut.coef, strict=True)
            )
            const = float(
                sum(c * v0[int(i)] for i, c in zip(cut.idx, cut.coef, strict=True))
            )

            m.addConstr(expr + const <= cut.eps + s, name=f"cut_hi_{ci}")
            m.addConstr(expr + const >= -cut.eps - s, name=f"cut_lo_{ci}")

        return slack_vars

    @staticmethod
    def _extract_solution(
        d_vars: dict[int, gp.Var],
        v0: NDArray[np.float64],
        slack_vars: list[gp.Var],
        n: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        `v0 + d` and the per-cut slack values, read off the solved model.

        Args:
            d_vars (dict[int, gp.Var]): Solved decision variables for `v - v0`.
            v0 (NDArray[np.float64]): Original leaf values, before this repair.
            slack_vars (list[gp.Var]): Solved per-cut slack variables.
            n (int): Number of leaf values (length of `v0`/`d_vars`).

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: `(v_new, slack)`.
        """

        v_new = v0 + np.array([d_vars[i].X for i in range(n)])
        slack: NDArray[np.float64] = np.array([s.X for s in slack_vars], dtype=float)
        return v_new, slack

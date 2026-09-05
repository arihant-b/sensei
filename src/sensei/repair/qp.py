import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB
from numpy.typing import NDArray

from sensei.repair.cuts import Cut


class RepairQP:
    """
    Quadratic program solver for the repair problem. Provides methods to solve the
    repair problem using either a squared-error loss or a log-loss with a
    piecewise-linear approximation. The repair problem aims to find a new model
    parameter vector `v` that minimizes the loss on the training data while satisfying
    the constraints imposed by the cuts, with a proximal term to keep `v` close to
    the original parameter vector `v0`.

    Raises:
        RuntimeError: If the Gurobi solver does not return an optimal solution for the
                      repair problem.
        RuntimeError: If the Gurobi solver does not return an optimal solution for the
                      log-loss variant of the repair problem.
    """

    @staticmethod
    def solve_qp_fast(
        v0: NDArray[np.float64],
        phi_train: sp.csr_matrix,
        cuts: list[Cut],
        mu: float,
        kap: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve the repair problem using a squared-error loss. This method is a faster
        variant of the repair problem solver, suitable for iterating on the loop itself.
        Squared-error variant: Loss(D_train, v) = sum_x (phi(x) @ (v - v0))^2 =
        d^T (Phi^T Phi) d for d = v - v0.

        Args:
            v0 (NDArray[np.float64]): The original model parameter vector before repair.
            phi_train (sp.csr_matrix): The feature matrix for the training data, where
                                       each row corresponds to a training example and
                                       each column corresponds to a feature.
            cuts (list[Cut]): A list of sparse leaf-difference cuts that represent the
                              constraints to be satisfied in the repair problem.
            mu (float): The weight of the proximal term that penalizes deviation from
                        the original parameter vector `v0`.
            kap (float): The weight of the slack variables that allow for soft
                         constraints in the repair problem.

        Raises:
            RuntimeError: If the Gurobi solver does not return an optimal solution for
                          the repair problem.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: A tuple containing the
                                                             repaired model parameter
                                                             vector `v` and the slack
                                                             variables `slack`.
        """

        n: int = len(v0)

        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model("repair_qp_fast", env=env)

        d_vars: gp.tupledict = m.addVars(n, lb=-GRB.INFINITY, name="d")
        d: NDArray[np.float64] = np.array([d_vars[i] for i in range(n)])

        quad: gp.QuadExpr = RepairQP._squared_error_objective(d, phi_train, mu)
        slack_vars: list[gp.Var] = RepairQP._add_soft_cut_constraints(m, d, v0, cuts)

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
        Solve the repair problem using a log-loss with a piecewise-linear approximation.
        Loss(D_train, v) = sum_x log(1 + exp(-(2y-1) * E_v(x))), a standard convex
        logistic loss in the margin.

        Args:
            v0 (NDArray[np.float64]): The original model parameter vector before repair.
            X_train_phi (sp.csr_matrix): The training data in feature space.
            y_train (NDArray[np.int64]): The training labels.
            base_score (float): The base score of the model.
            cuts (list[Cut]): The cuts to be applied.
            mu (float): The regularization parameter.
            kap (float): The penalty parameter.
            n_pwl_breakpoints (int, optional): The number of breakpoints for the
                                               piecewise-linear approximation. Defaults
                                               to 41.
            margin_bound (float, optional): The bound on the margin. Defaults to 20.0.

        Raises:
            RuntimeError: If the Gurobi solver does not return an optimal solution for
                          the repair problem.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: A tuple containing the
                                                             repaired model parameter
                                                             vector `v` and the slack
                                                             variables `slack`.
        """

        n = len(v0)

        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model("repair_qp_logloss", env=env)

        d_vars: gp.tupledict = m.addVars(n, lb=-GRB.INFINITY, name="d")
        d: NDArray[np.float64] = np.array([d_vars[i] for i in range(n)])

        loss_terms: list[gp.Var] = RepairQP._add_logloss_terms(
            m, d, v0, X_train_phi, y_train, base_score, n_pwl_breakpoints, margin_bound
        )
        quad: gp.LinExpr = mu * gp.quicksum(d[i] * d[i] for i in range(n))
        slack_vars: list[gp.Var] = RepairQP._add_soft_cut_constraints(m, d, v0, cuts)

        m.setObjective(
            gp.quicksum(loss_terms) + quad + kap * gp.quicksum(slack_vars), GRB.MINIMIZE
        )
        m.optimize()

        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"repair QP (log-loss) status {m.Status}")

        return RepairQP._extract_solution(d_vars, v0, slack_vars, n)

    @staticmethod
    def _squared_error_objective(
        d: NDArray, phi_train: sp.csr_matrix, mu: float
    ) -> gp.QuadExpr:
        """
        Compute the squared-error objective for the repair problem.
        d^T (Phi^T Phi) d -- the squared-error data term -- plus the mu proximal term.

        Args:
            d (NDArray): The decision variable representing the change in model
                         parameters.
            phi_train (sp.csr_matrix): The feature matrix for the training data, where
                         each row corresponds to a training example and each column
                         corresponds to a feature.
            mu (float): The weight of the proximal term that penalizes deviation from
                        the original parameter vector `v0`.

        Returns:
            gp.QuadExpr: The quadratic expression representing the squared-error
                         objective for the repair problem.
        """

        L: sp.coo_matrix = (phi_train.T @ phi_train).tocoo()
        quad: gp.QuadExpr = gp.quicksum(
            float(val) * d[i] * d[j]
            for i, j, val in zip(L.row, L.col, L.data, strict=True)
        )
        quad += mu * gp.quicksum(d[i] * d[i] for i in range(len(d)))
        return quad

    @staticmethod
    def _add_logloss_terms(
        m: gp.Model,
        d: NDArray,
        v0: NDArray[np.float64],
        X_train_phi: sp.csr_matrix,
        y_train: NDArray[np.int64],
        base_score: float,
        n_pwl_breakpoints: int,
        margin_bound: float,
    ) -> list[gp.Var]:
        """
        Add log-loss terms to the Gurobi model using a piecewise-linear approximation.

        Args:
            m (gp.Model): The Gurobi model to which the log-loss terms will be added.
            d (NDArray): The decision variable representing the change in model
                         parameters.
            v0 (NDArray[np.float64]): The original model parameter vector before repair.
            X_train_phi (sp.csr_matrix): The training data in feature space.
            y_train (NDArray[np.int64]): The training labels.
            base_score (float): The base score of the model.
            n_pwl_breakpoints (int): The number of breakpoints for the piecewise-linear
                                     approximation.
            margin_bound (float): The bound on the margin.

        Returns:
            list[gp.Var]: A list of Gurobi variables representing the log-loss terms for
                          each training example.
        """

        assert X_train_phi.shape is not None
        n_rows: int = X_train_phi.shape[0]

        z_pts: NDArray[np.float64] = np.linspace(
            -margin_bound, margin_bound, n_pwl_breakpoints
        )

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
            margin_expr += signed[r] * float(row @ v0)
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
    def _add_soft_cut_constraints(
        m: gp.Model, d: NDArray, v0: NDArray[np.float64], cuts: list[Cut]
    ) -> list[gp.Var]:
        """
        Add soft cut constraints to the Gurobi model. Each cut is represented as a
        pair of linear inequalities with a slack variable that allows for soft
        constraints.

        Args:
            m (gp.Model): The Gurobi model to which the soft cut constraints will be
                          added.
            d (NDArray): The decision variable representing the change in model
                         parameters.
            v0 (NDArray[np.float64]): The original model parameter vector before repair.
            cuts (list[Cut]): A list of sparse leaf-difference cuts that represent the
                              constraints to be satisfied in the repair problem.

        Returns:
            list[gp.Var]: A list of Gurobi variables representing the slack variables
                          for each cut, allowing for soft constraints in the repair
                          problem.
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
        Extract the solution from the Gurobi model after optimization.

        Args:
            d_vars (dict[int, gp.Var]): A dictionary mapping indices to Gurobi variables
                                        representing the change in model parameters.
            v0 (NDArray[np.float64]): The original model parameter vector before repair.
            slack_vars (list[gp.Var]): A list of Gurobi variables representing the slack
                                       variables for each cut, allowing for soft
                                       constraints in the repair problem.
            n (int): The number of model parameters.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]: A tuple containing the
                                                             repaired model parameter
                                                             vector `v` and the slack
                                                             variables `slack`.
        """

        v_new = v0 + np.array([d_vars[i].X for i in range(n)])
        slack: NDArray[np.float64] = np.array([s.X for s in slack_vars], dtype=float)
        return v_new, slack

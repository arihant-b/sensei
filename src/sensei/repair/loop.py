import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from scipy.sparse._csr import csr_matrix

from sensei.data.bins import FrozenBins
from sensei.eval.metrics import Metrics
from sensei.model.leaves import LeafMap
from sensei.oracle.milp.encoding import TreeEncoder, TreeStructure
from sensei.oracle.milp.nogoods import NoGood, NoGoodBuilder
from sensei.oracle.milp.solve import TierAOracle
from sensei.oracle.types import OracleSaturated, OracleTimeout, Pair
from sensei.repair.cuts import Cut, CutBuilder
from sensei.repair.pareto import ParetoCheckpoint, Snapshot
from sensei.repair.qp import RepairQP
from sensei.spec import Spec

log = logging.getLogger("sensei.repair.loop")


@dataclass(frozen=True)
class ConditionalCertificate:
    """
    A certificate that no valid pair of points exists under the given constraints,
    including the specified epsilon, theta, flip set, direction, and other parameters.
    """

    tier: str
    eps: float
    theta: float
    flip_set: tuple[str, ...]
    direction: str
    spec_hash: str
    oracle: str
    solver_status: str  # must be "UNSAT", never SATURATED/TIMEOUT
    time_limit_s: float
    n_cuts: int
    snapshot: Snapshot

    def statement(self) -> str:
        """
        Generate a human-readable statement describing the conditional certificate.

        Returns:
            str: A string summarizing the conditions under which no valid pair of points
                 exists, including the oracle used and the time limit.
        """

        return (
            f"No pair (x1, x2) was found that satisfies the encoded type and domain "
            f"rules, scores plaus >= {self.theta} under a product-of-frozen-marginals "
            f"plausibility model, differs only on {self.flip_set}, and produces a "
            f"margin gap exceeding {self.eps} on the repaired model -- under oracle "
            f"{self.oracle} within a {self.time_limit_s}s limit."
        )


@dataclass(frozen=True)
class ParetoBest:
    """
    A snapshot of the best state recorded during the repair process, capturing key
    metrics and information about the iteration at which it was recorded.
    """

    snapshot: Snapshot | None
    reason: str  # "accuracy_floor" | "stalled" | "iteration_cap"


@dataclass(frozen=True)
class Inconclusive:
    """
    An inconclusive result indicating that the repair process could not determine
    whether a valid pair of points exists, due to either a timeout or the oracle
    being saturated.
    """

    snapshot: Snapshot | None
    reason: str  # "TIMEOUT" | "ORACLE_SATURATED"
    iteration: int


class CegsalLoop:
    """
    The main loop of the CEGSAL repair process, which iteratively searches for
    counterexample pairs, constructs cuts, and repairs the model until a stopping
    condition is met. The loop can exit with a conditional certificate, the best
    recorded snapshot, or an inconclusive result.
    """

    def __init__(
        self,
        booster: xgb.Booster,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec,
        seed: int,
    ) -> None:
        self.booster: xgb.Booster = booster
        self.X_train: pd.DataFrame = X_train
        self.y_train: pd.Series = y_train
        self.X_test: pd.DataFrame = X_test
        self.y_test: pd.Series = y_test
        self.spec: Spec = spec
        self.columns: list[str] = columns
        self.feature_bounds: dict[str, tuple[float, float]] = feature_bounds
        self.seed: int = seed

    def run(
        self,
        flip_set: tuple[str, ...],
        direction: str,
        eps: float,
        theta: float,
        mu: float,
        kap: float,
        max_iters: int,
        a_min: float,
        stall_delta: float,
        cuts_per_round: int,
        oracle_time_limit_s: float,
        oracle_mip_gap: float,
        n_quantile_bins: int = 10,
    ) -> ConditionalCertificate | ParetoBest | Inconclusive:
        """
        Run the CEGSAL repair loop, iteratively searching for counterexample pairs,
        constructing cuts, and repairing the model until a stopping condition is met.

        Args:
            flip_set (tuple[str, ...]): The set of features that are allowed to differ
                                        between the counterexample pairs.
            direction (str): The direction of the margin gap to be considered, either
                             "positive" or "negative".
            eps (float): The minimum margin gap required for a counterexample pair to be
                         considered valid.
            theta (float): The minimum plausibility score required for a counterexample
                           pair to be considered valid.
            mu (float): The regularization parameter for the repair quadratic program.
            kap (float): The regularization parameter for the repair quadratic program.
            max_iters (int): The maximum number of iterations to run the repair loop.
            a_min (float): The minimum value for the repair parameter.
            stall_delta (float): The minimum change in the objective function to
                                 consider the optimization not stalled.
            cuts_per_round (int): The number of cuts to generate per iteration.
            oracle_time_limit_s (float): The time limit for the oracle in seconds.
            oracle_mip_gap (float): The MIP gap for the oracle.
            n_quantile_bins (int, optional): The number of quantile bins to use.
                                             Defaults to 10.

        Returns:
            ConditionalCertificate | ParetoBest | Inconclusive: _description_
        """

        self.leaf_map = LeafMap(self.booster)
        structure: TreeStructure = TreeEncoder.extract_tree_structure(
            self.booster, self.leaf_map.leaf_index, self.columns
        )
        n_trees: int = len(structure.trees_leaves)
        bins: FrozenBins = FrozenBins(n_quantile_bins).fit(self.X_train, self.columns)

        self.v: NDArray[np.float64] = self.leaf_map.v0.copy()
        self.cuts: list[Cut] = []
        nogoods: list[NoGood] = []
        warm_start: Pair | None = None
        previous_worst = float("inf")

        oracle = TierAOracle()
        phi_train: csr_matrix = self.leaf_map.phi_for(self.booster, self.X_train)

        pareto = ParetoCheckpoint(a_min)
        pareto.record(self._snapshot(-1, worst_gap=float("nan"), slack_mass=0.0))

        for t in range(max_iters):
            try:
                found: list[Pair] = []

                for _ in range(cuts_per_round):
                    pair: Pair | None = oracle.worst_valid_pair(
                        self.booster,
                        self.leaf_map,
                        self.columns,
                        self.feature_bounds,
                        self.spec,
                        flip_set,
                        direction,
                        mode="feasibility",
                        eps=eps,
                        seed=self.seed,
                        time_limit_s=oracle_time_limit_s,
                        mip_gap=oracle_mip_gap,
                        nogoods=nogoods,
                        warm_start=warm_start,
                        enforce_validity=True,
                        structure=structure,
                        v=self.v,
                        bins=bins,
                        theta=theta,
                    )

                    if pair is None:
                        break

                    found.append(pair)
                    nogoods.append(NoGoodBuilder.make_nogood(pair, n_trees))
                    warm_start = pair

            except OracleTimeout:
                return Inconclusive(pareto.restore(), reason="TIMEOUT", iteration=t)
            except OracleSaturated:
                return Inconclusive(
                    pareto.restore(), reason="ORACLE_SATURATED", iteration=t
                )

            if not found:
                return ConditionalCertificate(
                    tier="A",
                    eps=eps,
                    theta=theta,
                    flip_set=flip_set,
                    direction=direction,
                    spec_hash=self.spec.spec_hash,
                    oracle=f"sensei-milp@{n_trees}trees",
                    solver_status="UNSAT",
                    time_limit_s=oracle_time_limit_s,
                    n_cuts=len(self.cuts),
                    snapshot=self._snapshot(t, worst_gap=0.0, slack_mass=0.0),
                )

            worst: float = max(abs(p.gap) for p in found)

            for pair in found:
                self.cuts.append(CutBuilder.make_cut(pair, eps))

            self.v, slack = RepairQP.solve_qp_fast(
                self.leaf_map.v0, phi_train, self.cuts, mu, kap
            )

            snapshot: Snapshot = self._snapshot(
                t, worst_gap=worst, slack_mass=float(slack.sum())
            )
            pareto.record(snapshot)
            log.info(
                "iter %2d | gap %.4f | acc %.4f | sens %.4f | slack %.4f | cuts %d",
                t,
                worst,
                snapshot.accuracy,
                snapshot.sensitivity_rate,
                snapshot.slack_mass,
                snapshot.n_cuts,
            )

            if snapshot.accuracy < a_min:
                return ParetoBest(pareto.restore(), reason="accuracy_floor")

            if abs(previous_worst - worst) < stall_delta:
                return ParetoBest(pareto.restore(), reason="stalled")

            previous_worst: float = worst

        return ParetoBest(pareto.restore(), reason="iteration_cap")

    def _snapshot(
        self, iteration: int, worst_gap: float, slack_mass: float
    ) -> Snapshot:
        """
        Create a snapshot of the current state of the repair process, capturing key
        metrics and information about the current iteration.

        Args:
            iteration (int): The current iteration number of the repair loop.
            worst_gap (float): The worst gap in the current iteration.
            slack_mass (float): The total slack mass in the current iteration.

        Returns:
            Snapshot: A snapshot of the current state of the repair process, including
                      metrics and information about the current iteration.
        """

        return Snapshot(
            iteration=iteration,
            v=self.v.copy(),
            accuracy=Metrics.accuracy(
                self.booster, self.leaf_map, self.X_test, self.y_test, self.v
            ),
            sensitivity_rate=Metrics.sensitivity_rate(
                self.booster,
                self.leaf_map,
                self.X_test,
                self.spec.protected,
                self.v,
                seed=self.seed,
            ),
            worst_gap=worst_gap,
            slack_mass=slack_mass,
            n_cuts=len(self.cuts),
            cuts=list(self.cuts),
        )

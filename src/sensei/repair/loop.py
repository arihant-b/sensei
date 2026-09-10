import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse._csr import csr_matrix

from sensei.config import Settings, ensense_pin
from sensei.data.bins import FrozenBins
from sensei.eval.metrics import Metrics
from sensei.model.leaves import LeafMap
from sensei.oracle.ensense.adapter import EnsenseOracle
from sensei.oracle.sensei.encoding import TreeEncoder
from sensei.oracle.sensei.nogoods import NoGoodBuilder
from sensei.oracle.sensei.solve import SenseiOracle
from sensei.oracle.types import (
    NoGood,
    OracleSaturated,
    OracleTimeout,
    Pair,
    TreeStructure,
)
from sensei.repair.cuts import Cut, CutBuilder
from sensei.repair.pareto import ParetoCheckpoint, Snapshot
from sensei.repair.qp import RepairQP
from sensei.spec import Spec

log = logging.getLogger("sensei.repair.loop")


@dataclass(frozen=True)
class ConditionalCertificate:
    """Certificate record: UNSAT within the encoded domain, not "proved fair"."""

    oracle_type: str  # "sensei" | "ensense" -- which oracle produced this certificate
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
        The required certificate rendering -- never "the model is fair".

        Returns:
            str: The certificate's plain-language statement.
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
    """Exit B: the best non-dominated snapshot, and why the loop stopped."""

    snapshot: Snapshot | None
    reason: str  # "accuracy_floor" | "stalled" | "iteration_cap"


@dataclass(frozen=True)
class Inconclusive:
    """
    An inconclusive result indicating that the repair process could not determine
    whether a valid pair of points exists -- a timeout, the oracle saturating, the
    Q1+Q2-encoded domain turning out to be empty, or a feasibility/optimality search
    disagreeing with itself (a solver inconsistency, not something to certify).
    Never a certificate under any of these reasons.
    """

    snapshot: Snapshot | None
    # "TIMEOUT" | "ORACLE_SATURATED" | "EMPTY_DOMAIN" | "INCONSISTENT_OPTIMALITY"
    reason: str
    iteration: int


class CegsalLoop:
    """The CEGSAL loop: search -> cut -> repair, per flip_set/direction, to one
    of the three exits (the frozen-structure invariant holds throughout)."""

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
        settings: Settings,
        categorical_levels: dict[str, int] | None = None,
    ) -> None:
        self.booster: xgb.Booster = booster
        self.X_train: pd.DataFrame = X_train
        self.y_train: pd.Series = y_train
        self.X_test: pd.DataFrame = X_test
        self.y_test: pd.Series = y_test
        self.spec: Spec = spec
        self.columns: list[str] = columns
        self.feature_bounds: dict[str, tuple[float, float]] = feature_bounds
        self.settings: Settings = settings
        self.seed: int = settings.seed
        self.categorical_levels: dict[str, int] | None = categorical_levels

    def run(
        self, flip_set: tuple[str, ...], direction: str
    ) -> ConditionalCertificate | ParetoBest | Inconclusive:
        """
        Run one stage of the CEGSAL loop for `flip_set`/`direction` to one of
        the three exits, reading every hyperparameter off `self.settings`
        (fixed at construction -- see `__init__`).

        `settings.repair.type="approx"` selects `RepairQP.solve_qp_fast`
        (squared-error, dev speed) over the reported-run log-loss solver --
        a deliberate opt-in, never the default for real numbers.

        `settings.oracle.type` picks which oracle drives THIS loop's own
        search: "sensei" (own MILP, preferred) or "ensense" (Ensense core +
        postfilter, weaker). Independent of the Sensei self-check / held-out
        verification in `pipeline.py`, which always run Ensense core
        regardless of this.

        Args:
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.

        Returns:
            ConditionalCertificate | ParetoBest | Inconclusive: One of the
                three CEGSAL exits.

        Raises:
            ValueError: `oracle.type == "ensense"` with
                `direction == "monotone_wrong"` (unsupported combination).
        """

        oracle_type: str = self.settings.oracle.type

        if oracle_type == "ensense" and direction == "monotone_wrong":
            raise ValueError(
                "oracle_type='ensense' does not support direction='monotone_wrong' -- "
                "Ensense core's search has no way to pin which side of the pair is "
                "the lower/higher monotone side, and always returns "
                "direction='protected' pairs. Use oracle_type='sensei' for a monotone "
                "feature instead."
            )

        self._init_shared_state(self.settings.bins.n_quantile_bins)
        oracle: SenseiOracle | EnsenseOracle = (
            SenseiOracle() if oracle_type == "sensei" else EnsenseOracle()
        )
        phi_train: csr_matrix = self.leaf_map.phi_for(self.booster, self.X_train)
        # phi_train is frozen for the whole stage (the booster's tree structure
        # never changes -- see LeafMap's docstring), so the Gram matrix is
        # computed once here rather than inside solve_qp_fast on every one of
        # `max_iters` CEGSAL iterations.
        gram_train: coo_matrix = (phi_train.T @ phi_train).tocoo()
        pareto = ParetoCheckpoint(self.settings.loop.A_min)
        self.flip_set = flip_set
        self.direction = direction
        label: str = f"{direction}:{','.join(flip_set)}"
        self._record(pareto, self._snapshot(0, slack_mass=0.0), label)

        return self._run_stage(
            flip_set, direction, oracle, gram_train, pareto, label, phi_train
        )

    def _init_shared_state(self, n_quantile_bins: int) -> None:
        """
        Set up everything that's shared across the whole model (tree
        structure, plausibility bins) and starts fresh exactly once per
        `run` call -- `self.v`/`self.cuts` in particular.

        Args:
            n_quantile_bins (int): Quantile bins per numeric feature, for
                `FrozenBins.fit_or_load`.
        """

        self.leaf_map = LeafMap(self.booster)
        self.structure: TreeStructure = TreeEncoder.extract_tree_structure(
            self.booster, self.leaf_map.leaf_index, self.columns
        )
        self.bins: FrozenBins = FrozenBins.fit_or_load(
            self.spec.dataset,
            self.seed,
            n_quantile_bins,
            self.X_train,
            self.columns,
            self.categorical_levels,
        )
        self.v: NDArray[np.float64] = self.leaf_map.v0.copy()
        self.cuts: list[Cut] = []
        self._last_slack_mass: float = 0.0
        self.history: list[Snapshot] = []
        self.stage_history: dict[str, list[Snapshot]] = {}

    def _record(self, pareto: ParetoCheckpoint, snapshot: Snapshot, label: str) -> None:
        """
        Record one snapshot everywhere it needs to be recorded: the stage's
        own `ParetoCheckpoint` (used for THIS stage's accuracy-floor/stall/
        certificate decisions, unchanged behavior), plus `self.history` and
        `self.stage_history[label]` (reporting/plotting only, never read by
        any loop decision).

        Args:
            pareto (ParetoCheckpoint): This stage's Pareto tracker.
            snapshot (Snapshot): The iteration's state to record.
            label (str): This stage's label in `self.stage_history`.
        """

        pareto.record(snapshot)
        self.history.append(snapshot)
        self.stage_history.setdefault(label, []).append(snapshot)

    def _run_stage(
        self,
        flip_set: tuple[str, ...],
        direction: str,
        oracle: SenseiOracle | EnsenseOracle,
        gram_train: coo_matrix,
        pareto: ParetoCheckpoint,
        label: str,
        phi_train: csr_matrix,
    ) -> ConditionalCertificate | ParetoBest | Inconclusive:
        """
        Search/cut/repair for ONE flip_set/direction until ITS OWN local
        convergence: UNSAT, `max_iters` exhausted, a stall, the accuracy
        floor, a timeout, or the oracle saturating. Does NOT touch
        `self.v`/`self.cuts` at the start (the caller, `run`, owns
        initializing those exactly once); no-goods and this stage's own
        stall-tracking are local here and start fresh every call.

        Every hyperparameter comes off `self.settings` directly.

        `label` identifies this stage for `self.stage_history` (reporting/
        plotting only) -- `run` synthesizes one from flip_set/direction.

        `oracle` must match `settings.oracle.type` ("sensei" for
        `SenseiOracle`, "ensense" for `EnsenseOracle`) -- `run` already
        constructed it to match; `settings.oracle.type` also decides which
        UNSAT path below applies and which
        `ConditionalCertificate.oracle_type`/`.oracle` to record.

        Args:
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            oracle (SenseiOracle | EnsenseOracle): The oracle driving this
                stage's search, matching `settings.oracle.type`.
            gram_train (coo_matrix): `Phi_train^T @ Phi_train`, for
                `RepairQP.solve_qp_fast`.
            pareto (ParetoCheckpoint): This stage's Pareto tracker.
            label (str): This stage's label in `self.stage_history`.
            phi_train (csr_matrix): Training rows' leaf-indicator matrix,
                for `RepairQP.solve_qp_exact_small`.

        Returns:
            ConditionalCertificate | ParetoBest | Inconclusive: This
                stage's exit.
        """

        settings: Settings = self.settings
        eps: float = settings.sensitivity.eps
        theta: float = settings.sensitivity.theta
        oracle_type: str = settings.oracle.type
        repair_type: str = settings.repair.type
        n_trees: int = len(self.structure.trees_leaves)

        nogoods: list[NoGood] = []
        previous_worst = float("inf")

        if oracle_type == "sensei":
            assert isinstance(oracle, SenseiOracle)
            search: Callable[..., Pair | None] = oracle.worst_valid_pair
        else:
            assert isinstance(oracle, EnsenseOracle)
            search = oracle.worst_valid_pair_loop

        for t in range(settings.loop.max_iters):
            iter_num: int = t + 1

            try:
                found: list[Pair] = []
                rounds: int = (
                    1 if oracle_type == "ensense" else settings.loop.cuts_per_round
                )

                for _ in range(rounds):
                    pair: Pair | None = search(
                        self.booster,
                        self.leaf_map,
                        self.columns,
                        self.feature_bounds,
                        self.spec,
                        flip_set,
                        direction,
                        mode="feasibility",
                        settings=settings,
                        nogoods=nogoods,
                        warm_start=None,
                        enforce_validity=True,
                        structure=self.structure,
                        v=self.v,
                        bins=self.bins,
                    )

                    if pair is None:
                        break

                    found.append(pair)
                    nogoods.append(NoGoodBuilder.make_nogood(pair, n_trees))

            except OracleTimeout:
                return Inconclusive(
                    pareto.restore(), reason="TIMEOUT", iteration=iter_num
                )
            except OracleSaturated:
                return Inconclusive(
                    pareto.restore(), reason="ORACLE_SATURATED", iteration=iter_num
                )

            if not found and oracle_type == "ensense":
                log.info(
                    "ensense oracle: search+postfilter found nothing for "
                    "flip_set=%s, direction=%s over %d round(s) -- UNSAT (absence "
                    "within the post-filtered representable space, not an "
                    "exhaustive search)",
                    flip_set,
                    direction,
                    iter_num,
                )
                certificate_snapshot: Snapshot = self._snapshot(
                    iter_num, slack_mass=self._last_slack_mass
                )
                self._record(pareto, certificate_snapshot, label)

                return ConditionalCertificate(
                    oracle_type="ensense",
                    eps=eps,
                    theta=theta,
                    flip_set=flip_set,
                    direction=direction,
                    spec_hash=self.spec.spec_hash,
                    oracle=f"ensense-core@{ensense_pin()}",
                    solver_status="UNSAT",
                    time_limit_s=settings.oracle.time_limit_s,
                    n_cuts=len(self.cuts),
                    snapshot=certificate_snapshot,
                )

            if not found:
                assert isinstance(oracle, SenseiOracle)
                try:
                    confirmation: Pair | None = oracle.worst_valid_pair(
                        self.booster,
                        self.leaf_map,
                        self.columns,
                        self.feature_bounds,
                        self.spec,
                        flip_set,
                        direction,
                        mode="optimality",
                        settings=settings,
                        enforce_validity=True,
                        structure=self.structure,
                        v=self.v,
                        bins=self.bins,
                    )
                except OracleTimeout:
                    return Inconclusive(
                        pareto.restore(), reason="TIMEOUT", iteration=iter_num
                    )

                if confirmation is None:
                    log.warning(
                        "EMPTY_DOMAIN for flip_set=%s, direction=%s at theta=%s -- "
                        "the Q1+Q2-encoded domain has no valid pair at all; not a "
                        "certificate",
                        flip_set,
                        direction,
                        theta,
                    )
                    return Inconclusive(
                        pareto.restore(), reason="EMPTY_DOMAIN", iteration=iter_num
                    )

                if confirmation.gap > eps + 1e-6:
                    log.error(
                        "feasibility search reported UNSAT at eps=%.6f but a fresh "
                        "optimality search just found gap=%.6f -- inconsistent, "
                        "not certifying",
                        eps,
                        confirmation.gap,
                    )
                    return Inconclusive(
                        pareto.restore(),
                        reason="INCONSISTENT_OPTIMALITY",
                        iteration=iter_num,
                    )

                certificate_snapshot: Snapshot = self._snapshot(
                    iter_num,
                    slack_mass=self._last_slack_mass,
                    worst_gap=confirmation.gap,
                )
                self._record(pareto, certificate_snapshot, label)

                return ConditionalCertificate(
                    oracle_type="sensei",
                    eps=eps,
                    theta=theta,
                    flip_set=flip_set,
                    direction=direction,
                    spec_hash=self.spec.spec_hash,
                    oracle=f"sensei-milp@{n_trees}trees",
                    solver_status="UNSAT",
                    time_limit_s=settings.oracle.time_limit_s,
                    n_cuts=len(self.cuts),
                    snapshot=certificate_snapshot,
                )

            worst: float = max(abs(p.gap) for p in found)

            for pair in found:
                self.cuts.append(CutBuilder.make_cut(pair, eps))

            if repair_type == "approx":
                self.v, slack = RepairQP.solve_qp_fast(
                    self.leaf_map.v0,
                    gram_train,
                    self.cuts,
                    settings.repair.mu,
                    settings.repair.kap,
                )
            else:
                self.v, slack = RepairQP.solve_qp_exact_small(
                    self.leaf_map.v0,
                    phi_train,
                    self.y_train.to_numpy(),
                    self.leaf_map.base_score,
                    self.cuts,
                    settings.repair.mu,
                    settings.repair.kap,
                )

            self._last_slack_mass = float(slack.sum())
            snapshot: Snapshot = self._snapshot(
                iter_num, slack_mass=self._last_slack_mass
            )
            self._record(pareto, snapshot, label)
            log.info(
                "iter %2d | trigger_gap %.4f | worst_gap %.4f | acc %.4f | sens %.4f "
                "| slack %.4f | cuts %d",
                iter_num,
                worst,
                snapshot.worst_gap,
                snapshot.accuracy,
                snapshot.sensitivity_rate,
                snapshot.slack_mass,
                snapshot.n_cuts,
            )

            if snapshot.accuracy < float(settings.loop.A_min):
                return ParetoBest(pareto.restore(), reason="accuracy_floor")

            if abs(previous_worst - worst) < float(settings.loop.stall_delta):
                return ParetoBest(pareto.restore(), reason="stalled")

            previous_worst: float = worst

        return ParetoBest(pareto.restore(), reason="iteration_cap")

    def _snapshot(
        self, iteration: int, slack_mass: float, worst_gap: float | None = None
    ) -> Snapshot:
        """
        `worst_gap` is computed fresh against whatever `self.v` currently is
        (mode="optimality", per the metric's own definition) -- NOT
        reused from whichever pair triggered this round's repair. Those are
        different numbers: the pair that triggered a repair was found
        against the model BEFORE that repair; by the time a snapshot is
        taken, `self.v` is already the repaired model. Pairing the new `v`
        with the old violation's gap silently misreports the model actually
        being snapshotted, including on the final ConditionalCertificate,
        where it previously hardcoded 0.0 regardless of what the true worst
        gap or accumulated slack actually was.

        Pass `worst_gap` explicitly to skip recomputing it when the caller
        already ran that exact optimality search (e.g. `_run_stage`'s
        empty-domain check, avoiding the same MILP solve twice).

        Args:
            iteration (int): The iteration number to stamp on the snapshot.
            slack_mass (float): `sum(s_i)` at this point in the loop.
            worst_gap (float | None): Precomputed worst gap, or None to
                compute it fresh via `_worst_gap_now`.

        Returns:
            Snapshot: This iteration's full state.
        """

        if worst_gap is None:
            worst_gap = self._worst_gap_now()

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

    def _worst_gap_now(self) -> float:
        """
        The worst valid gap for the CURRENT `self.v`, at this loop's own
        flip_set/direction/`self.settings`, from `self.settings.oracle.type`
        -- a proven-optimal value for "sensei", a one-shot heuristic result
        (not a true optimum) for "ensense", per `Metrics.worst_valid_gap`'s
        own docstring. Returns NaN (with a logged warning) if the search
        itself times out -- a timeout here must not be silently reported as
        "gap 0", which would misrepresent an unknown quantity as a
        known-safe one.

        Returns:
            float: The current worst valid gap, or NaN on timeout/saturation.
        """

        try:
            return Metrics.worst_valid_gap(
                self.booster,
                self.leaf_map,
                self.columns,
                self.feature_bounds,
                self.spec,
                self.bins,
                self.flip_set,
                self.direction,
                self.settings,
                structure=self.structure,
                v=self.v,
            )
        except OracleTimeout as e:
            log.warning(e)
            return float("nan")
        except OracleSaturated as e:
            log.warning(e)
            return float("nan")

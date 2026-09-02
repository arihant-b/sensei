import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from config import Config
from data import Dataset
from metrics import ParetoCheckpoint, Snapshot, accuracy, sensitivity_rate
from model import Ensemble
from oracle import Counterexample, OracleLike, SensitivityOracle
from repair import LeafRepairQP
from sampler import SamplingScreen

log = logging.getLogger(__name__)


class SensEILoop:
    """
    Counterexample-guided sensitivity repair.

      train -> verify -> cut -> re-solve QP -> verify -> ...

    Terminates when the oracle proves no valid undesirable counterexample
    with gap > epsilon remains. That UNSAT is the certificate.
    """

    def __init__(
        self,
        model: Ensemble,
        data: Dataset,
        oracle: OracleLike,
        sampler: SamplingScreen,
        cfg: Config,
    ) -> None:
        self.model = model
        self.data = data
        self.oracle = oracle
        self.sampler = sampler
        self.cfg = cfg

        self.qp = LeafRepairQP(model, data.X_train, cfg)
        self.checkpoint = ParetoCheckpoint(cfg.min_accuracy)
        self.certified = False

    def run(self, flip_set: list[str]) -> Snapshot | None:
        prev_gap = float("inf")

        for it in range(self.cfg.max_iters):
            ces, used_oracle = self._find_counterexamples(flip_set)

            if not ces:
                if used_oracle:
                    log.info("UNSAT at iteration %d -- certified", it)
                    self.certified = True
                    break

                # sampling found nothing; escalate to the exact oracle
                ces = self.oracle.find_batch(
                    self.model, flip_set, self.cfg.cuts_per_round)

                if not ces:
                    self.certified = True
                    break

            worst = max(abs(c.gap) for c in ces)

            for ce in ces:
                self.qp.add_cut(ce)

            v_new = self.qp.solve()
            self.model.set_leaf_values(v_new)

            # leaf values changed: previously-forbidden regions may now be fine
            # TODO: comment the following line if oscillation happens
            self.oracle.clear_nogoods()
            self.oracle.set_warm_start(ces[0])

            snap = self._snapshot(it, worst)
            self.checkpoint.record(snap)
            log.info("iter %2d | gap %.4f | acc %.4f | cuts %d",
                      it, worst, snap.accuracy, self.qp.n_cuts)

            if snap.accuracy < self.cfg.min_accuracy:
                log.warning("accuracy floor breached; stopping")
                break

            if abs(prev_gap - worst) < self.cfg.stall_delta:
                log.warning("stalled; stopping")
                break

            prev_gap = worst

        best = self.checkpoint.restore()

        if best is not None:
            self.model.set_leaf_values(best.leaf_values)

        return best

    # ------------------------------------------------------------ helpers
    def _find_counterexamples(
        self, flip_set: list[str]
    ) -> tuple[list[Counterexample], bool]:
        """Cheap screen first; only pay for the MILP when sampling comes up empty."""
        if self.cfg.use_sampling_screen:
            ces = self.sampler.find_violations(
                self.model, self.data.X_train, flip_set[0],
                k=self.cfg.cuts_per_round)

            if ces:
                return [self._to_ce(a, b, g) for a, b, g in ces], False

        return (
            self.oracle.find_batch(self.model, flip_set, self.cfg.cuts_per_round),
            True,
        )

    def _to_ce(self, x1, x2, gap: float) -> Counterexample:
        """
        Wrap a sampled (x1, x2, gap) triple as a Counterexample. leaves1/2
        must come from the SAME global indexing model.phi() uses -- a
        mismatch here silently produces cuts that constrain the wrong
        leaves (see CLAUDE.md, "Leaf indexing").
        """
        return Counterexample(
            x1=np.asarray(x1, dtype=float),
            x2=np.asarray(x2, dtype=float),
            gap=float(gap),
            leaves1=self._leaf_ids(x1),
            leaves2=self._leaf_ids(x2),
        )

    def _leaf_ids(self, x) -> np.ndarray:
        """Global leaf indices model.phi() assigns to a single row x."""
        if isinstance(x, pd.Series):
            row = x.to_frame().T
        elif isinstance(x, pd.DataFrame):
            row = x.iloc[[0]]
        else:
            row = pd.DataFrame(np.atleast_2d(x), columns=self.data.columns)

        return self.model.phi(row).indices.copy()

    def _snapshot(self, it: int, worst_gap: float) -> Snapshot:
        assert self.model.v is not None, "model must be fit before snapshotting"
        assert self.data.X_test is not None and self.data.y_test is not None, (
            "call data.load() before running the loop"
        )
        return Snapshot(
            iteration=it,
            accuracy=accuracy(self.model, self.data.X_test, self.data.y_test),
            sensitivity_rate=sensitivity_rate(
                self.model, self.data.X_test, None, self.sampler),
            worst_gap=worst_gap,
            n_cuts=self.qp.n_cuts,
            leaf_values=self.model.v.copy(),
        )

    def verify_generalization(
        self, flip_set: list[str], seed: int
    ) -> Counterexample | None:
        """
        Fresh oracle run with a different seed and no accumulated no-goods.
        Adversarial training famously patches only what it saw -- this is
        the check for that.
        """
        if not isinstance(self.oracle, SensitivityOracle):
            raise TypeError(
                "verify_generalization needs a real SensitivityOracle "
                "(reads its validity/spec) -- not an OracleLike stand-in")

        fresh_cfg = replace(self.cfg, seed=seed)
        fresh_oracle = SensitivityOracle(
            self.oracle.validity, self.oracle.spec, fresh_cfg)

        ce: Counterexample | None = fresh_oracle.find_worst(self.model, flip_set)

        if ce is None:
            log.info(
                "generalisation check on %s: UNSAT under seed %d -- repair holds",
                flip_set, seed)
        else:
            log.warning(
                "generalisation check on %s: gap %.4f found under seed %d -- "
                "repair did NOT generalise",
                flip_set, ce.gap, seed)

        return ce

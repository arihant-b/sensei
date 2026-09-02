from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from data import FrozenBins
from oracle import Counterexample
from sampler import SamplingScreen
from spec import SensitivitySpec


class PredictsLike(Protocol):
    """Structural type for accuracy()/sensitivity_rate(): anything with
    Ensemble.predict's signature works, not just Ensemble itself (e.g.
    a plain classifier wrapped for baseline comparison in experiments/)."""

    def predict(self, X, v: np.ndarray | None = None) -> np.ndarray: ...


@dataclass
class Snapshot:
    iteration: int
    accuracy: float
    sensitivity_rate: float
    worst_gap: float
    n_cuts: int
    leaf_values: np.ndarray


def accuracy(
    model: PredictsLike, X: pd.DataFrame, y: np.ndarray, v: np.ndarray | None = None
) -> float:
    return float((model.predict(X, v) == y).mean())


def sensitivity_rate(
    model: PredictsLike,
    X: pd.DataFrame,
    spec: SensitivitySpec | None,
    sampler: SamplingScreen,
    v: np.ndarray | None = None,
) -> float:
    """
    Fraction of sampled rows whose predicted LABEL flips when at least one
    protected feature is flipped.

    This is a raw behavioural measurement, not a certificate: unlike the
    oracle it does not gate on validity/plausibility, and unlike the
    sampling screen it can only say "this many flipped", never "none do".
    `spec` defaults to the sampler's own (declared, not inferred) spec
    when not given explicitly.
    """
    active_spec = spec if spec is not None else sampler.spec
    protected = active_spec.protected

    if not protected or len(X) == 0:
        return 0.0

    n = min(sampler.cfg.n_sample_rows, len(X))
    idx = sampler.rng.choice(len(X), size=n, replace=False)
    rows = X.iloc[idx]

    base = model.predict(rows, v)
    flipped_any = np.zeros(len(rows), dtype=bool)

    for feature in protected:
        flipped_rows = sampler._flip(rows, feature)
        flipped_any |= model.predict(flipped_rows, v) != base

    return float(flipped_any.mean())


def held_out_density_score(
    counterexamples: list[Counterexample], eval_density_model: FrozenBins
) -> float:
    """
    Mean log-likelihood under a density model fitted on X_eval only.
    Independent judge: X_eval never touched the validity module.
    """
    if not counterexamples:
        return float("nan")

    # FrozenBins keys its per-feature dicts in the exact fit() column
    # order, so this reconstructs the feature-name -> value mapping
    # log_marginal() needs from the bare arrays Counterexample stores.
    columns = list(eval_density_model.edges.keys())
    scores = [
        eval_density_model.log_marginal(dict(zip(columns, x, strict=True)))
        for ce in counterexamples
        for x in (ce.x1, ce.x2)
    ]
    return float(np.mean(scores))


class ParetoCheckpoint:
    """Safety net: the loop may oscillate or degrade. Keep the best model seen."""

    def __init__(self, min_accuracy: float):
        self.min_accuracy = min_accuracy
        self.history: list[Snapshot] = []
        self.best: Snapshot | None = None

    def record(self, snap: Snapshot):
        self.history.append(snap)
        if snap.accuracy < self.min_accuracy:
            return
        if self.best is None or snap.worst_gap < self.best.worst_gap:
            self.best = snap

    def restore(self) -> Snapshot | None:
        return self.best

"""
Synthetic fixtures for testing without Gurobi/Ensense.

A tiny 3-tree Ensemble with a planted, exactly-known sensitivity
violation between two synthetic rows, plus a FakeOracle that returns
hand-made Counterexamples so repair.py/loop.py can be exercised without
a real solver. Per CLAUDE.md: get repair.py/loop.py working against
this before touching Ensense.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config  # noqa: E402
from model import Ensemble  # noqa: E402
from oracle import Counterexample  # noqa: E402

FEATURES = ["protected", "free_a", "free_b"]
N_TREES = 3


def make_synthetic_dataset(
    seed: int = 0, n: int = 200
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    'protected' dominates the label (coefficient 2.0 vs free_a's 1.0) so
    a tiny tree ensemble reliably splits on it -- that is what makes the
    planted violation below a property of the FITTED trees, not a
    coincidence of one random draw.
    """
    rng = np.random.default_rng(seed)
    protected = rng.integers(0, 2, size=n).astype(float)
    free_a = rng.uniform(0.0, 1.0, size=n)
    free_b = rng.uniform(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 0.05, size=n)

    score = 2.0 * protected + free_a - 0.5 + noise
    label = (score > 0.5).astype(int)

    X = pd.DataFrame({"protected": protected, "free_a": free_a, "free_b": free_b})
    return X, label


def make_synthetic_model(seed: int = 0) -> Ensemble:
    cfg = Config(n_estimators=N_TREES, max_depth=2, seed=seed)
    X, y = make_synthetic_dataset(seed=seed)
    return Ensemble(cfg).fit(X, y)


def make_planted_counterexample(model: Ensemble) -> Counterexample:
    """
    Two rows differing only in 'protected', chosen so the fitted model's
    raw score gap between them is large -- the fixture's known
    violation. leaves1/leaves2 come from model.phi() itself, so they are
    guaranteed consistent with model.leaf_index by construction.
    """
    x1 = pd.DataFrame([{"protected": 1.0, "free_a": 0.1, "free_b": 0.1}])
    x2 = pd.DataFrame([{"protected": 0.0, "free_a": 0.1, "free_b": 0.1}])

    gap = float(model.raw_score(x1)[0] - model.raw_score(x2)[0])
    leaves1 = model.phi(x1).indices.copy()
    leaves2 = model.phi(x2).indices.copy()

    return Counterexample(
        x1=x1.iloc[0].to_numpy(dtype=float),
        x2=x2.iloc[0].to_numpy(dtype=float),
        gap=gap,
        leaves1=leaves1,
        leaves2=leaves2,
    )


class FakeOracle:
    """
    Stands in for SensitivityOracle: serves the one planted
    counterexample the first time it's asked, then reports UNSAT (None)
    forever after -- simulating "found the one violation, repaired it,
    nothing left". Mirrors SensitivityOracle's public surface closely
    enough for loop.py to drive it identically.
    """

    def __init__(self, planted: Counterexample):
        self._planted = planted
        self._served = False
        self._nogoods: list[np.ndarray] = []
        self._warm_start: Counterexample | None = None

    def find_worst(
        self, model, flip_set, certifying: bool = False
    ) -> Counterexample | None:
        if self._served:
            return None
        self._served = True
        return self._planted

    def find_batch(self, model, flip_set, k: int, certifying: bool = False) -> list:
        found = []
        for _ in range(k):
            ce = self.find_worst(model, flip_set, certifying=certifying)
            if ce is None:
                break
            found.append(ce)
            self.add_nogood(ce)
        return found

    def add_nogood(self, ce: Counterexample) -> None:
        self._nogoods.append(np.union1d(ce.leaves1, ce.leaves2))

    def clear_nogoods(self) -> None:
        self._nogoods = []

    def set_warm_start(self, ce: Counterexample) -> None:
        self._warm_start = ce

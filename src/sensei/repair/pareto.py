from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sensei.repair.cuts import Cut


@dataclass
class Snapshot:
    """One CEGSAL iteration's (v, accuracy, sensitivity) state, for Pareto tracking."""

    iteration: int
    v: NDArray[np.float64]
    accuracy: float
    sensitivity_rate: float
    worst_gap: float
    slack_mass: float
    n_cuts: int
    cuts: list[Cut] = field(default_factory=list)


class ParetoCheckpoint:
    def __init__(self, a_min: float) -> None:
        self.a_min: float = a_min
        self.history: list[Snapshot] = []
        self.best: Snapshot | None = None

    def record(self, snapshot: Snapshot) -> None:
        """
        Append to history; update `best` if `snapshot` clears `a_min` and
        beats it on worst-gap magnitude.

        Args:
            snapshot (Snapshot): The iteration's state to record.
        """

        self.history.append(snapshot)

        if snapshot.accuracy < self.a_min:
            return

        if (
            self.best is None
            or np.isnan(self.best.worst_gap)
            or abs(snapshot.worst_gap) < abs(self.best.worst_gap)
        ):
            self.best = snapshot

    def restore(self) -> Snapshot | None:
        """
        The best snapshot recorded so far, or None if none has cleared `a_min`.

        Returns:
            Snapshot | None: The best snapshot, or None.
        """

        return self.best

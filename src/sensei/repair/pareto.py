from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sensei.repair.cuts import Cut


@dataclass
class Snapshot:
    """
    A snapshot of the current state of the repair process, capturing key metrics and
    information about the current iteration.
    """

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
        Record a snapshot of the current state of the repair process.

        Args:
            snap (Snapshot): The snapshot to record, containing metrics and information
                             about the current iteration.
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
        Restore the best snapshot recorded during the repair process.

        Returns:
            Snapshot | None: The best snapshot recorded, or None if no valid snapshot
                             was recorded.
        """

        return self.best

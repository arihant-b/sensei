from typing import Protocol

import xgboost as xgb

from sensei.oracle.types import Pair
from sensei.spec import Spec


class SensitivityOracle(Protocol):
    """
    A protocol for sensitivity oracles that can find the worst valid pair of points
    in a given context. Both Tier A and Tier B implement this shape.
    """

    def worst_valid_pair(
        self,
        booster: xgb.Booster,
        spec: Spec,
        theta: float,
        flip_set: tuple[str, ...],
        direction: str,
        mode: str,  # "feasibility" | "optimality"
        eps: float | None,  # required iff mode == "feasibility"
        seed: int,
        time_limit_s: float,
        nogoods=None,  # Tier A only
        warm_start=None,
    ) -> Pair | None: ...

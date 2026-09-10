import logging
from dataclasses import dataclass

import xgboost as xgb

from sensei.config import Settings
from sensei.data.bins import FrozenBins
from sensei.model.leaves import LeafMap
from sensei.oracle.ensense.adapter import EnsenseOracle
from sensei.oracle.types import OracleDegenerate, OracleSaturated, Pair
from sensei.spec import Spec

log: logging.Logger = logging.getLogger("sensei.eval.heldout_verify")


@dataclass(frozen=True)
class HeldoutResult:
    """Result of a held-out verification on a repaired booster."""

    flip_set: tuple[str, ...]
    generalized: bool | None  # True=fresh UNSAT, False=fresh SAT, None=inconclusive
    fresh_pair: Pair | None
    note: str


class HeldoutVerifier:
    """
    Re-check a repaired booster with Ensense core (via `EnsenseOracle`), independent
    of whatever repaired it (`SenseiOracle`).
    """

    @staticmethod
    def verify(
        booster: xgb.Booster,
        columns: list[str],
        flip_set: tuple[str, ...],
        spec: Spec,
        bins: FrozenBins,
        feature_bounds: dict[str, tuple[float, float]],
        settings: Settings,
    ) -> HeldoutResult:
        """
        Held-out check: fresh seed, fresh start, via Ensense core --
        fresh UNSAT means the repair generalized; fresh SAT means it only
        patched specific points. Distinct from a saturated/rejected postfilter,
        which is inconclusive, not evidence either way.

        Args:
            booster (xgb.Booster): The repaired model to re-check.
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            spec (Spec): Dataset spec, for the Ensense postfilter.
            bins (FrozenBins): Frozen plausibility bins.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            settings (Settings): Supplies `oracle.method`, `sensitivity.gap`/
                `.theta`, `oracle.time_limit_s`.

        Returns:
            HeldoutResult: Whether the repair generalized, with the fresh
                pair (if any) and an explanatory note.
        """

        leaf_map = LeafMap(booster)

        try:
            pair: Pair | None = EnsenseOracle().worst_valid_pair(
                booster,
                leaf_map,
                columns,
                flip_set,
                spec,
                bins,
                feature_bounds,
                settings,
            )
        except (OracleDegenerate, OracleSaturated) as e:
            log.warning(
                "held-out verify on %s: ensense oracle rejected/saturated -- "
                "inconclusive, not evidence either way (%s)",
                flip_set,
                e,
            )
            return HeldoutResult(
                flip_set=flip_set,
                generalized=None,
                fresh_pair=None,
                note=(
                    f"Ensense's postfilter could not produce a valid pair ({e}) -- "
                    "rejected/saturated. This is NOT the same as Ensense finding no "
                    "violation; report it as inconclusive, not as generalized."
                ),
            )

        if pair is None:
            log.info(
                "held-out verify on %s: fresh UNSAT (Ensense core) -- repair holds",
                flip_set,
            )
            return HeldoutResult(
                flip_set=flip_set,
                generalized=True,
                fresh_pair=None,
                note=(
                    "Ensense core (via EnsenseOracle, independent of the SenseiOracle "
                    "repairer) found no violation. Caveat: not distinguishable from a "
                    "timeout at these settings -- this is evidence, not a proof of "
                    "UNSAT."
                ),
            )

        log.warning(
            "held-out verify on %s: fresh SAT (gap=%.4f) -- repair did NOT generalize",
            flip_set,
            pair.gap,
        )
        return HeldoutResult(
            flip_set=flip_set,
            generalized=False,
            fresh_pair=pair,
            note="Ensense core found a fresh violation -- the repair patched specific "
            "cases, not the property.",
        )

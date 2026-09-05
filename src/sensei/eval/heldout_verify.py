import logging
from dataclasses import dataclass

import xgboost as xgb

from sensei.model.leaves import LeafMap
from sensei.oracle.ensense_adapter import TierBOracle
from sensei.oracle.types import Pair

log: logging.Logger = logging.getLogger("sensei.eval.heldout_verify")


@dataclass(frozen=True)
class HeldoutResult:
    """
    Result of a held-out verification on a repaired booster.
    """

    flip_set: tuple[str, ...]
    generalized: bool  # True iff Ensense core (fresh, independent) found nothing
    fresh_pair: Pair | None
    note: str


class HeldoutVerifier:
    """
    Re-check a repaired booster with Ensense core (Tier B), independent of whatever
    repaired it (Tier A).
    """

    @staticmethod
    def verify(
        booster: xgb.Booster,
        columns: list[str],
        flip_set: tuple[str, ...],
        details_csv: str | None,
        output_gap: tuple[float, float],
        timeout: int = 120,
        method: str = "pb",
    ) -> HeldoutResult:
        """
        Verify a repaired booster with Ensense core (Tier B), independent of whatever
        repaired it (Tier A).

        Args:
            booster (xgb.Booster): The repaired XGBoost booster to verify.
            columns (list[str]): The list of column names in the dataset.
            flip_set (tuple[str, ...]): The set of features that have been flipped.
            details_csv (str | None): The path to the CSV file containing details of the
                                      verification.
            output_gap (tuple[float, float]): The gap in the output values.
            timeout (int, optional): The timeout for the verification process. Defaults
                                     to 120.
            method (str, optional): The method to use for the verification. Defaults to
                                    "pb".

        Returns:
            HeldoutResult: _description_
        """

        leaf_map = LeafMap(booster)
        pair: Pair | None = TierBOracle().worst_valid_pair(
            booster,
            leaf_map,
            columns,
            flip_set,
            method=method,
            details_csv=details_csv,
            output_gap=output_gap,
            timeout=timeout,
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
                    "Ensense core (Tier B, independent of the Tier A repairer) found "
                    "no violation. Caveat: not distinguishable from a timeout at these "
                    "settings -- this is evidence, not a proof of UNSAT."
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

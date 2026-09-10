from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy._typing._array_like import NDArray

from sensei.model.leaves import LeafMap
from sensei.oracle.types import Pair
from sensei.repair.cuts import Cut


@dataclass(frozen=True)
class OverlapResult:
    """
    Result of a region-overlap check for a fresh counterexample.
    """

    matched_existing_cut: bool
    fresh_d_positive: frozenset[int]  # leaves only in x1's set
    fresh_d_negative: frozenset[int]  # leaves only in x2's set


class RegionOverlapAnalyzer:
    """
    Check whether a fresh counterexample's leaf-difference pattern matches one already
    cut. High overlap with a low fresh-violation count is the evidence for the
    regions-not-points claim; this module computes the overlap, it doesn't assert the
    conclusion.
    """

    @staticmethod
    def check_overlap(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        fresh_pair: Pair,
        columns: list[str],
        cuts: list[Cut],
    ) -> OverlapResult:
        """
        `fresh_pair`'s (pos, neg) leaf-difference sets match an existing
        cut's, up to sign (the cut doesn't record which side was x1 vs x2).

        Args:
            booster (xgb.Booster): The model `fresh_pair` was found against.
            leaf_map (LeafMap): Global leaf map for `booster`.
            fresh_pair (Pair): The freshly found counterexample to check.
            columns (list[str]): All feature names, in model column order.
            cuts (list[Cut]): Already-accumulated cuts to check overlap against.

        Returns:
            OverlapResult: Whether `fresh_pair` matched an existing cut,
                with its own difference sets.
        """

        row1 = pd.DataFrame([fresh_pair.x1], columns=columns)
        row2 = pd.DataFrame([fresh_pair.x2], columns=columns)

        ell1: NDArray[np.int64] = leaf_map.leaf_ids_for_row(booster, row1)
        ell2: NDArray[np.int64] = leaf_map.leaf_ids_for_row(booster, row2)

        fresh_pos, fresh_neg = RegionOverlapAnalyzer._difference_sets(ell1, ell2)

        for cut in cuts:
            cut_pos: frozenset[int] = frozenset(
                int(i) for i, c in zip(cut.idx, cut.coef, strict=True) if c > 0
            )
            cut_neg: frozenset[int] = frozenset(
                int(i) for i, c in zip(cut.idx, cut.coef, strict=True) if c < 0
            )

            if (fresh_pos, fresh_neg) in ((cut_pos, cut_neg), (cut_neg, cut_pos)):
                return OverlapResult(True, fresh_pos, fresh_neg)

        return OverlapResult(False, fresh_pos, fresh_neg)

    @staticmethod
    def _difference_sets(
        ell1: NDArray[np.int64], ell2: NDArray[np.int64]
    ) -> tuple[frozenset[int], frozenset[int]]:
        """
        (leaves only in `ell1`, leaves only in `ell2`) -- shared leaves cancel.

        Args:
            ell1 (NDArray[np.int64]): x1's active global leaf indices.
            ell2 (NDArray[np.int64]): x2's active global leaf indices.

        Returns:
            tuple[frozenset[int], frozenset[int]]: `(only in ell1, only in
                ell2)`.
        """

        set1: set[int] = {int(n) for n in ell1}
        set2: set[int] = {int(n) for n in ell2}
        return frozenset(set1 - set2), frozenset(set2 - set1)

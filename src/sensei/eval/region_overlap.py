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
        Check the overlap of a fresh counterexample with existing cuts.

        Args:
            booster (xgb.Booster): The XGBoost booster to analyze.
            leaf_map (LeafMap): The leaf map to use for leaf ID computation.
            fresh_pair (Pair): The fresh counterexample to check.
            columns (list[str]): The column names for the input data.
            cuts (list[Cut]): The existing cuts to compare against.

        Returns:
            OverlapResult: The result of the overlap check.
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
    def overlap_rate(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        fresh_pairs: list[Pair],
        columns: list[str],
        cuts: list[Cut],
    ) -> float:
        """
        Fraction of fresh_pairs whose exact difference pattern was already cut.

        Args:
            booster (xgb.Booster): The XGBoost booster to analyze.
            leaf_map (LeafMap): The leaf map to use for leaf ID computation.
            fresh_pairs (list[Pair]): The fresh counterexamples to check.
            columns (list[str]): The column names for the input data.
            cuts (list[Cut]): The existing cuts to compare against.

        Returns:
            float: The overlap rate.
        """

        if not fresh_pairs:
            return float("nan")

        matches: int = sum(
            RegionOverlapAnalyzer.check_overlap(
                booster, leaf_map, p, columns, cuts
            ).matched_existing_cut
            for p in fresh_pairs
        )
        return matches / len(fresh_pairs)

    @staticmethod
    def _difference_sets(
        ell1: NDArray[np.int64], ell2: NDArray[np.int64]
    ) -> tuple[frozenset[int], frozenset[int]]:
        """
        Compute the difference between two sets of leaf IDs.

        Args:
            ell1 (NDArray[np.int64]): The first set of leaf IDs.
            ell2 (NDArray[np.int64]): The second set of leaf IDs.

        Returns:
            tuple[frozenset[int], frozenset[int]]: The difference sets.
        """

        set1: set[int] = {int(n) for n in ell1}
        set2: set[int] = {int(n) for n in ell2}
        return frozenset(set1 - set2), frozenset(set2 - set1)

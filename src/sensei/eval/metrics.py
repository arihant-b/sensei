import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.sparse._csr import csr_matrix

from sensei.config import Settings
from sensei.data.bins import FrozenBins
from sensei.eval.region_overlap import RegionOverlapAnalyzer
from sensei.model.leaves import LeafMap
from sensei.oracle.ensense.adapter import EnsenseOracle
from sensei.oracle.sensei.solve import SenseiOracle
from sensei.oracle.types import Pair, TreeStructure
from sensei.repair.cuts import Cut
from sensei.spec import Spec

log: logging.Logger = logging.getLogger("sensei.eval.metrics")


class Metrics:
    """
    Frozen metric definitions. Changing a definition invalidates every earlier result.
    Bump Metrics.VERSION and re-run everything if one must change.
    """

    VERSION = "0.1.0"

    @staticmethod
    def predict(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        X: pd.DataFrame,
        v: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        """
        Threshold `E_v(x) > 0` (margin, not probability). `v` defaults to `v0`.

        Args:
            booster (xgb.Booster): The model to score.
            leaf_map (LeafMap): Global leaf map for `booster`.
            X (pd.DataFrame): Rows to predict.
            v (NDArray[np.float64] | None): Leaf values to score with;
                `leaf_map.v0` if None.

        Returns:
            NDArray[np.int64]: `{0, 1}` predictions, one per row.
        """

        v = leaf_map.v0 if v is None else v
        phi: csr_matrix = leaf_map.phi_for(booster, X)
        margin: NDArray[np.float64] = leaf_map.margin_score(phi, v)
        return (np.asarray(margin) > 0).astype(int)

    @staticmethod
    def accuracy(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        X: pd.DataFrame,
        y: pd.Series,
        v: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Fraction of `X`/`y` where `predict` matches the true label.

        Args:
            booster (xgb.Booster): The model to score.
            leaf_map (LeafMap): Global leaf map for `booster`.
            X (pd.DataFrame): Rows to predict.
            y (pd.Series): True labels, `{0, 1}`.
            v (NDArray[np.float64] | None): Leaf values to score with;
                `leaf_map.v0` if None.

        Returns:
            float: Test accuracy on `X`/`y`.
        """

        preds: NDArray[np.int64] = Metrics.predict(booster, leaf_map, X, v)
        return float((preds == np.asarray(y)).mean())

    @staticmethod
    def sensitivity_rate(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        X: pd.DataFrame,
        protected: tuple[str, ...],
        v: NDArray[np.float64] | None = None,
        n_sample_rows: int = 1000,
        seed: int = 42,
    ) -> float:
        """
        Sensitivity rate: fraction of `n_sample_rows` held-out rows
        whose prediction flips when any `protected` feature is bumped to its
        next sorted level (`_flip_next_level`).

        Args:
            booster (xgb.Booster): The model to score.
            leaf_map (LeafMap): Global leaf map for `booster`.
            X (pd.DataFrame): Held-out rows to sample from.
            protected (tuple[str, ...]): Protected features to flip.
            v (NDArray[np.float64] | None): Leaf values to score with;
                `leaf_map.v0` if None.
            n_sample_rows (int): Number of rows to sample from `X`.
            seed (int): Sampling seed.

        Returns:
            float: Fraction of sampled rows whose prediction flipped.
        """

        if not protected or len(X) == 0:
            return 0.0

        rng: Generator = np.random.default_rng(seed)
        n: int = min(n_sample_rows, len(X))
        idx: NDArray[np.int64] = rng.choice(len(X), size=n, replace=False)
        rows: pd.DataFrame = X.iloc[idx]

        base: NDArray[np.int64] = Metrics.predict(booster, leaf_map, rows, v)
        flipped_any: NDArray[np.bool] = np.zeros(len(rows), dtype=bool)

        for feature in protected:
            flipped_rows: pd.DataFrame = Metrics._flip_next_level(rows, feature)
            flipped_any |= Metrics.predict(booster, leaf_map, flipped_rows, v) != base

        return float(flipped_any.mean())

    @staticmethod
    def worst_valid_gap(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec,
        bins: FrozenBins,
        flip_set: tuple[str, ...],
        direction: str,
        settings: Settings,
        structure: TreeStructure | None = None,
        v: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Worst valid gap for reporting/sweeps: the given oracle's
        `worst_valid_pair` in mode="optimality" (true max signed gap). NaN on
        `EMPTY_DOMAIN` -- a misconfiguration, never a certified 0.

        `settings.oracle.type="sensei"` solves to a proven-optimal worst
        gap. `"ensense"` has no equivalent notion of "solve to optimality"
        -- Ensense's search is a one-shot heuristic, so this reports
        whatever single pair that search happens to return, not a proven
        worst case. Use it only as a rough cross-check against the Sensei
        oracle's number, never as the reported worst gap on its own.

        Args:
            booster (xgb.Booster): The model to search.
            leaf_map (LeafMap): Global leaf map for `booster`.
            columns (list[str]): All feature names, in model column order.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            spec (Spec): Dataset spec, for validity/plausibility encoding.
            bins (FrozenBins): Frozen plausibility bins.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            settings (Settings): Supplies `oracle.type` and everything the
                selected oracle needs.
            structure (TreeStructure | None): Pre-extracted tree structure;
                extracted fresh if None.
            v (NDArray[np.float64] | None): Leaf values to search with;
                `leaf_map.v0` if None.

        Returns:
            float: The worst valid gap, or NaN on `EMPTY_DOMAIN`.
        """

        oracle_type: str = settings.oracle.type
        oracle: SenseiOracle | EnsenseOracle = (
            SenseiOracle() if oracle_type == "sensei" else EnsenseOracle()
        )

        if oracle_type == "sensei":
            assert isinstance(oracle, SenseiOracle)
            search: Callable[..., Pair | None] = oracle.worst_valid_pair
        else:
            assert isinstance(oracle, EnsenseOracle)
            search = oracle.worst_valid_pair_loop

        pair: Pair | None = search(
            booster,
            leaf_map,
            columns,
            feature_bounds,
            spec,
            flip_set,
            direction,
            mode="optimality",
            settings=settings,
            enforce_validity=True,
            structure=structure,
            v=v,
            bins=bins,
        )

        if pair is None:
            log.warning(
                "worst_valid_gap: EMPTY_DOMAIN for flip_set=%s -- misconfiguration,"
                "not a certified zero gap",
                flip_set,
            )
            return float("nan")

        return pair.gap

    @staticmethod
    def slack_mass(slack: NDArray[np.float64]) -> float:
        """
        `sum(s_i)` at repair termination -- >0 is a finding, not a failure.

        Args:
            slack (NDArray[np.float64]): Per-cut slack values.

        Returns:
            float: The total slack mass.
        """

        return float(np.sum(slack))

    @staticmethod
    def overlap_rate(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        fresh_pairs: list[Pair],
        columns: list[str],
        cuts: list[Cut],
    ) -> float:
        """
        Fraction of fresh_pairs whose exact leaf-difference pattern was already
        cut. The per-pair matching logic (does this pair's `d` equal an
        existing cut's, up to sign) lives in
        `region_overlap.py::RegionOverlapAnalyzer.check_overlap` -- this is
        just the aggregate over many pairs, same shape as `sensitivity_rate`.
        High overlap alongside a low fresh-violation count is the evidence
        that a cut fixes a whole region of input space, not one point (see
        `eval/heldout_verify.py`).

        Args:
            booster (xgb.Booster): The model `fresh_pairs` were found against.
            leaf_map (LeafMap): Global leaf map for `booster`.
            fresh_pairs (list[Pair]): Freshly found counterexamples to check.
            columns (list[str]): All feature names, in model column order.
            cuts (list[Cut]): Already-accumulated cuts to check overlap against.

        Returns:
            float: Fraction of `fresh_pairs` matching an existing cut, or
                NaN if `fresh_pairs` is empty.
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
    def _flip_next_level(rows: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Roll `feature` to the next level in its sorted unique values
        (unchanged if only one level exists).

        Args:
            rows (pd.DataFrame): Rows to flip `feature` on.
            feature (str): The feature to roll to its next level.

        Returns:
            pd.DataFrame: `rows` with `feature` rolled, a copy.
        """

        levels: NDArray[np.str_] = np.sort(rows[feature].unique())

        if len(levels) <= 1:
            return rows.copy()

        next_level: dict[str, str] = dict(zip(levels, np.roll(levels, -1), strict=True))
        flipped: pd.DataFrame = rows.copy()
        flipped[feature] = rows[feature].map(next_level)
        return flipped

import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.sparse._csr import csr_matrix

from sensei.data.bins import FrozenBins
from sensei.model.leaves import LeafMap
from sensei.oracle.milp.encoding import TreeStructure
from sensei.oracle.milp.solve import TierAOracle
from sensei.oracle.types import Pair
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
        Predict labels for a dataset using a repaired booster and its leaf map.

        Args:
            booster (xgb.Booster): The XGBoost booster to use for prediction.
            leaf_map (LeafMap): The leaf map corresponding to the booster.
            X (pd.DataFrame): The input features for prediction.
            v (NDArray[np.float64] | None, optional): The leaf values to use for
                                                      prediction. If None, the default
                                                      leaf values from the leaf map are
                                                      used. Defaults to None.

        Returns:
            NDArray[np.int64]: The predicted labels as a NumPy array of integers.
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
        Compute the accuracy of a repaired booster on a given dataset.

        Args:
            booster (xgb.Booster): The XGBoost booster to evaluate.
            leaf_map (LeafMap): The leaf map corresponding to the booster.
            X (pd.DataFrame): The input features for evaluation.
            y (pd.Series): The true labels for the evaluation dataset.
            v (NDArray[np.float64] | None, optional): The leaf values to use for
                                                      prediction. If None, the default
                                                      leaf values from the leaf map are
                                                      used. Defaults to None.

        Returns:
            float: The accuracy of the booster on the evaluation dataset, computed as
                   the fraction of correctly predicted labels.
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
        Compute the sensitivity rate of a repaired booster on a given dataset with
        respect to a set of protected features. The sensitivity rate is defined as the
        fraction of rows in the dataset for which flipping any of the protected features
        results in a change in the predicted label.

        Args:
            booster (xgb.Booster): The XGBoost booster to evaluate.
            leaf_map (LeafMap): The leaf map corresponding to the booster.
            X (pd.DataFrame): The input features for evaluation.
            protected (tuple[str, ...]): The set of protected features.
            v (NDArray[np.float64] | None, optional): The leaf values to use for
                                                      prediction. If None, the default
                                                      leaf values from the leaf map are
                                                      used. Defaults to None.
            n_sample_rows (int, optional): The number of sample rows to use for the
                                           evaluation. Defaults to 1000.
            seed (int, optional): The random seed to use for sampling. Defaults to 42.

        Returns:
            float: The sensitivity rate of the booster on the evaluation dataset.
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
        theta: float,
        seed: int,
        time_limit_s: float,
        mip_gap: float,
        structure: TreeStructure | None = None,
        v: NDArray[np.float64] | None = None,
    ) -> float:
        """
        Compute the worst valid gap for a repaired booster with respect to a set of
        flipped features. The worst valid gap is defined as the maximum difference in
        predicted labels between the original and flipped datasets, subject to the
        constraints defined by the dataset specification and the frozen bins.

        Args:
            booster (xgb.Booster): The XGBoost booster to evaluate.
            leaf_map (LeafMap): The leaf map for the booster.
            columns (list[str]): The list of column names in the dataset.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.
            spec (Spec): The dataset specification.
            bins (FrozenBins): The frozen bins for the dataset.
            flip_set (tuple[str, ...]): The set of features to flip.
            direction (str): The direction of the flip.
            theta (float): The threshold for the gap.
            seed (int): The random seed.
            time_limit_s (float): The time limit in seconds.
            mip_gap (float): The MIP gap.
            structure (TreeStructure | None, optional): The tree structure. Defaults to
                                                        None.
            v (NDArray[np.float64] | None, optional): The vector of values. Defaults to
                                                      None.

        Returns:
            float: The worst valid gap for the repaired booster with respect to the
                   flipped features.
        """

        oracle = TierAOracle()
        pair: Pair | None = oracle.worst_valid_pair(
            booster,
            leaf_map,
            columns,
            feature_bounds,
            spec,
            flip_set,
            direction,
            mode="optimality",
            eps=0.0,
            seed=seed,
            time_limit_s=time_limit_s,
            mip_gap=mip_gap,
            enforce_validity=True,
            structure=structure,
            v=v,
            bins=bins,
            theta=theta,
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
        Compute the sum of the slack values at repair termination.

        Args:
            slack (NDArray[np.float64]): The array of slack values at repair
                                         termination.

        Returns:
            float: The sum of the slack values.
        """

        return float(np.sum(slack))

    @staticmethod
    def _flip_next_level(rows: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Flip the values of a given feature in a DataFrame to the next level in its
        sorted unique values. If the feature has only one unique value, the DataFrame
        is returned unchanged.

        Args:
            rows (pd.DataFrame): The DataFrame containing the rows to flip.
            feature (str): The feature whose values are to be flipped.

        Returns:
            pd.DataFrame: The DataFrame with the specified feature flipped to the next
                          level.
        """

        levels: NDArray[np.str_] = np.sort(rows[feature].unique())

        if len(levels) <= 1:
            return rows.copy()

        next_level: dict[str, str] = dict(zip(levels, np.roll(levels, -1), strict=True))
        flipped: pd.DataFrame = rows.copy()
        flipped[feature] = rows[feature].map(next_level)
        return flipped

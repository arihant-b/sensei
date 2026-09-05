from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from scipy.sparse._csr import csr_matrix

from sensei.model.leaves import LeafMap
from sensei.oracle.types import Pair
from sensei.spec import Direction, Spec


@dataclass(frozen=True)
class BaselineResult:
    """
    The result of fitting a baseline model.
    """

    name: str
    booster: xgb.Booster
    columns: list[str]
    policy: str | None = None


class Baselines:
    """
    Stage 0 baselines. `policy=` is required with no default for baseline 4 so it can
    never be chosen by accident.

    Raises:
        ValueError: If an invalid policy is provided for baseline 4.
    """

    @staticmethod
    def fit_plain(
        X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, seed: int
    ) -> BaselineResult:
        """
        Fit a plain XGBoost classifier.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.
            n_estimators (int): The number of trees to build.
            max_depth (int): The maximum depth of the trees.
            seed (int): The random seed for reproducibility.

        Returns:
            BaselineResult: The result of fitting the baseline model.
        """

        classifier = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        classifier.fit(X, y)
        return BaselineResult("plain", classifier.get_booster(), list(X.columns))

    @staticmethod
    def fit_dropped(
        X: pd.DataFrame,
        y: pd.Series,
        spec: Spec,
        n_estimators: int,
        max_depth: int,
        seed: int,
    ) -> BaselineResult:
        """
        Fit an XGBoost classifier after dropping protected features.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.
            spec (Spec): The specification containing the protected features.
            n_estimators (int): The number of trees to build.
            max_depth (int): The maximum depth of the trees.
            seed (int): The random seed for reproducibility.

        Returns:
            BaselineResult: The result of fitting the baseline model.
        """

        keep: list[str] = list(filter(lambda c: c not in spec.protected, X.columns))
        classifier = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        classifier.fit(X[keep], y)
        return BaselineResult("protected_dropped", classifier.get_booster(), keep)

    @staticmethod
    def fit_monotone(
        X: pd.DataFrame,
        y: pd.Series,
        spec: Spec,
        n_estimators: int,
        max_depth: int,
        seed: int,
    ) -> BaselineResult:
        """
        Fit an XGBoost classifier with monotone constraints on protected features.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target labels.
            spec (Spec): The specification containing the protected features and their
                         monotone constraints.
            n_estimators (int): The number of trees to build.
            max_depth (int): The maximum depth of the trees.
            seed (int): The random seed for reproducibility.

        Returns:
            BaselineResult: The result of fitting the baseline model.
        """

        sign: dict[Direction, int] = {Direction.INCREASING: 1, Direction.DECREASING: -1}
        constraints: tuple[int, ...] = tuple(
            map(
                lambda c: sign[spec.monotone[c]] if c in spec.monotone else 0, X.columns
            )
        )
        classifier = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            monotone_constraints=constraints,
        )
        classifier.fit(X, y)
        return BaselineResult(
            "monotone_constraints", classifier.get_booster(), list(X.columns)
        )

    @staticmethod
    def label_p1_majority_protected_anchoring(
        pairs: list[Pair],
        columns: list[str],
        spec: Spec,
        X_train: pd.DataFrame,
        m0_booster: xgb.Booster,
        m0_leaf_map: LeafMap,
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64], int]]:
        """
        Anchor protected features to majority value in training data, then label x1/x2
        by m0's prediction on the anchored x1.

        Args:
            pairs (list[Pair]): The list of pairs to label.
            columns (list[str]): The list of column names.
            spec (Spec): The specification containing the protected features.
            X_train (pd.DataFrame): The training data.
            m0_booster (xgb.Booster): The trained XGBoost model.
            m0_leaf_map (LeafMap): The leaf map of the trained XGBoost model.

        Returns:
            list[tuple[NDArray[np.float64], NDArray[np.float64], int]]: A list of tuples
                                                                        containing the
                                                                        original x1, x2,
                                                                        and the assigned
                                                                        label.
        """

        majority: dict[str, Any] = {
            f: X_train[f].mode().iloc[0] for f in spec.protected
        }
        labeled: list[tuple[NDArray[np.float64], NDArray[np.float64], int]] = []

        for pair in pairs:
            anchored: NDArray[np.float64] = pair.x1.copy()

            for f in spec.protected:
                if f in columns:
                    anchored[columns.index(f)] = majority[f]

            prob: float = Baselines._predict_one(
                m0_booster, m0_leaf_map, anchored, columns
            )
            label = int(prob >= 0.5)
            labeled.append((pair.x1, pair.x2, label))

        return labeled

    @staticmethod
    def label_p2_score_averaging(
        pairs: list[Pair],
        columns: list[str],
        m0_booster: xgb.Booster,
        m0_leaf_map: LeafMap,
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64], int]]:
        """
        Label x1/x2 by averaging m0's predictions on x1 and x2, then rounding to nearest
        integer: round(sigmoid((E(x1) + E(x2)) / 2)) for both.

        Args:
            pairs (list[Pair]): The list of pairs to label.
            columns (list[str]): The list of column names.
            m0_booster (xgb.Booster): The trained XGBoost model.
            m0_leaf_map (LeafMap): The leaf map of the trained XGBoost model.

        Returns:
            list[tuple[NDArray[np.float64], NDArray[np.float64], int]]: A list of tuples
                                                                        containing the
                                                                        original x1, x2,
                                                                        and the assigned
                                                                        label.
        """

        labeled: list[tuple[NDArray[np.float64], NDArray[np.float64], int]] = []

        for pair in pairs:
            p1: float = Baselines._predict_one(
                m0_booster, m0_leaf_map, pair.x1, columns
            )
            p2: float = Baselines._predict_one(
                m0_booster, m0_leaf_map, pair.x2, columns
            )
            label = int(round((p1 + p2) / 2))
            labeled.append((pair.x1, pair.x2, label))

        return labeled

    @staticmethod
    def fit_counterexample_retrained(
        X: pd.DataFrame,
        y: pd.Series,
        pairs: list[Pair],
        columns: list[str],
        spec: Spec,
        m0_booster: xgb.Booster,
        m0_leaf_map: LeafMap,
        policy: str,
        n_estimators: int,
        max_depth: int,
        seed: int,
    ) -> BaselineResult:
        """
        Fit an XGBoost classifier after augmenting the training data with
        counterexamples labeled by the specified policy.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target values.
            pairs (list[Pair]): The list of pairs to use for counterexamples.
            columns (list[str]): The list of column names.
            spec (Spec): The specification for the model.
            m0_booster (xgb.Booster): The trained XGBoost model.
            m0_leaf_map (LeafMap): The leaf map of the trained XGBoost model.
            policy (str): The policy to use for labeling the counterexamples.
            n_estimators (int): The number of trees to use in the XGBoost classifier.
            max_depth (int): The maximum depth of the trees in the XGBoost classifier.
            seed (int): The random seed to use.

        Raises:
            ValueError: If an invalid policy is provided.

        Returns:
            BaselineResult: The result of fitting the baseline model with counterexample
                            retraining.
        """

        if policy == "P1":
            labeled: list[tuple[NDArray[np.float64], NDArray[np.float64], int]] = (
                Baselines.label_p1_majority_protected_anchoring(
                    pairs, columns, spec, X, m0_booster, m0_leaf_map
                )
            )
        elif policy == "P2":
            labeled = Baselines.label_p2_score_averaging(
                pairs, columns, m0_booster, m0_leaf_map
            )
        else:
            raise ValueError(f"policy must be 'P1' or 'P2', got {policy!r}")

        extra_rows = []
        extra_labels = []

        for x1, x2, label in labeled:
            extra_rows.append(x1)
            extra_rows.append(x2)
            extra_labels.append(label)
            extra_labels.append(label)

        if extra_rows:
            X_aug: pd.DataFrame = pd.concat(
                [X, pd.DataFrame(extra_rows, columns=columns)], ignore_index=True
            )
            y_aug: pd.Series = pd.concat(
                [y, pd.Series(extra_labels)], ignore_index=True
            )
        else:
            X_aug, y_aug = X, y

        classifier = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        classifier.fit(X_aug, y_aug)
        return BaselineResult(
            "counterexample_retrained", classifier.get_booster(), columns, policy=policy
        )

    @staticmethod
    def _predict_one(
        booster: xgb.Booster,
        leaf_map: LeafMap,
        row: NDArray[np.float64],
        columns: list[str],
    ) -> float:
        """
        Predict the probability for a single row using the provided booster and leaf
        map.

        Args:
            booster (xgb.Booster): The trained XGBoost model.
            leaf_map (LeafMap): The leaf map to use for prediction.
            row (NDArray[np.float64]): The row to predict.
            columns (list[str]): The column names for the row.

        Returns:
            float: The predicted probability.
        """

        df = pd.DataFrame([row], columns=columns)
        phi: csr_matrix = leaf_map.phi_for(booster, df)
        margin = float(leaf_map.margin_score(phi, leaf_map.v0)[0])
        return 1.0 / (1.0 + np.exp(-margin))

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from scipy.sparse._csr import csr_matrix

from sensei.model.leaves import LeafMap
from sensei.model.train import Trainer
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
    The four comparison models SensEI's repair is measured against: a plain
    model, one with protected features dropped, one with monotone
    constraints, and one retrained on Ensense counterexamples. `policy=` on
    baseline 4 has no default, so a caller can't pick a labeling policy by
    accident.
    """

    @staticmethod
    def fit_plain(
        X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, seed: int
    ) -> BaselineResult:
        """
        Baseline 1: a plain XGBoost classifier, no fairness intervention.
        Wraps `model/train.py::Trainer.train_baseline` (M0 itself) rather
        than retraining separately -- this is the model the other three
        baselines and CEGSAL's own repair are all trying to improve on.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels, `{0, 1}`.
            n_estimators (int): Number of trees.
            max_depth (int): Maximum depth per tree.
            seed (int): Training seed.

        Returns:
            BaselineResult: The trained plain model.
        """

        booster: xgb.Booster = Trainer.train_baseline(
            X, y, n_estimators, max_depth, seed
        )
        return BaselineResult("plain", booster, list(X.columns))

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
        Baseline 2: drop every `spec.protected` column, then train plain
        XGBoost on what's left ("fairness through unawareness" -- proxies
        for the dropped feature can still survive in the other columns).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels, `{0, 1}`.
            spec (Spec): Declares `protected`.
            n_estimators (int): Number of trees.
            max_depth (int): Maximum depth per tree.
            seed (int): Training seed.

        Returns:
            BaselineResult: The trained model, with `protected` columns dropped.
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
        Baseline 3: train with XGBoost's native `monotone_constraints`,
        forcing the model's response to each `spec.monotone` feature to move
        only in its declared direction (never enforced on protected features
        directly -- only on monotone ones).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels, `{0, 1}`.
            spec (Spec): Declares `monotone`.
            n_estimators (int): Number of trees.
            max_depth (int): Maximum depth per tree.
            seed (int): Training seed.

        Returns:
            BaselineResult: The trained model, with monotone constraints applied.
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
        Policy P1 for baseline 4: overwrite each pair's protected features
        with their most common (`.mode()`) value in `X_train`, predict with
        M0 on that anchored copy, threshold at 0.5, and give that single
        label to BOTH the original `x1` and `x2` (not the anchored copy --
        anchoring is only used to decide the label). Declared explicitly
        because there's no ground truth for a solver-generated pair, and
        assigning one label to both members is itself a fairness assumption.

        Args:
            pairs (list[Pair]): Counterexample pairs to label.
            columns (list[str]): All feature names, in model column order.
            spec (Spec): Declares `protected`.
            X_train (pd.DataFrame): Training data, for the majority-value anchor.
            m0_booster (xgb.Booster): M0, scored to derive the label.
            m0_leaf_map (LeafMap): Global leaf map for `m0_booster`.

        Returns:
            list[tuple[NDArray[np.float64], NDArray[np.float64], int]]: Each
                pair's `(x1, x2, label)`.
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
        Policy P2 for baseline 4, the robustness check against P1's
        majority-anchoring assumption: label both x1 and x2 with
        round(sigmoid((E(x1) + E(x2)) / 2)) -- split the difference between
        what M0 said about each side, no anchoring.

        Args:
            pairs (list[Pair]): Counterexample pairs to label.
            columns (list[str]): All feature names, in model column order.
            m0_booster (xgb.Booster): M0, scored to derive the label.
            m0_leaf_map (LeafMap): Global leaf map for `m0_booster`.

        Returns:
            list[tuple[NDArray[np.float64], NDArray[np.float64], int]]: Each
                pair's `(x1, x2, label)`.
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
        Baseline 4, the one SensEI has to beat: label every counterexample
        pair with `policy` ("P1" or "P2", required -- no default), add both
        `x1` and `x2` to the training data with that label, and retrain
        plain XGBoost on the result.

        Args:
            X (pd.DataFrame): Original training features.
            y (pd.Series): Original training labels, `{0, 1}`.
            pairs (list[Pair]): Counterexample pairs to add to training data.
            columns (list[str]): All feature names, in model column order.
            spec (Spec): Declares `protected`, for policy P1's anchoring.
            m0_booster (xgb.Booster): M0, scored to derive each pair's label.
            m0_leaf_map (LeafMap): Global leaf map for `m0_booster`.
            policy (str): `"P1"` or `"P2"`, required -- no default.
            n_estimators (int): Number of trees.
            max_depth (int): Maximum depth per tree.
            seed (int): Training seed.

        Returns:
            BaselineResult: The retrained model, labeled with `policy`.

        Raises:
            ValueError: `policy` is neither "P1" nor "P2".
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
        Sigmoid of M0's margin on a single raw row -- always scores against
        the original `v0`, never a repaired model (baselines are a
        comparison point, not something CEGSAL touches).

        Args:
            booster (xgb.Booster): The model to score.
            leaf_map (LeafMap): Global leaf map for `booster`.
            row (NDArray[np.float64]): A single raw feature row.
            columns (list[str]): All feature names, in model column order.

        Returns:
            float: The predicted probability, in `[0, 1]`.
        """

        df = pd.DataFrame([row], columns=columns)
        phi: csr_matrix = leaf_map.phi_for(booster, df)
        margin = float(leaf_map.margin_score(phi, leaf_map.v0)[0])
        return 1.0 / (1.0 + np.exp(-margin))

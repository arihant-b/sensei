import pandas as pd
import xgboost as xgb


class Trainer:
    """Trains M0, the one XGBoost model everything else in this project starts from."""

    @staticmethod
    def train_baseline(
        X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, seed: int
    ) -> xgb.Booster:
        """
        Train a plain XGBoost classifier. This is the ONLY place `xgb.fit()`
        (tree structure changing) is allowed to run -- everything after this
        call treats the tree structure as frozen and only ever rewrites leaf
        values (see `model/leaves.py::LeafMap.write_leaf_values`).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels, `{0, 1}`.
            n_estimators (int): Number of trees.
            max_depth (int): Maximum depth per tree.
            seed (int): Training seed.

        Returns:
            xgb.Booster: The trained model, M0.
        """

        classifier = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        classifier.fit(X, y)
        return classifier.get_booster()

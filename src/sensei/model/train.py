import pandas as pd
import xgboost as xgb


class Trainer:
    """
    A class for training XGBoost models.
    """

    @staticmethod
    def train_baseline(
        X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, seed: int
    ) -> xgb.Booster:
        """
        Train a baseline XGBoost model on the given features and labels, and return the
        trained booster. The structure of the booster is frozen immediately after
        training.

        Args:
            X (pd.DataFrame): The input features for training.
            y (pd.Series): The target labels for training.
            n_estimators (int): The number of trees to train.
            max_depth (int): The maximum depth of each tree.
            seed (int): The random seed for reproducibility.

        Returns:
            xgb.Booster: The trained XGBoost booster with frozen structure.
        """

        classifier = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
        classifier.fit(X, y)
        return classifier.get_booster()

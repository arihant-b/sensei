import hashlib

import pandas as pd
import xgboost as xgb

_STRUCTURAL_COLUMNS: list[str] = ["Tree", "Node", "ID", "Feature", "Split", "Yes", "No"]


class FrozenStructure:
    """
    Guard that asserts that the tree structure of a booster is frozen after initial
    training.
    """

    def __init__(self, booster: xgb.Booster) -> None:
        self.fingerprint: str = self._structure_fingerprint(booster)

    def assert_unchanged(self, booster: xgb.Booster) -> None:
        """
        Assert that the tree structure of the booster has not changed since the
        creation of this FrozenStructure instance.

        Args:
            booster (xgb.Booster): The XGBoost booster to check.
        """

        current: str = self._structure_fingerprint(booster)
        assert current == self.fingerprint, (
            "tree structure changed after freezing -- violation. This means "
            "xgb.train() (or equivalent) ran somewhere it shouldn't have."
            f"expected {self.fingerprint[:12]}..., got {current[:12]}..."
        )

    def _structure_fingerprint(self, booster: xgb.Booster) -> str:
        """
        Compute a fingerprint of the tree structure of the booster.

        Args:
            booster (xgb.Booster): The XGBoost booster to compute the fingerprint for.

        Returns:
            str: The fingerprint of the tree structure.
        """

        df: pd.DataFrame = booster.trees_to_dataframe()[_STRUCTURAL_COLUMNS]
        canonical: str = df.to_csv(index=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

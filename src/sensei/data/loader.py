import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

_DATASET_ROOT: Path = Path(__file__).resolve().parents[3] / "dataset"

log: logging.Logger = logging.getLogger("sensei.data.loader")


class EvalAccessError(RuntimeError):
    """D_eval was accessed more than once in this Dataset instance's lifetime."""


class Dataset:
    """
    D_train (80% of train.csv) trains the model and builds all validity/bins machinery.
    D_eval (the held-out 20%) is untouched until final reporting.
    """

    def __init__(self, name: str, eval_holdout: float, seed: int) -> None:
        self.name = name
        self.eval_holdout = eval_holdout
        self.seed = seed

        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None
        self.columns: list[str] | None = None
        self.feature_bounds: dict[str, tuple[float, float]] | None = None

        self._X_eval: pd.DataFrame | None = None
        self._y_eval: pd.Series | None = None
        self._eval_access_count = 0

    def load(self) -> "Dataset":
        """
        Load the dataset from disk, splitting train.csv into D_train and D_eval.

        Returns:
            Dataset: The Dataset instance with loaded data.
        """

        root: Path = _DATASET_ROOT / self.name
        train_df: pd.DataFrame = pd.read_csv(root / "train.csv")
        test_df: pd.DataFrame = pd.read_csv(root / "test.csv")

        X_train_full, y_train_full, self.columns = self._split_features_label(train_df)
        X_test, y_test, _ = self._split_features_label(test_df, columns=self.columns)

        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train_full,
            y_train_full,
            test_size=self.eval_holdout,
            random_state=self.seed,
            stratify=y_train_full,
        )

        self.X_train = X_train.reset_index(drop=True)
        self._X_eval = X_eval.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self._y_eval = y_eval.reset_index(drop=True)
        self.y_test = y_test
        self.feature_bounds = self._load_feature_bounds(root)
        return self

    @property
    def X_eval(self) -> pd.DataFrame:
        """
        Access the evaluation features. Raises EvalAccessError if accessed more than
        once.

        Returns:
            pd.DataFrame: The evaluation features.
        """

        eval: pd.DataFrame | pd.Series = self._eval_access("X_eval", self._X_eval)
        assert isinstance(eval, pd.DataFrame)
        return eval

    @property
    def y_eval(self) -> pd.Series:
        """
        Access the evaluation labels. Raises EvalAccessError if accessed more than
        once.

        Returns:
            pd.Series: The evaluation labels.
        """

        eval: pd.DataFrame | pd.Series = self._eval_access("y_eval", self._y_eval)
        assert isinstance(eval, pd.Series)
        return eval

    def _eval_access(
        self, field_name: str, value: pd.DataFrame | pd.Series | None
    ) -> pd.DataFrame | pd.Series:
        """
        Access the evaluation data. Raises EvalAccessError if accessed more than once.

        Args:
            field_name (str): The name of the field being accessed (X_eval or y_eval).
            value (pd.DataFrame | pd.Series | None): The value of the field being
                                                     accessed. If None, raises an
                                                     assertion error.

        Raises:
            EvalAccessError: If the evaluation data has been accessed more than once.

        Returns:
            pd.DataFrame | pd.Series: The value of the field being accessed.
        """

        assert value is not None, "call load() before accessing D_eval"

        self._eval_access_count += 1
        log.info(
            "D_eval access #%d on dataset=%s field=%s",
            self._eval_access_count,
            self.name,
            field_name,
        )

        if self._eval_access_count > 2:  # X_eval + y_eval together count as one read
            raise EvalAccessError(
                f"D_eval accessed {self._eval_access_count} times for dataset="
                f"{self.name} -- allows one access per experiment "
                f"(X_eval + y_eval together). This is a leak, not a warning."
            )

        return value

    def _load_feature_bounds(self, root: Path) -> dict[str, tuple[float, float]]:
        """
        Load the feature bounds from a saved scaler.

        Args:
            root (Path): The root path of the dataset.

        Returns:
            dict[str, tuple[float, float]]: A dictionary mapping feature names to their
                                            (min, max) bounds.
        """

        scaler_path: Path = root / "scaler.pkl"

        if not scaler_path.exists():
            return {}

        scaler: MinMaxScaler = joblib.load(scaler_path)
        return {
            f: (float(lo), float(hi))
            for f, lo, hi in zip(
                scaler.feature_names_in_,
                scaler.data_min_,
                scaler.data_max_,
                strict=True,
            )
        }

    def _split_features_label(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        Split the features and label from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to split.
            columns (list[str] | None, optional): The columns to include in the
                                                  features. If None, all columns except
                                                  the last one are included. Defaults to
                                                  None.

        Returns:
            tuple[pd.DataFrame, pd.Series, list[str]]: A tuple containing the features
                                                       DataFrame, the label Series, and
                                                       the list of feature names.
        """

        label_col: str = df.columns[-1]

        if columns is None:
            columns = list(filter(lambda c: c != label_col, df.columns))

        return df[columns], df[label_col], columns

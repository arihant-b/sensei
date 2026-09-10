import json
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
    Splits train.csv into D_train (trains the model, builds validity/bins
    machinery) and D_eval (the `eval_holdout` fraction held out, untouched
    until final reporting). test.csv loads separately as X_test/y_test.
    """

    def __init__(self, name: str, eval_holdout: float, seed: int) -> None:
        self.name: str = name
        self.eval_holdout: float = eval_holdout
        self.seed: int = seed

        self.X_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None
        self.columns: list[str] | None = None
        self.feature_bounds: dict[str, tuple[float, float]] | None = None
        self.categorical_levels: dict[str, int] | None = None

        self._X_eval: pd.DataFrame | None = None
        self._y_eval: pd.Series | None = None
        self._eval_access_count = 0

    def load(self) -> "Dataset":
        """
        Read train.csv/test.csv off disk and split train.csv further into
        D_train and the held-out D_eval.

        Returns:
            Dataset: `self`, loaded -- chains with the constructor:
                `Dataset(name, holdout, seed).load()`.
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
        self.categorical_levels = self._load_categorical_levels(
            root, self.feature_bounds
        )
        return self

    @property
    def X_eval(self) -> pd.DataFrame:
        """
        D_eval's features.

        Returns:
            pd.DataFrame: D_eval's features.

        Raises:
            EvalAccessError: This is more than the one allowed D_eval access.
        """

        eval: pd.DataFrame | pd.Series = self._eval_access("X_eval", self._X_eval)
        assert isinstance(eval, pd.DataFrame)
        return eval

    @property
    def y_eval(self) -> pd.Series:
        """
        D_eval's labels.

        Returns:
            pd.Series: D_eval's labels.

        Raises:
            EvalAccessError: This is more than the one allowed D_eval access.
        """

        eval: pd.DataFrame | pd.Series = self._eval_access("y_eval", self._y_eval)
        assert isinstance(eval, pd.Series)
        return eval

    def _eval_access(
        self, field_name: str, value: pd.DataFrame | pd.Series | None
    ) -> pd.DataFrame | pd.Series:
        """
        Log and count one D_eval read, raising EvalAccessError past the
        first (X_eval + y_eval together count as one access, since a real
        run needs both).

        Args:
            field_name (str): Which field is being accessed, for logging.
            value (pd.DataFrame | pd.Series | None): The cached D_eval value.

        Returns:
            pd.DataFrame | pd.Series: `value`, unchanged.

        Raises:
            EvalAccessError: This is more than the one allowed D_eval access.
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
        (min, max) raw-value bounds per feature, read off the fitted scaler.pkl.

        Args:
            root (Path): The dataset's directory, containing `scaler.pkl`.

        Returns:
            dict[str, tuple[float, float]]: Raw `(lo, hi)` bounds per
                feature, or `{}` if no scaler was found.
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

    def _load_categorical_levels(
        self, root: Path, feature_bounds: dict[str, tuple[float, float]]
    ) -> dict[str, int]:
        """
        Number of distinct raw values per categorical feature: its
        encoding_map.json category count, plus one if the column also has a
        missing-value sentinel. Ensense core's own encoding (data/builder.py
        matches it) writes a missing categorical value as raw code -1
        instead of dropping the row, so e.g. `workclass` really has
        `len(encoding_map["workclass"])` real categories PLUS that sentinel.
        `feature_bounds[feature][0] < 0` detects the sentinel (a real
        ordinal code is never negative). This count feeds
        `FrozenBins._categorical_edges`, which lays bin edges evenly across
        the column's scaled range -- undercounting by one misaligns every
        bin for that column.

        Args:
            root (Path): The dataset's directory, containing `encoding_map.json`.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature, to detect the missing-value sentinel.

        Returns:
            dict[str, int]: Distinct raw value count per categorical
                feature, or `{}` if no encoding map was found.
        """

        encoding_map_path: Path = root / "encoding_map.json"

        if not encoding_map_path.exists():
            return {}

        raw_map: dict[str, list[str]] = json.loads(encoding_map_path.read_text())
        levels: dict[str, int] = {}

        for feature, categories in raw_map.items():
            if self.columns is None or feature not in self.columns:
                continue

            has_missing_sentinel: bool = feature_bounds.get(feature, (0.0, 0.0))[0] < 0
            levels[feature] = len(categories) + (1 if has_missing_sentinel else 0)

        return levels

    def _split_features_label(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        Split (features, label, feature names) out of `df`. The label is
        always `df`'s last column; `columns` lets a caller reuse train's own
        feature list for test, instead of re-deriving "everything but the
        last column" from test's (possibly differently-ordered) columns.

        Args:
            df (pd.DataFrame): The DataFrame to split.
            columns (list[str] | None): Feature names to use; derived from
                `df` (everything but the last column) if None.

        Returns:
            tuple[pd.DataFrame, pd.Series, list[str]]: `(X, y, columns)`.
        """

        label_col: str = df.columns[-1]

        if columns is None:
            columns = list(filter(lambda c: c != label_col, df.columns))

        return df[columns], df[label_col], columns

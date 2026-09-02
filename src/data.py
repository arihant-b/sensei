from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from config import Config

_DATASET_ROOT: Path = Path(__file__).resolve().parents[1].joinpath("dataset")

# Guards log(0) on an empty bin. Bins are built from quantiles of the same
# data they cover, so an empty bin should not occur in practice -- this only
# protects against pathological columns (e.g. near-constant after scaling).
_MIN_BIN_MASS = 1e-12


def _split_features_label(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, NDArray[np.int64], list[str]]:
    """
    train.csv/test.csv are pre-encoded (see encoding_map.json, scaler.pkl)
    but not perfectly consistent about the target column's name across
    files (e.g. adult's test.csv calls it 'income' where train.csv calls
    it 'label'). Normalise on 'label', falling back to the last column.
    """
    label_col: str = "label" if "label" in df.columns else df.columns[-1]

    if columns is None:
        columns = [c for c in df.columns if c != label_col]

    return df[columns], df[label_col].to_numpy(), columns


class FrozenBins:
    """
    Quantile bins built ONCE from the training data.

    Deliberately NOT built from the model's guard thresholds: those move
    every time leaf values change the effective model, which would make
    'realistic' a moving target across loop iterations.
    """

    def __init__(self, n_bins: int) -> None:
        self.n_bins = n_bins
        self.edges: dict[str, NDArray[np.float64]] = {}   # feature -> bin edges
        self.log_p: dict[str, NDArray[np.float64]] = {}   # feature -> log mass per bin

    def fit(self, X: pd.DataFrame, columns: list[str]) -> "FrozenBins":
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)

        for col in columns:
            values = np.asarray(X[col], dtype=float)
            edges = np.unique(np.quantile(values, quantiles))

            if edges.size < 2:
                # constant column: fabricate a single bin around the value
                edges = np.array([edges[0] - 0.5, edges[0] + 0.5])

            counts, _ = np.histogram(values, bins=edges)
            mass = counts / counts.sum()
            mass = np.clip(mass, _MIN_BIN_MASS, None)

            self.edges[col] = edges
            self.log_p[col] = np.log(mass)

        return self

    def bin_of(self, feature: str, value: float) -> int:
        edges = self.edges[feature]
        # interior edges only, so digitize output already indexes
        # directly into log_p[feature] (length len(edges) - 1)
        return int(np.digitize(value, edges[1:-1]))

    def log_marginal(self, x) -> float:
        """sum_f log pi_f(x_f)  -- linear once bins are fixed."""
        total = 0.0

        for feature in self.edges:
            idx: int = self.bin_of(feature, x[feature])
            total += float(self.log_p[feature][idx])

        return total


class Dataset:
    def __init__(self, cfg: Config) -> None:
        self.cfg: Config = cfg
        self.X_train: pd.DataFrame | None = None
        self.y_train: NDArray[np.int64] | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_test: NDArray[np.int64] | None = None
        self.X_eval: pd.DataFrame | None = None         # held out from EVERYTHING
        self.y_eval: NDArray[np.int64] | None = None
        self.columns: list[str] | None = None
        # feature -> (raw_min, raw_max) from the dataset's own scaler.pkl,
        # i.e. the pre-scaling values -- lets validity.py check things
        # like integrality, which only make sense in raw units, against
        # data that is actually min-max scaled. {} when no scaler.pkl
        # exists for this dataset.
        self.feature_bounds: dict[str, tuple[float, float]] | None = None

    def load(self) -> "Dataset":
        """
        train.csv / test.csv on disk are already encoded (categoricals
        ordinal-mapped via encoding_map.json, numerics scaled via
        scaler.pkl) so there is no re-encoding to do here -- just read,
        then carve X_eval out of the training pool. X_test stays exactly
        the file's own held-out split; X_eval is never touched again
        after this point.
        """
        root: Path = _DATASET_ROOT.joinpath(self.cfg.dataset)
        train_df: pd.DataFrame = pd.read_csv(root.joinpath("train.csv"))
        test_df: pd.DataFrame = pd.read_csv(root.joinpath("test.csv"))

        X_train_full, y_train_full, self.columns = _split_features_label(train_df)
        X_test, y_test, _ = _split_features_label(test_df, columns=self.columns)

        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(
            X_train_full,
            y_train_full,
            test_size=self.cfg.eval_holdout,
            random_state=self.cfg.seed,
            stratify=y_train_full,
        )
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_eval = self.X_eval.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_test = y_test
        self.feature_bounds = self._load_feature_bounds(root)
        return self

    def _load_feature_bounds(self, root: Path) -> dict[str, tuple[float, float]]:
        scaler_path: Path = root.joinpath("scaler.pkl")

        if not scaler_path.exists():
            return {}

        scaler = joblib.load(scaler_path)
        names = list(scaler.feature_names_in_)
        return {
            f: (float(lo), float(hi))
            for f, lo, hi in zip(names, scaler.data_min_, scaler.data_max_, strict=True)
        }

    def build_bins(self) -> FrozenBins:
        assert self.X_train is not None and self.columns is not None, (
            "call load() before build_bins()"
        )
        return FrozenBins(self.cfg.n_quantile_bins).fit(self.X_train, self.columns)

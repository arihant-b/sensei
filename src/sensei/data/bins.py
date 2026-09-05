import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_RESULTS_DIR: Path = Path(__file__).resolve().parents[3] / "results"


@dataclass(frozen=True)
class Point:
    """
    A single row of a dataset, represented as a mapping from feature name to
    value. The values are always floats, even for categorical features.
    """

    values: Mapping[str, Any]

    def __getitem__(self, feature: str) -> Any:
        return self.values[feature]

    def __contains__(self, feature: str) -> bool:
        return feature in self.values


class FrozenBins:
    """
    Quantile bins built from a training dataset, cached to results/bins_<dataset>.json,
    and asserted on every later load. A mismatch invalidates the run rather than
    silently rebinning.
    """

    def __init__(self, n_bins: int) -> None:
        self.n_bins: int = n_bins
        self.edges: dict[str, NDArray[np.float64]] = {}
        self.log_p: dict[str, NDArray[np.float64]] = {}  # Laplace-smoothed

    def fit(self, X: pd.DataFrame, columns: list[str]) -> "FrozenBins":
        """
        Fit the frozen bins to the training data.

        Args:
            X (pd.DataFrame): The training data to fit the bins to.
            columns (list[str]): The columns to fit the bins to.

        Returns:
            FrozenBins: The fitted FrozenBins object.
        """

        quantiles: NDArray[np.float64] = np.linspace(0.0, 1.0, self.n_bins + 1)

        for col in columns:
            values: NDArray[Any] = np.asarray(X[col], dtype=float)
            edges: NDArray[Any] = np.unique(np.quantile(values, quantiles))

            if edges.size < 2:
                # if only one value, we create a single bin around that value.
                edges = np.array([edges[0] - 0.5, edges[0] + 0.5])

            counts, _ = np.histogram(values, bins=edges)
            n_bins: int = len(counts)
            # Laplace smoothing: p = (count + 1) / (N + n_bins).
            # Zeros make logs infeasible for reasons unrelated to sensitivity.
            mass = (counts + 1) / (counts.sum() + n_bins)

            self.edges[col] = edges
            self.log_p[col] = np.log(mass)

        return self

    def bin_of(self, feature: str, value: float) -> int:
        """
        Get the bin index for a given feature and value.

        Args:
            feature (str): The feature for which to get the bin index.
            value (float): The value for which to get the bin index.

        Returns:
            int: The bin index.
        """

        edges = self.edges[feature]
        return int(np.digitize(value, edges[1:-1]))

    def log_plaus(self, x: Point) -> float:
        """
        Compute the log plausibility of a point.
        log plaus(x) = sum_f log p_f(x_f). This is not a density.

        Args:
            x (Point): The point for which to compute the log plausibility.

        Returns:
            float: The log plausibility of the point.
        """

        total = 0.0

        for feature in self.edges:
            total += float(self.log_p[feature][self.bin_of(feature, x[feature])])

        return total

    def content_hash(self) -> str:
        """
        Compute a content hash of the FrozenBins object. This is used to ensure
        that the cached bins are consistent with the current state of the object.

        Returns:
            str: The content hash of the FrozenBins object.
        """

        payload: dict[str, Any] = {
            "n_bins": self.n_bins,
            "edges": {k: v.tolist() for k, v in sorted(self.edges.items())},
            "log_p": {k: v.tolist() for k, v in sorted(self.log_p.items())},
        }
        canonical: str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, dataset: str) -> str:
        """
        Save the FrozenBins object to a results/bins_<dataset>.json file.

        Args:
            dataset (str): The dataset for which to save the bins.

        Returns:
            str: The content hash of the saved bins.
        """

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        content_hash: str = self.content_hash()
        payload: dict[str, Any] = {
            "hash": content_hash,
            "n_bins": self.n_bins,
            "edges": {k: v.tolist() for k, v in self.edges.items()},
            "log_p": {k: v.tolist() for k, v in self.log_p.items()},
        }
        (_RESULTS_DIR / f"bins_{dataset}.json").write_text(
            json.dumps(payload, indent=2)
        )
        return content_hash

    @classmethod
    def load(cls, dataset: str) -> "FrozenBins":
        """
        Load the FrozenBins object from a results/bins_<dataset>.json file. The
        content hash is checked against the current state of the object to ensure
        that the cached bins are consistent with the current state of the object.

        Args:
            dataset (str): The dataset for which to load the bins.

        Returns:
            FrozenBins: The loaded FrozenBins object.
        """

        path: Path = _RESULTS_DIR / f"bins_{dataset}.json"
        payload: dict[str, Any] = json.loads(path.read_text())

        bins: FrozenBins = cls(payload["n_bins"])
        bins.edges = {k: np.array(v, dtype=float) for k, v in payload["edges"].items()}
        bins.log_p = {k: np.array(v, dtype=float) for k, v in payload["log_p"].items()}

        actual_hash: str = bins.content_hash()
        assert actual_hash == payload["hash"], (
            f"bins_{dataset}.json hash mismatch: stored {payload['hash'][:12]}... != "
            f"recomputed {actual_hash[:12]}... -- cache is stale or was edited by hand"
        )
        return bins

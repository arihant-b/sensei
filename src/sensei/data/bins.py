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
class EncodedSample:
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

    def fit(
        self,
        X: pd.DataFrame,
        columns: list[str],
        categorical_levels: dict[str, int] | None = None,
    ) -> "FrozenBins":
        """
        Fit one marginal per column in `columns`: quantile bins for a
        numeric feature, one bin per raw level for a categorical one (given
        by `categorical_levels`, feature name -> level count). Each bin's
        probability is Laplace-smoothed, `p = (count + 1) / (N + n_bins)`,
        so an empty bin gets a small nonzero probability instead of a zero
        that would make `log(p)` infeasible.

        Args:
            X (pd.DataFrame): Training data to fit marginals from.
            columns (list[str]): Feature names to fit bins for.
            categorical_levels (dict[str, int] | None): Feature name -> raw
                level count, for categorical features.

        Returns:
            FrozenBins: `self`, fitted.
        """

        categorical_levels = categorical_levels or {}
        quantiles: NDArray[np.float64] = np.linspace(0.0, 1.0, self.n_bins + 1)

        for col in columns:
            values: NDArray[Any] = np.asarray(X[col], dtype=float)

            if col in categorical_levels:
                edges: NDArray[Any] = FrozenBins._categorical_edges(
                    categorical_levels[col]
                )
            else:
                edges = np.unique(np.quantile(values, quantiles))

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
        Which of `feature`'s fitted bins `value` falls into.

        Args:
            feature (str): The feature to look up bins for.
            value (float): The value to place into a bin.

        Returns:
            int: The bin index.
        """

        edges = self.edges[feature]
        return int(np.digitize(value, edges[1:-1]))

    def log_plaus(self, x: EncodedSample) -> float:
        """
        log_plaus(x) = sum_f log p_f(x_f) -- a product of independent
        per-feature marginals in log space. This is a plausibility SCORE,
        not a log-density: it assumes features are independent, so it can
        rate a combination as plausible even when the joint combination
        (e.g. age 25 with a PhD) never occurs together in the data.

        Args:
            x (EncodedSample): The point to score.

        Returns:
            float: The log-plausibility score.
        """

        total = 0.0

        for feature in self.edges:
            total += float(self.log_p[feature][self.bin_of(feature, x[feature])])

        return total

    def content_hash(self) -> str:
        """
        SHA256 over n_bins/edges/log_p, to detect a stale or hand-edited cache.

        Returns:
            str: The hex-encoded SHA256 digest.
        """

        payload: dict[str, Any] = {
            "n_bins": self.n_bins,
            "edges": {k: v.tolist() for k, v in sorted(self.edges.items())},
            "log_p": {k: v.tolist() for k, v in sorted(self.log_p.items())},
        }
        canonical: str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, dataset: str, seed: int) -> str:
        """
        Write this to its (dataset, seed, n_bins) cache file; returns its hash.

        Args:
            dataset (str): Dataset name, part of the cache key.
            seed (int): Seed, part of the cache key.

        Returns:
            str: The content hash written alongside the cache.
        """

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        content_hash: str = self.content_hash()
        payload: dict[str, Any] = {
            "hash": content_hash,
            "n_bins": self.n_bins,
            "edges": {k: v.tolist() for k, v in self.edges.items()},
            "log_p": {k: v.tolist() for k, v in self.log_p.items()},
        }
        FrozenBins._cache_path(dataset, seed, self.n_bins).write_text(
            json.dumps(payload, indent=2)
        )
        return content_hash

    @classmethod
    def load(cls, dataset: str, seed: int, n_bins: int) -> "FrozenBins":
        """
        Read the (dataset, seed, n_bins) cache file and verify its stored
        hash still matches its own content before returning it.

        Args:
            dataset (str): Dataset name, part of the cache key.
            seed (int): Seed, part of the cache key.
            n_bins (int): Quantile bin count, part of the cache key.

        Returns:
            FrozenBins: The loaded, verified bins.

        Raises:
            AssertionError: The cache's recomputed hash doesn't match the
                stored one -- it's stale or was edited by hand.
        """

        path: Path = FrozenBins._cache_path(dataset, seed, n_bins)
        payload: dict[str, Any] = json.loads(path.read_text())

        bins: FrozenBins = cls(payload["n_bins"])
        bins.edges = {k: np.array(v, dtype=float) for k, v in payload["edges"].items()}
        bins.log_p = {k: np.array(v, dtype=float) for k, v in payload["log_p"].items()}

        actual_hash: str = bins.content_hash()
        assert actual_hash == payload["hash"], (
            f"{path.name} hash mismatch: stored {payload['hash'][:12]}... != "
            f"recomputed {actual_hash[:12]}... -- cache is stale or was edited by hand"
        )
        return bins

    @classmethod
    def fit_or_load(
        cls,
        dataset: str,
        seed: int,
        n_bins: int,
        X: pd.DataFrame,
        columns: list[str],
        categorical_levels: dict[str, int] | None = None,
    ) -> "FrozenBins":
        """
        Fit bins fresh from `X`, then either write them as the first cache
        for this (dataset, seed, n_bins), or verify they match an existing
        cache exactly. Always returns the freshly-fit bins, never the cached
        object -- when a cache exists the two are numerically identical
        anyway, so this never silently returns stale bins.

        Args:
            dataset (str): Dataset name, part of the cache key.
            seed (int): Seed, part of the cache key.
            n_bins (int): Quantile bin count, part of the cache key.
            X (pd.DataFrame): Training data to fit marginals from.
            columns (list[str]): Feature names to fit bins for.
            categorical_levels (dict[str, int] | None): Feature name -> raw
                level count, for categorical features.

        Returns:
            FrozenBins: The freshly-fit bins.

        Raises:
            ValueError: A cache exists but disagrees with the freshly-fit
                bins -- train.csv or encoding_map.json changed on disk since
                the cache was written. Not resolved automatically; delete
                the cache file to accept the new bins.
        """

        fresh: FrozenBins = cls(n_bins).fit(X, columns, categorical_levels)
        path: Path = cls._cache_path(dataset, seed, n_bins)

        if not path.exists():
            fresh.save(dataset, seed)
            return fresh

        cached: FrozenBins = cls.load(dataset, seed, n_bins)

        if cached.content_hash() != fresh.content_hash():
            raise ValueError(
                f"{path.name} exists but does not match bins freshly fit for "
                f"dataset={dataset!r} seed={seed} n_bins={n_bins} -- the cache "
                f"is already keyed by every setting that should legitimately "
                f"change the bins, so this means dataset/{dataset}/train.csv "
                f"or encoding_map.json changed on disk since the cache was "
                f"written. Delete {path.name} to accept the new bins; this is "
                f"not done automatically."
            )

        return fresh

    @staticmethod
    def _cache_path(dataset: str, seed: int, n_bins: int) -> Path:
        """
        The cache file for one (dataset, seed, n_bins) combination.

        Args:
            dataset (str): Dataset name.
            seed (int): Seed.
            n_bins (int): Quantile bin count.

        Returns:
            Path: The cache file path.
        """

        return _RESULTS_DIR / f"bins_{dataset}_seed{seed}_n{n_bins}.json"

    @staticmethod
    def _categorical_edges(n_levels: int) -> NDArray[np.float64]:
        """
        `n_levels + 1` bin edges, one per raw code, evenly spaced across the
        column's scaled range and centered on each code's own scaled
        position -- so code `i`'s bin is `[midpoint(i-1, i), midpoint(i,
        i+1))`, not an arbitrary quantile split.

        Args:
            n_levels (int): Number of raw categorical levels.

        Returns:
            NDArray[np.float64]: `n_levels + 1` bin edges.
        """

        if n_levels <= 1:
            return np.array([-0.5, 0.5])

        positions: NDArray[np.float64] = np.linspace(0.0, 1.0, n_levels)
        half_step: float = (positions[1] - positions[0]) / 2
        return np.concatenate(
            [
                [positions[0] - half_step],
                (positions[:-1] + positions[1:]) / 2,
                [positions[-1] + half_step],
            ]
        )

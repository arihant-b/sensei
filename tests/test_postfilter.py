import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.data.bins import FrozenBins  # noqa: E402
from sensei.data.loader import Dataset  # noqa: E402
from sensei.oracle.types import OracleSaturated, Pair  # noqa: E402
from sensei.spec import Spec, load_spec  # noqa: E402
from sensei.validity.postfilter import MAX_REJECTIONS_PER_ITER, Postfilter  # noqa: E402


def _fixture() -> tuple[Dataset, Spec, FrozenBins]:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.X_train is not None and ds.columns is not None

    spec: Spec = load_spec("adult")
    bins: FrozenBins = FrozenBins(10).fit(ds.X_train, ds.columns)
    return ds, spec, bins


def _make_pair(values: dict, columns: list[str]) -> Pair:
    x: NDArray[np.float64] = np.array(
        [values.get(c, 0.5) for c in columns], dtype=float
    )
    return Pair(
        x1=x,
        x2=x.copy(),
        ell1=np.array([0]),
        ell2=np.array([1]),
        gap=1.0,
        flip_set=("sex",),
        direction="protected",
        solver_status="FEASIBLE",
    )


def test_oracle_returning_none_means_insensitive() -> None:
    ds, spec, bins = _fixture()

    assert ds.columns is not None and ds.feature_bounds is not None

    result = Postfilter.find_valid_pair(
        lambda: None,
        ds.columns,
        spec,
        bins,
        theta=1e-6,
        feature_bounds=ds.feature_bounds,
    )
    assert result is None


def test_a_valid_pair_is_returned_immediately() -> None:
    ds, spec, bins = _fixture()

    assert ds.columns is not None and ds.feature_bounds is not None

    row: pd.Series = ds.X_train.iloc[0]
    values: dict[str, float] = dict(row)
    pair: Pair = _make_pair(values, ds.columns)

    result: Pair | None = Postfilter.find_valid_pair(
        lambda: pair,
        ds.columns,
        spec,
        bins,
        theta=1e-9,
        feature_bounds=ds.feature_bounds,
    )
    assert result is pair


def test_a_permanently_invalid_pair_exhausts_the_rejection_budget() -> None:
    ds, spec, bins = _fixture()

    assert ds.columns is not None and ds.feature_bounds is not None

    bad_values: dict[str, float] = {c: -5.0 for c in ds.columns}
    bad_pair: Pair = _make_pair(bad_values, ds.columns)

    with pytest.raises(OracleSaturated):
        Postfilter.find_valid_pair(
            lambda: bad_pair,
            ds.columns,
            spec,
            bins,
            theta=0.5,
            feature_bounds=ds.feature_bounds,
            max_rejections=5,
        )


def test_default_rejection_budget_matches_module_constant() -> None:
    assert MAX_REJECTIONS_PER_ITER == 20

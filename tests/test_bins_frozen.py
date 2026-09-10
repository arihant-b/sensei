import sys
from pathlib import Path

import pandas as pd

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.data.bins import EncodedSample, FrozenBins  # noqa: E402
from sensei.data.loader import Dataset  # noqa: E402

_TEST_DATASET = "adult_test_bins_frozen"
_TEST_SEED = 42
_TEST_N_BINS = 10


def _cache_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"bins_{_TEST_DATASET}_seed{_TEST_SEED}_n{_TEST_N_BINS}.json"
    )


def test_bins_hash_stable_across_save_and_load() -> None:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=_TEST_SEED).load()

    assert ds.X_train is not None and ds.columns is not None
    assert ds.X_test is not None

    bins: FrozenBins = FrozenBins(_TEST_N_BINS).fit(ds.X_train, ds.columns)

    saved_hash: str = bins.save(_TEST_DATASET, _TEST_SEED)
    reloaded: FrozenBins = FrozenBins.load(_TEST_DATASET, _TEST_SEED, _TEST_N_BINS)

    assert reloaded.content_hash() == saved_hash
    assert reloaded.content_hash() == bins.content_hash()

    # log_plaus must also agree, not just the hash
    row: pd.Series = ds.X_test.iloc[0]
    sample = EncodedSample(dict(row))
    assert bins.log_plaus(sample) == reloaded.log_plaus(sample)

    # cleanup
    _cache_path().unlink(missing_ok=True)


def test_bins_load_rejects_a_hand_edited_cache() -> None:
    import json

    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=_TEST_SEED).load()

    assert ds.X_train is not None and ds.columns is not None

    bins: FrozenBins = FrozenBins(_TEST_N_BINS).fit(ds.X_train, ds.columns)
    bins.save(_TEST_DATASET, _TEST_SEED)

    cache_path: Path = _cache_path()
    payload = json.loads(cache_path.read_text())
    payload["log_p"][ds.columns[0]][0] += (
        1.0  # tamper with a value, leave the hash alone
    )
    cache_path.write_text(json.dumps(payload))

    try:
        FrozenBins.load(_TEST_DATASET, _TEST_SEED, _TEST_N_BINS)
        raise AssertionError("expected a hash-mismatch failure on a tampered cache")
    except AssertionError as e:
        assert "hash mismatch" in str(e)
    finally:
        cache_path.unlink(missing_ok=True)


def test_fit_or_load_creates_then_reuses_cache() -> None:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=_TEST_SEED).load()

    assert ds.X_train is not None and ds.columns is not None

    cache_path: Path = _cache_path()
    cache_path.unlink(missing_ok=True)

    try:
        assert not cache_path.exists()
        first: FrozenBins = FrozenBins.fit_or_load(
            _TEST_DATASET, _TEST_SEED, _TEST_N_BINS, ds.X_train, ds.columns
        )
        assert cache_path.exists()

        second: FrozenBins = FrozenBins.fit_or_load(
            _TEST_DATASET, _TEST_SEED, _TEST_N_BINS, ds.X_train, ds.columns
        )
        assert second.content_hash() == first.content_hash()
    finally:
        cache_path.unlink(missing_ok=True)


def test_fit_or_load_fails_loudly_on_mismatch() -> None:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=_TEST_SEED).load()

    assert ds.X_train is not None and ds.columns is not None

    cache_path: Path = _cache_path()
    cache_path.unlink(missing_ok=True)

    try:
        FrozenBins.fit_or_load(
            _TEST_DATASET, _TEST_SEED, _TEST_N_BINS, ds.X_train, ds.columns
        )
        # Simulate the dataset changing on disk under an unchanged
        # (dataset, seed, n_bins) key by fitting from a different X_train.
        mutated: pd.DataFrame = ds.X_train.copy()
        mutated.iloc[:, 0] = mutated.iloc[:, 0] + 1.0

        try:
            FrozenBins.fit_or_load(
                _TEST_DATASET, _TEST_SEED, _TEST_N_BINS, mutated, ds.columns
            )
            raise AssertionError("expected a mismatch failure, not a silent rebin")
        except ValueError as e:
            assert "does not match" in str(e)
    finally:
        cache_path.unlink(missing_ok=True)

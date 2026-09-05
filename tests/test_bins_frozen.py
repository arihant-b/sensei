import sys
from pathlib import Path

import pandas as pd

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.data.bins import FrozenBins, Point  # noqa: E402
from sensei.data.loader import Dataset  # noqa: E402

_TEST_DATASET = "adult_test_bins_frozen"


def test_bins_hash_stable_across_save_and_load() -> None:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.X_train is not None and ds.columns is not None
    assert ds.X_test is not None

    bins: FrozenBins = FrozenBins(10).fit(ds.X_train, ds.columns)

    saved_hash: str = bins.save(_TEST_DATASET)
    reloaded: FrozenBins = FrozenBins.load(_TEST_DATASET)

    assert reloaded.content_hash() == saved_hash
    assert reloaded.content_hash() == bins.content_hash()

    # log_plaus must also agree, not just the hash
    row: pd.Series = ds.X_test.iloc[0]
    assert bins.log_plaus(Point(row)) == reloaded.log_plaus(Point(row))

    # cleanup
    (
        Path(__file__).resolve().parents[1] / "results" / f"bins_{_TEST_DATASET}.json"
    ).unlink(missing_ok=True)


def test_bins_load_rejects_a_hand_edited_cache() -> None:
    import json

    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.X_train is not None and ds.columns is not None

    bins: FrozenBins = FrozenBins(10).fit(ds.X_train, ds.columns)
    bins.save(_TEST_DATASET)

    cache_path: Path = (
        Path(__file__).resolve().parents[1] / "results" / f"bins_{_TEST_DATASET}.json"
    )
    payload = json.loads(cache_path.read_text())
    payload["log_p"][ds.columns[0]][0] += (
        1.0  # tamper with a value, leave the hash alone
    )
    cache_path.write_text(json.dumps(payload))

    try:
        FrozenBins.load(_TEST_DATASET)
        raise AssertionError("expected a hash-mismatch failure on a tampered cache")
    except AssertionError as e:
        assert "hash mismatch" in str(e)
    finally:
        cache_path.unlink(missing_ok=True)

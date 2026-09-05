import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse._csr import csr_matrix
from xgboost import Booster

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.data.loader import Dataset  # noqa: E402
from sensei.model.leaves import LeafMap  # noqa: E402
from sensei.model.train import Trainer  # noqa: E402


def _fixture() -> tuple[Dataset, Booster]:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.X_train is not None and ds.y_train is not None

    booster: Booster = Trainer.train_baseline(
        ds.X_train, ds.y_train, n_estimators=10, max_depth=3, seed=42
    )
    return ds, booster


def test_roundtrip_with_unchanged_v_is_bit_identical() -> None:
    ds, booster = _fixture()
    leaf_map = LeafMap(booster)

    roundtripped: Booster = leaf_map.write_leaf_values(booster, leaf_map.v0)

    original_margin: NDArray[np.float64] = np.asarray(
        booster.predict(__import__("xgboost").DMatrix(ds.X_test), output_margin=True)
    )
    new_margin: NDArray[np.float64] = np.asarray(
        roundtripped.predict(
            __import__("xgboost").DMatrix(ds.X_test), output_margin=True
        )
    )

    assert np.allclose(original_margin, new_margin, atol=1e-6)


def test_roundtrip_with_changed_v_actually_changes_predictions():
    ds, booster = _fixture()

    assert ds.X_test is not None

    leaf_map = LeafMap(booster)
    v_shifted = leaf_map.v0 + 1.0
    repaired: Booster = leaf_map.write_leaf_values(booster, v_shifted)

    phi: csr_matrix = leaf_map.phi_for(booster, ds.X_test)

    assert phi.shape is not None

    n_trees: int | None = phi.shape[1] if phi.ndim == 2 else None

    original_margin: NDArray[np.float64] = np.asarray(
        booster.predict(__import__("xgboost").DMatrix(ds.X_test), output_margin=True)
    )
    new_margin: NDArray[np.float64] = np.asarray(
        repaired.predict(__import__("xgboost").DMatrix(ds.X_test), output_margin=True)
    )

    n_trees = len(leaf_map._tree_leaf_lookup)
    assert np.allclose(new_margin - original_margin, n_trees, atol=1e-4)


def test_write_leaf_values_asserts_on_a_wrong_node_index() -> None:
    ds, booster = _fixture()
    leaf_map = LeafMap(booster)
    bad_leaf_index: dict[tuple[int, int], int] = dict(leaf_map.leaf_index)
    some_key = next(iter(bad_leaf_index))
    bad_leaf_index[(some_key[0], 0)] = bad_leaf_index.pop(some_key)
    leaf_map.leaf_index = bad_leaf_index

    try:
        leaf_map.write_leaf_values(booster, leaf_map.v0)
        raise AssertionError("expected an assertion error for a non-leaf node index")
    except AssertionError as e:
        assert "is not a leaf" in str(e)

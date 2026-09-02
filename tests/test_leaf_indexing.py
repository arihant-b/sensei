"""
Guards the ONE thing most likely to silently break sensitivity repair:
model.phi() and oracle.py's tree-path walk must agree, exactly, on every
leaf's global index and on which conditions route a point to it. A
mismatch here does not crash -- cuts just quietly constrain the wrong
leaves, gaps stop shrinking, and nothing errors. Run this after
touching anything in model.py or oracle.py (see CLAUDE.md).
"""

import numpy as np
import xgboost as xgb
from fixtures.synthetic import make_synthetic_dataset, make_synthetic_model

from oracle import SensitivityOracle


def test_leaf_indices_are_a_bijection_onto_0_n_leaves():
    model = make_synthetic_model(seed=0)
    indices = sorted(model.leaf_index.values())
    assert indices == list(range(model.n_leaves))


def test_leaf_values_come_from_the_gain_column():
    """XGBoost stores leaf values in Gain, not Value -- easy to get wrong."""
    model = make_synthetic_model(seed=0)
    assert model.booster is not None and model.v0 is not None
    df = model.booster.trees_to_dataframe()
    leaves = df[df["Feature"] == "Leaf"]

    assert len(leaves) == model.n_leaves

    for row in leaves.itertuples():
        tree_id = int(row.Tree)
        node_id = int(str(row.ID).split("-")[1])
        g = model.leaf_index[(tree_id, node_id)]
        assert model.v0[g] == float(row.Gain)


def test_phi_leaf_sum_matches_xgboost_raw_score():
    """
    Independent check: phi(x) @ v0 (our own leaf-value bookkeeping) must
    equal XGBoost's own raw margin score for x, computed entirely inside
    XGBoost with no reference to leaf_index/phi(). If leaf_index or
    phi() ever mis-maps a leaf, this is the check that catches it.
    """
    model = make_synthetic_model(seed=0)
    assert model.booster is not None
    X, _ = make_synthetic_dataset(seed=1, n=30)

    xgb_margin = model.booster.predict(xgb.DMatrix(X), output_margin=True)
    our_score = model.raw_score(X, v=model.v0)

    np.testing.assert_allclose(our_score, xgb_margin, atol=1e-6)


def test_oracle_tree_paths_agree_with_actual_routing():
    """
    oracle.py's root-to-leaf path walk (used to build the MILP's
    indicator constraints) must correctly describe which conditions
    route a point to which leaf. Cross-checked against XGBoost's own
    routing (pred_leaf=True) on real rows, not against oracle.py's own
    logic -- this is exactly the class of bug that silently produces
    wrong cuts.
    """
    model = make_synthetic_model(seed=0)
    assert model.booster is not None
    # _tree_leaf_paths only touches model -- validity/spec are unused here.
    oracle = SensitivityOracle(validity=None, spec=None, cfg=model.cfg)  # type: ignore[arg-type]
    _trees_leaves, leaf_paths = oracle._tree_leaf_paths(model)

    X, _ = make_synthetic_dataset(seed=2, n=50)
    leaf_ids_per_tree = np.atleast_2d(
        model.booster.predict(xgb.DMatrix(X), pred_leaf=True))

    for r in range(len(X)):
        row = X.iloc[r]
        for t in range(leaf_ids_per_tree.shape[1]):
            g = model.leaf_index[(t, int(leaf_ids_per_tree[r, t]))]

            for feat, thresh, direction in leaf_paths[g]:
                value = row[feat]
                if direction == "yes":
                    assert value < thresh, (
                        f"row {r} tree {t} leaf {g}: expected {feat}={value} "
                        f"< {thresh} on the 'yes' branch")
                else:
                    assert value >= thresh, (
                        f"row {r} tree {t} leaf {g}: expected {feat}={value} "
                        f">= {thresh} on the 'no' branch")

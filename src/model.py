import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb


class Ensemble:
    """
    Wraps an XGBoost model and exposes it in the form the repair QP needs:

        E_v(x) = sum_n  phi_n(x) * v_n

    where phi(x) is a 0/1 indicator over ALL leaves in the ensemble.
    Structure is frozen after fit(), so phi never changes again.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.booster = None
        self.v0 = None              # original leaf values
        self.v = None               # current (repaired) leaf values
        self.leaf_index = {}        # (tree_id, xgb_leaf_id) -> global leaf idx
        self.n_leaves = 0

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        clf = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            random_state=self.cfg.seed,
        )
        clf.fit(X, y)
        self.booster = clf.get_booster()
        self._index_leaves()
        self.v = self.v0.copy()
        return self

    def _index_leaves(self):
        """Assign a global index to every leaf and read off its value."""
        df = self.booster.trees_to_dataframe()
        leaves = df[df["Feature"] == "Leaf"]
        values = []

        for i, row in enumerate(leaves.itertuples()):
            tree_id = int(row.Tree)
            node_id = int(str(row.ID).split("-")[1])
            self.leaf_index[(tree_id, node_id)] = i
            values.append(float(row.Gain))       # leaf value lives in 'Gain'

        self.v0 = np.array(values, dtype=float)
        self.n_leaves = len(values)

    # -------------------------------------------------------------- phi / E
    def phi(self, X) -> sp.csr_matrix:
        """
        Leaf indicator matrix, shape (n_rows, n_leaves).
        Exactly one 1 per tree per row.
        """
        leaf_ids = self.booster.predict(xgb.DMatrix(X), pred_leaf=True)
        leaf_ids = np.atleast_2d(leaf_ids)
        n_rows, n_trees = leaf_ids.shape
        rows, cols = [], []

        for r in range(n_rows):
            for t in range(n_trees):
                rows.append(r)
                cols.append(self.leaf_index[(t, int(leaf_ids[r, t]))])

        data = np.ones(len(rows))
        return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, self.n_leaves))

    def raw_score(self, X, v=None) -> np.ndarray:
        v = self.v if v is None else v
        return self.phi(X) @ v

    def predict(self, X, v=None) -> np.ndarray:
        return (self.raw_score(X, v) > 0).astype(int)

    def set_leaf_values(self, v):
        self.v = np.asarray(v, dtype=float)

    def per_tree_gap(self, x1, x2) -> np.ndarray:
        """
        Delta_T = v[leaf_T(x1)] - v[leaf_T(x2)] for every tree T, so that
        sum(per_tree_gap(x1, x2)) == raw_score(x1) - raw_score(x2).

        Used to target the top-k contributing trees instead of treating
        an overall gap as one opaque number (e.g. via
        np.argsort(-np.abs(gaps))[:k]).
        """
        assert self.booster is not None and self.v is not None, "call fit() first"

        leaves1 = np.atleast_2d(
            self.booster.predict(self._single_row(x1), pred_leaf=True))[0]
        leaves2 = np.atleast_2d(
            self.booster.predict(self._single_row(x2), pred_leaf=True))[0]

        gaps = np.empty(len(leaves1), dtype=float)
        for t, (l1, l2) in enumerate(zip(leaves1, leaves2, strict=True)):
            g1 = self.leaf_index[(t, int(l1))]
            g2 = self.leaf_index[(t, int(l2))]
            gaps[t] = self.v[g1] - self.v[g2]

        return gaps

    def _single_row(self, x) -> xgb.DMatrix:
        """
        Wrap one row (Series, 1-row DataFrame, or bare array) for xgboost.

        The booster was fit on named columns, so it validates DMatrix
        feature names on predict(): a bare array needs those names
        supplied explicitly, or xgboost rejects it.
        """
        if isinstance(x, pd.Series):
            return xgb.DMatrix(x.to_frame().T)
        if isinstance(x, pd.DataFrame):
            return xgb.DMatrix(x.iloc[[0]])

        assert self.booster is not None, "call fit() first"
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        return xgb.DMatrix(arr, feature_names=self.booster.feature_names)

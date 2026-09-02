import numpy as np
import pandas as pd
import xgboost as xgb


class SamplingScreen:
    """
    Cheap violation finder. No solver, no XGBoost predict per call.

    Can say "here is a violation" but can NEVER say "none exists".
    That is why the oracle is still required for the certificate.

    Tree structure is frozen after Ensemble.fit() (CLAUDE.md invariant
    1): which leaf a row lands in per tree never changes across the
    repair loop, only the leaf VALUES do. So routing is computed once
    per (model, X) -- one XGBoost predict call -- and cached; every
    find_violations() call after that is array indexing against the
    CURRENT model.v, plus a pure-Python re-walk for the (typically few)
    rows whose flip actually crosses a split threshold.
    """

    def __init__(self, spec, validity, cfg, rng: np.random.Generator | None = None):
        self.spec = spec
        self.validity = validity
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)

        self._cache_key: tuple | None = None
        self._leaf_table: np.ndarray | None = None    # (n_rows, n_trees) leaf idx
        self._leaf_paths: dict[int, list[tuple[str, float, str]]] | None = None
        self._tree_nodes: dict[int, dict[str, tuple]] | None = None
        self._columns: list[str] | None = None
        self._X_values: np.ndarray | None = None

    def find_violations(
        self, model, X: pd.DataFrame, feature: str, k: int
    ) -> list[tuple[pd.Series, pd.Series, float]]:
        """Flip `feature` on every row of X; keep the worst k gaps."""
        assert model.v is not None, "model must be fit before screening"
        self._ensure_tables(model, X)
        assert self._leaf_table is not None and self._leaf_paths is not None
        assert self._columns is not None and self._X_values is not None

        flipped = self._flip(X, feature)
        new_vals = flipped[feature].to_numpy()

        flipped_leaf_table = self._flipped_leaf_table(model, feature, new_vals)

        v = model.v
        gaps = v[self._leaf_table].sum(axis=1) - v[flipped_leaf_table].sum(axis=1)

        order = np.argsort(-np.abs(gaps))
        out = []

        for i in order:
            if abs(gaps[i]) <= self.cfg.epsilon:
                break

            if self.validity.is_valid_pair(X.iloc[i], flipped.iloc[i]):
                out.append((X.iloc[i], flipped.iloc[i], float(gaps[i])))

            if len(out) >= k:
                break

        return out

    def _flip(self, rows: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Cycles `feature` to its next level in sorted order, wrapping
        around. For a binary feature (two observed levels) this is
        exactly a flip; for a multi-valued categorical it steps to the
        next category.

        Levels come from `rows` itself, not an external registry --
        this screen is cheap and incomplete by design (see class
        docstring), so a rare category absent from `rows` is simply not
        considered here.
        """
        levels = np.sort(rows[feature].unique())

        if len(levels) <= 1:
            return rows.copy()

        next_level = dict(zip(levels, np.roll(levels, -1), strict=True))
        flipped = rows.copy()
        flipped[feature] = rows[feature].map(next_level)
        return flipped

    # ------------------------------------------------------------ tables
    def _ensure_tables(self, model, X: pd.DataFrame) -> None:
        """Builds routing tables once per (model, X); reused for the model's
        whole lifetime since tree structure never changes after fit()."""
        key = (id(model), id(X))
        if key == self._cache_key:
            return

        self._tree_nodes, self._leaf_paths = self._parse_tree_structure(model)
        self._leaf_table = self._build_leaf_table(model, X)
        self._columns = list(X.columns)
        self._X_values = X.to_numpy()
        self._cache_key = key

    def _parse_tree_structure(
        self, model
    ) -> tuple[dict[int, dict[str, tuple]], dict[int, list[tuple[str, float, str]]]]:
        """
        Per tree: a walkable node map (used only as a fallback when a
        flip actually crosses a threshold) and, per global leaf index,
        the root-to-leaf split conditions (used to cheaply check whether
        a flip keeps a row in the same leaf). Global leaf indices come
        from model.leaf_index -- the SAME dict model.phi() uses.
        """
        assert model.booster is not None
        df = model.booster.trees_to_dataframe()
        by_tree: dict[int, dict[str, object]] = {}

        for row in df.itertuples():
            by_tree.setdefault(int(row.Tree), {})[str(row.ID)] = row

        nodes: dict[int, dict[str, tuple]] = {}
        leaf_paths: dict[int, list[tuple[str, float, str]]] = {}

        for tree_id, tree_rows in by_tree.items():
            nodes[tree_id] = {}
            root = f"{tree_id}-0"
            stack: list[tuple[str, list[tuple[str, float, str]]]] = [(root, [])]

            while stack:
                node_id, path = stack.pop()
                row = tree_rows[node_id]

                if row.Feature == "Leaf":                       # type: ignore[attr-defined]
                    leaf_num = int(node_id.split("-")[1])
                    g = model.leaf_index[(tree_id, leaf_num)]
                    nodes[tree_id][node_id] = ("leaf", g)
                    leaf_paths[g] = path
                else:
                    feat = str(row.Feature)                      # type: ignore[attr-defined]
                    thresh = float(row.Split)                    # type: ignore[attr-defined]
                    yes_id, no_id = str(row.Yes), str(row.No)    # type: ignore[attr-defined]
                    nodes[tree_id][node_id] = ("split", feat, thresh, yes_id, no_id)
                    stack.append((yes_id, [*path, (feat, thresh, "yes")]))
                    stack.append((no_id, [*path, (feat, thresh, "no")]))

        return nodes, leaf_paths

    def _build_leaf_table(self, model, X: pd.DataFrame) -> np.ndarray:
        """The ONE XGBoost predict call this module makes -- global leaf
        index per row per tree, for the routing that's frozen thereafter."""
        assert model.booster is not None
        raw_leaf_ids = np.atleast_2d(
            model.booster.predict(xgb.DMatrix(X), pred_leaf=True)).astype(int)
        n_rows, n_trees = raw_leaf_ids.shape

        by_tree: dict[int, dict[int, int]] = {}
        for (t, raw_id), g in model.leaf_index.items():
            by_tree.setdefault(t, {})[raw_id] = g

        table = np.full((n_rows, n_trees), -1, dtype=int)
        for t in range(n_trees):
            mapping = by_tree[t]
            leaf_map = np.full(max(mapping) + 1, -1, dtype=int)
            for raw_id, g in mapping.items():
                leaf_map[raw_id] = g
            table[:, t] = leaf_map[raw_leaf_ids[:, t]]

        return table

    # ------------------------------------------------------- flip routing
    def _flipped_leaf_table(
        self, model, feature: str, new_vals: np.ndarray
    ) -> np.ndarray:
        assert self._leaf_table is not None and self._leaf_paths is not None

        levels = np.sort(np.unique(new_vals))
        consistency = self._leaf_consistency(feature, levels)

        new_level_idx = np.searchsorted(levels, new_vals)
        # is_consistent[r, t]: does row r's new value keep tree t's leaf unchanged?
        is_consistent = consistency[self._leaf_table, new_level_idx[:, None]]

        flipped_leaf_table = self._leaf_table.copy()
        rows_needing_walk, trees_needing_walk = np.nonzero(~is_consistent)

        for r in np.unique(rows_needing_walk):
            assert self._columns is not None and self._X_values is not None
            row_values = dict(zip(self._columns, self._X_values[r], strict=True))
            row_values[feature] = new_vals[r]

            for t in trees_needing_walk[rows_needing_walk == r]:
                flipped_leaf_table[r, int(t)] = self._walk(int(t), row_values)

        return flipped_leaf_table

    def _leaf_consistency(self, feature: str, levels: np.ndarray) -> np.ndarray:
        """
        (n_leaves_total, len(levels)) bool array: for each leaf and each
        candidate value of `feature`, does that value still satisfy the
        leaf's condition(s) on `feature`? Leaves whose path never tests
        `feature` default to True (always unchanged) for every level.
        """
        assert self._leaf_paths is not None
        n_leaves = max(self._leaf_paths) + 1
        consistency = np.ones((n_leaves, len(levels)), dtype=bool)

        for g, path in self._leaf_paths.items():
            conds = [(thresh, direction) for feat, thresh, direction in path
                     if feat == feature]
            if not conds:
                continue

            for i, val in enumerate(levels):
                consistency[g, i] = all(
                    (val < thresh) if direction == "yes" else (val >= thresh)
                    for thresh, direction in conds
                )

        return consistency

    def _walk(self, tree_id: int, row_values: dict) -> int:
        """Root-to-leaf, O(depth): only reached when a flip actually
        crosses a threshold, so this is the minority case per call."""
        assert self._tree_nodes is not None
        nodes = self._tree_nodes[tree_id]
        node_id = f"{tree_id}-0"

        while True:
            info = nodes[node_id]
            if info[0] == "leaf":
                return info[1]
            _, feat, thresh, yes_id, no_id = info
            node_id = yes_id if row_values[feat] < thresh else no_id

import json
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from numpy.typing import NDArray


class LeafMap:
    """
    Flat leaf indexing for a frozen XGBoost tree structure. Each leaf is assigned
    a unique global index n, which is used for all leaf-value vectors v, leaf
    indicator vectors ell, and cut definitions. The mapping is built once from a
    booster and never changes, even if the leaf values themselves are repaired.
    """

    def __init__(self, booster: xgb.Booster) -> None:
        self.leaf_index: dict[tuple[int, int], int] = {}
        self.n_leaves: int = 0
        self.v0: NDArray[np.float64]
        self.base_score: float
        self.raw_base_score: float
        self._tree_leaf_lookup: dict[int, NDArray[np.int64]] = {}

        self._index_leaves(booster)
        self._build_tree_leaf_lookup()
        self.base_score = self._read_base_score(booster)

    def phi_for(self, booster: xgb.Booster, X: pd.DataFrame) -> sp.csr_matrix:
        """
        Leaf indicator matrix phi, shape (n_samples, n_leaves): row i, column
        n is 1 iff sample i reaches leaf n, else 0 -- exactly one 1 per tree
        per row. Depends only on the frozen tree structure, so it's computed
        once per stage and reused against every candidate `v` (see
        `margin_score`).

        Args:
            booster (xgb.Booster): The model to route `X` through.
            X (pd.DataFrame): Rows to compute leaf indicators for.

        Returns:
            sp.csr_matrix: The `(n_samples, n_leaves)` leaf indicator matrix.
        """

        leaf_ids: NDArray[np.int64] = np.atleast_2d(
            booster.predict(xgb.DMatrix(X), pred_leaf=True)
        ).astype(np.int64)
        n_rows, n_trees = leaf_ids.shape

        global_cols: NDArray[np.int64] = np.empty((n_rows, n_trees), dtype=np.int64)

        for t in range(n_trees):
            global_cols[:, t] = self._tree_leaf_lookup[t][leaf_ids[:, t]]

        rows: NDArray[np.int64] = np.repeat(np.arange(n_rows), n_trees)
        cols: NDArray[np.int64] = global_cols.ravel()
        data: NDArray[np.float64] = np.ones(len(rows))
        return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, self.n_leaves))

    def margin_score(
        self, phi: sp.csr_matrix, v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        E_v(x) = base_score + phi(x) @ v, for every row in phi -- the raw
        ensemble margin (not a probability; sigmoid only at reporting).

        Args:
            phi (sp.csr_matrix): Leaf indicator matrix from `phi_for`.
            v (NDArray[np.float64]): Leaf values to score with.

        Returns:
            NDArray[np.float64]: The margin score per row.
        """

        leaf_scores: NDArray[np.float64] = np.asarray(phi @ v, dtype=np.float64)
        return leaf_scores + self.base_score

    def leaf_ids_for_row(
        self, booster: xgb.Booster, x: pd.DataFrame
    ) -> NDArray[np.int64]:
        """
        One row's global leaf index per tree -- `ell` for a single point x.

        Args:
            booster (xgb.Booster): The model to route `x` through.
            x (pd.DataFrame): A single-row DataFrame.

        Returns:
            NDArray[np.int64]: The active global leaf index per tree.
        """

        raw: NDArray[np.int64] = np.atleast_2d(
            booster.predict(xgb.DMatrix(x), pred_leaf=True)
        )[0].astype(int)
        return np.array(
            [self._tree_leaf_lookup[t][raw[t]] for t in range(len(raw))], dtype=np.int64
        )

    def write_leaf_values(
        self, booster: xgb.Booster, v: NDArray[np.float64]
    ) -> xgb.Booster:
        """
        Dump `booster` to JSON, overwrite every leaf's value with `v`
        (structure -- splits, thresholds, tree count -- untouched), and
        reload as a new `Booster`. This is the only way leaf values get
        written back: XGBoost has no in-place "set this leaf's value" API.

        Args:
            booster (xgb.Booster): The model whose leaf values to overwrite.
            v (NDArray[np.float64]): New leaf values, indexed by global leaf.

        Returns:
            xgb.Booster: A new booster with `v`'s leaf values, same structure.
        """

        with tempfile.TemporaryDirectory() as tmp:
            path: Path = Path(tmp) / "model.json"
            booster.save_model(str(path))
            model: dict[str, Any] = json.loads(path.read_text())
            trees: list[dict[str, Any]] = model["learner"]["gradient_booster"]["model"][
                "trees"
            ]

            for tree_id, mapping in self._by_tree(self.leaf_index).items():
                tree: dict[str, Any] = trees[tree_id]
                left_children = tree["left_children"]

                for node_id, g in mapping.items():
                    assert left_children[node_id] == -1, (
                        f"tree {tree_id} node {node_id} is not a leaf -- leaf_map is "
                        f"inconsistent with this booster's structure"
                    )
                    tree["base_weights"][node_id] = float(v[g])
                    tree["split_conditions"][node_id] = float(v[g])

            out_path: Path = Path(tmp) / "model_repaired.json"
            out_path.write_text(json.dumps(model))

            new_booster = xgb.Booster()
            new_booster.load_model(str(out_path))

        return new_booster

    def _index_leaves(self, booster: xgb.Booster) -> None:
        """
        Assign every (tree_id, node_id) leaf a flat global index n, in the
        order `trees_to_dataframe()` lists them, and record their original
        values as v0. Built once at construction; never touched again.

        Args:
            booster (xgb.Booster): The model to index leaves for.
        """

        df: pd.DataFrame = booster.trees_to_dataframe()
        leaves: pd.DataFrame = df[df["Feature"] == "Leaf"]
        values: list[float] = []

        for i, row in enumerate(leaves.itertuples()):
            tree_id: int = cast(int, row.Tree)
            node_id: int = cast(int, row.Node)
            self.leaf_index[(tree_id, node_id)] = i
            values.append(cast(float, row.Gain))

        self.v0 = np.array(values, dtype=float)
        self.n_leaves = len(values)

    def _build_tree_leaf_lookup(self) -> None:
        """
        Per-tree array: node_id -> global leaf n (-1 where node_id isn't a leaf).

        Builds `self._tree_leaf_lookup` from `self.leaf_index`.
        """

        by_tree: dict[int, dict[int, int]] = {}

        for (tree_id, node_id), g in self.leaf_index.items():
            by_tree.setdefault(tree_id, {})[node_id] = g

        for tree_id, mapping in by_tree.items():
            lookup: NDArray[np.int64] = np.full(max(mapping) + 1, -1, dtype=np.int64)

            for node_id, g in mapping.items():
                lookup[node_id] = g

            self._tree_leaf_lookup[tree_id] = lookup

    def _read_base_score(self, booster: xgb.Booster) -> float:
        """
        Base score in MARGIN space: XGBoost stores it as the raw
        `binary:logistic` probability, so this reads that value and applies
        the inverse sigmoid (logit) to get the constant `margin_score` adds.
        It's a fixed bias, unrelated to leaf values -- never touched by
        repair, but required for real predictions/accuracy since it cancels
        out of every gap (`E_v(x1) - E_v(x2)`).

        Args:
            booster (xgb.Booster): The model to read the base score from.

        Returns:
            float: The base score, in margin space.

        Raises:
            AssertionError: The booster's objective isn't `binary:logistic`
                (the only one this inverse-link has been verified against).
        """

        config: dict[str, Any] = json.loads(booster.save_config())
        objective: str = config["learner"]["objective"]["name"]
        base_score_raw: str = config["learner"]["learner_model_param"]["base_score"]
        raw_base_score = float(str(base_score_raw).strip("[]"))

        assert objective == "binary:logistic", (
            f"base_score link transform is only verified for binary:logistic, "
            f"got objective={objective!r} -- do not assume logit here without "
            f"re-deriving and re-verifying the correct inverse link first"
        )

        self.raw_base_score = raw_base_score
        return float(np.log(raw_base_score / (1.0 - raw_base_score)))

    def _by_tree(
        self, leaf_index: dict[tuple[int, int], int]
    ) -> dict[int, dict[int, int]]:
        """
        Regroup {(tree_id, node_id): n} as {tree_id: {node_id: n}}, for
        `write_leaf_values`.

        Args:
            leaf_index (dict[tuple[int, int], int]): `(tree_id, node_id) ->`
                flat global leaf index map.

        Returns:
            dict[int, dict[int, int]]: `tree_id -> {node_id: n}`.
        """

        by_tree: dict[int, dict[int, int]] = {}

        for (tree_id, node_id), g in leaf_index.items():
            by_tree.setdefault(tree_id, {})[node_id] = g

        return by_tree

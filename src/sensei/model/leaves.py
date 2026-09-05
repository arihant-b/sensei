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
        Compute the leaf indicator matrix phi for a dataset X. Each row corresponds
        to a sample in X, and each column corresponds to a global leaf index n. The
        matrix is sparse, with a 1 indicating that the sample reaches the corresponding
        leaf in the booster.

        Args:
            booster (xgb.Booster): The XGBoost booster to use for leaf ID computation.
            X (pd.DataFrame): The input features for which to compute the leaf indicator
                              matrix.

        Returns:
            sp.csr_matrix: The sparse leaf indicator matrix phi of shape (n_samples,
                           n_leaves).
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
        Compute the margin scores for a dataset given the leaf indicator matrix phi and
        the leaf values v.
        E_v(x) = base_score + sum_t v[leaf_of(t, x)], for every row in phi.

        Args:
            phi (sp.csr_matrix): The sparse leaf indicator matrix of shape (n_samples,
                                 n_leaves).
            v (NDArray[np.float64]): The leaf values vector of shape (n_leaves,).

        Returns:
            NDArray[np.float64]: The margin scores for each sample in the dataset.
        """

        leaf_scores: NDArray[np.float64] = np.asarray(phi @ v, dtype=np.float64)
        return leaf_scores + self.base_score

    def leaf_ids_for_row(
        self, booster: xgb.Booster, x: pd.DataFrame
    ) -> NDArray[np.int64]:
        """
        Compute the global leaf indices for a single row of input features x. Each
        element in the returned array corresponds to a tree in the booster, and the
        value is the global leaf index n that the row reaches in that tree.

        Args:
            booster (xgb.Booster): The XGBoost booster to use for leaf ID computation.
            x (pd.DataFrame): The input features for which to compute the global leaf
                              indices.

        Returns:
            NDArray[np.int64]: The global leaf indices for the input row.
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
        Write leaf values v into a copy of the booster, returning a new booster with
        the updated leaf values. The tree structure remains unchanged.

        Args:
            booster (xgb.Booster): The XGBoost booster to copy and update with new leaf
                                   values.
            v (NDArray[np.float64]): The new leaf values to write into the booster.

        Returns:
            xgb.Booster: A new XGBoost booster with the updated leaf values.
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
        Build the leaf index mapping from (tree_id, node_id) to a flat global index n,
        and store the original leaf values in v0. This mapping is built once from the
        booster and never changes, even if the leaf values themselves are repaired.

        Args:
            booster (xgb.Booster): The XGBoost booster from which to build the leaf
                                   index mapping.
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
        Build a lookup table for each tree that maps node IDs to global leaf indices.
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
        Read the base score (global bias term) from the booster configuration. The base
        score is a constant that is added to the margin scores and is required for
        computing actual predictions and accuracy. It is not part of the leaf values and
        does not change during repair.

        Args:
            booster (xgb.Booster): The XGBoost booster from which to read the base
                                   score.

        Returns:
            float: The base score (global bias term) of the booster.
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
        Group the leaf indices by tree. This is used by `write_leaf_values`.

        Args:
            leaf_index (dict[tuple[int, int], int]): The mapping from (tree_id, node_id)
                                                     to global leaf index n.

        Returns:
            dict[int, dict[int, int]]: A dictionary where each key is a tree_id and the
                                       value is another dictionary mapping node_id to
                                       global leaf index n for that tree.
        """

        by_tree: dict[int, dict[int, int]] = {}

        for (tree_id, node_id), g in leaf_index.items():
            by_tree.setdefault(tree_id, {})[node_id] = g

        return by_tree

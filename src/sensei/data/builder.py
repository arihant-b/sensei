import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

_DATASET_ROOT: Path = Path(__file__).resolve().parents[3] / "dataset"

log: logging.Logger = logging.getLogger("sensei.data.builder")


class DatasetBuilder:
    """
    Build a dataset/<name>/ directory from a raw CSV, with categorical encoding,
    feature scaling, and a train/test split. The output is cached to disk for
    later use by loader.py and the Ensense core. Encoding/decoding conventions
    here are matched to Ensense core's own reference implementation
    (ensense/src/read_output.py::compute_scaling_func/decode_point) on purpose,
    NOT independently chosen -- a model trained on our encoding must mean the
    same thing to Ensense core's oracle as it does to us. Artifacts expected on
    disk:
        encoding_map.json    raw categorical value -> ordinal index (sorted,
                             missing/unmapped values excluded from the list --
                             see _encode_categoricals)
        scaler.pkl           sklearn MinMaxScaler, fit on the FULL encoded
                             feature matrix (train+test together, before the
                             split) -- deliberately, to match Ensense core's own
                             scaler exactly, confirmed empirically: Ensense's
                             adult scaler reports fnlwgt's global minimum as
                             data_min_, but that row only exists in ITS test
                             split, which is only possible if the scaler saw
                             both splits at fit time. Fitting on train alone
                             would silently make our scaler disagree with any
                             Ensense-side computation that assumes the shared
                             convention.
        full.csv             whole dataset, scaled, index column kept
        train.csv / test.csv stratified split of full.csv, no index column
        feature_map.json     column name -> f<i> (position in train.csv)
        details.csv          feature,name,lb,ub.
    """

    @staticmethod
    def build_dataset(
        raw_csv: Path,
        name: str,
        label_col: str | None = None,
        categorical_cols: list[str] | None = None,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        """
        Build `dataset/<name>/` from `raw_csv`: encode, scale, split, and
        write every artifact listed in this class's own docstring.

        Args:
            raw_csv (Path): Path to the raw CSV.
            name (str): `dataset/<name>/` to write into.
            label_col (str | None): Label column; the last column if None.
            categorical_cols (list[str] | None): Categorical columns;
                every string-dtype feature column if None.
            test_size (float): Fraction held out for `test.csv`.
            seed (int): Split seed.
        """

        df: pd.DataFrame = pd.read_csv(raw_csv)

        label_col, feature_cols, categorical_cols = DatasetBuilder._resolve_columns(
            df, label_col, categorical_cols
        )

        encoded, encoding_map = DatasetBuilder._encode_categoricals(
            df, categorical_cols
        )
        DatasetBuilder._encode_label(df, encoded, label_col, encoding_map)

        full, scaler = DatasetBuilder._scale_features(encoded, feature_cols, label_col)
        train_df, test_df = DatasetBuilder._split(full, label_col, test_size, seed)

        out_dir: Path = _DATASET_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        DatasetBuilder._write_csvs(out_dir, full, train_df, test_df)
        DatasetBuilder._write_scaler(out_dir, scaler)
        DatasetBuilder._write_encoding_map(out_dir, encoding_map)
        DatasetBuilder._write_feature_map(out_dir, feature_cols, label_col)
        DatasetBuilder._write_details(out_dir, feature_cols, train_df)

        log.info(
            f"wrote dataset/{name}/ "
            f"({len(feature_cols)} features, {len(categorical_cols)} categorical, "
            f"{len(train_df)} train / {len(test_df)} test rows)"
        )

    @staticmethod
    def _resolve_columns(
        df: pd.DataFrame, label_col: str | None, categorical_cols: list[str] | None
    ) -> tuple[str, list[str], list[str]]:
        """
        Fill in whatever `build_dataset` callers left as None: the label
        defaults to `df`'s last column, categorical columns default to
        every string-dtype feature column.

        Args:
            df (pd.DataFrame): The raw DataFrame.
            label_col (str | None): Label column; the last column if None.
            categorical_cols (list[str] | None): Categorical columns;
                every string-dtype feature column if None.

        Returns:
            tuple[str, list[str], list[str]]: `(label_col, feature_cols,
                categorical_cols)`, fully resolved.
        """

        if label_col is None:
            label_col = df.columns[-1]

        feature_cols: list[str] = list(filter(lambda c: c != label_col, df.columns))

        if categorical_cols is None:
            # pandas 3.x infers a native 'str' dtype for CSV text columns, not
            # the legacy 'object' dtype -- is_string_dtype() catches both.
            categorical_cols = list(
                filter(lambda c: is_string_dtype(df[c]), feature_cols)
            )

        return label_col, feature_cols, categorical_cols

    @staticmethod
    def _encode_categoricals(
        df: pd.DataFrame, categorical_cols: list[str]
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Encode categorical columns as ordinal indices, and return the encoding map.
        Matches Ensense core's own encoding exactly
        (ensense/src/read_output.py::compute_scaling_func): the category list is
        the column's distinct non-missing values, sorted; a raw value with no
        entry in that list (missing, or an unrecognized category) is encoded as
        -1, NOT dropped -- confirmed against Ensense's own bundled adult
        scaler, whose data_min_ is exactly -1.0 for the three columns
        (workclass, occupation, native-country) that have missing values in the
        raw data. `.map(index_of)` already returns NaN for both a real missing
        value and an unrecognized one, so `.fillna(-1)` alone handles both
        cases without distinguishing them, same as upstream.

        Args:
            df (pd.DataFrame): The raw DataFrame.
            categorical_cols (list[str]): Categorical columns to encode.

        Returns:
            tuple[pd.DataFrame, dict[str, list[str]]]: The encoded
                DataFrame and its `feature -> sorted categories` map.
        """

        encoded: pd.DataFrame = df.copy()
        encoding_map: dict[str, list[str]] = {}

        for col in categorical_cols:
            categories: list[str] = sorted(df[col].dropna().unique().tolist())
            encoding_map[col] = categories
            index_of: dict[str, int] = {v: i for i, v in enumerate(categories)}
            encoded[col] = df[col].map(index_of).fillna(-1).astype(int)

        return encoded, encoding_map

    @staticmethod
    def _encode_label(
        df: pd.DataFrame,
        encoded: pd.DataFrame,
        label_col: str,
        encoding_map: dict[str, list[str]],
    ) -> None:
        """
        If the label is a string column, ordinal-encode it too (sorted,
        same convention as `_encode_categoricals`) and record it in
        `encoding_map`; a numeric label is left as-is. Mutates `encoded`
        in place.

        Args:
            df (pd.DataFrame): The raw DataFrame (for the label's raw values).
            encoded (pd.DataFrame): Mutated in place with the encoded label.
            label_col (str): The label column's name.
            encoding_map (dict[str, list[str]]): Mutated in place with the
                label's categories, if encoded.
        """

        if is_string_dtype(encoded[label_col]):
            label_categories: list[str] = sorted(df[label_col].unique().tolist())
            encoding_map[label_col] = label_categories
            encoded[label_col] = df[label_col].map(
                {v: i for i, v in enumerate(label_categories)}
            )

    @staticmethod
    def decode_point(
        raw_point: Mapping[str, float], encoding_map: dict[str, list[str]]
    ) -> dict[str, str | float]:
        """
        Decode one row of already-UNSCALED (raw ordinal-code, not [0,1]) values
        back to human-readable form, matching Ensense core's own
        `decode_point` exactly (ensense/src/read_output.py): round a
        categorical feature to the nearest integer and look it up in its
        sorted category list; a code outside the list's range (-1 for
        missing, or a fractional/out-of-domain value from a solver's
        continuous relaxation) is reported as `"<unknown_{idx}>"` rather than
        raising. Non-categorical features pass through unchanged. Call
        `scaler.inverse_transform` first -- this only maps ordinal codes back
        to category names, it does not undo the [0,1] scaling.

        Args:
            raw_point (Mapping[str, float]): One row's already-unscaled
                (raw ordinal-code) values.
            encoding_map (dict[str, list[str]]): Feature -> sorted categories.

        Returns:
            dict[str, str | float]: The row with categorical codes decoded
                to category names (or `"<unknown_{idx}>"`).
        """

        decoded: dict[str, str | float] = {}

        for feature, value in raw_point.items():
            if feature in encoding_map:
                idx: int = int(round(float(value)))
                categories: list[str] = encoding_map[feature]
                in_range: bool = 0 <= idx < len(categories)
                decoded[feature] = categories[idx] if in_range else f"<unknown_{idx}>"
            else:
                decoded[feature] = value

        return decoded

    @staticmethod
    def _scale_features(
        encoded: pd.DataFrame, feature_cols: list[str], label_col: str
    ) -> tuple[pd.DataFrame, MinMaxScaler]:
        """
        Scale the feature columns to [0, 1] with MinMaxScaler fit on the FULL
        encoded feature matrix (before the train/test split), matching Ensense
        core's own convention exactly -- see `build_dataset`'s docstring for
        why this is deliberate and not a leakage bug.

        Args:
            encoded (pd.DataFrame): The encoded DataFrame to scale.
            feature_cols (list[str]): Feature columns to scale.
            label_col (str): The label column, appended unscaled.

        Returns:
            tuple[pd.DataFrame, MinMaxScaler]: The scaled DataFrame and the
                fitted scaler.
        """

        scaler = MinMaxScaler()
        scaled_features = pd.DataFrame(
            scaler.fit_transform(encoded[feature_cols]),
            columns=feature_cols,
            index=encoded.index,
        )
        full: pd.DataFrame = pd.concat([scaled_features, encoded[label_col]], axis=1)
        return full, scaler

    @staticmethod
    def _split(
        full: pd.DataFrame, label_col: str, test_size: float, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified train/test split of the already-scaled `full` DataFrame.

        Args:
            full (pd.DataFrame): The already-scaled DataFrame to split.
            label_col (str): The label column, for stratification.
            test_size (float): Fraction held out for the test split.
            seed (int): Split seed.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: `(train_df, test_df)`.
        """

        train_df, test_df = train_test_split(
            full,
            test_size=test_size,
            random_state=seed,
            stratify=full[label_col],
        )
        return train_df, test_df

    @staticmethod
    def _write_csvs(
        out_dir: Path, full: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """
        full.csv keeps its index column; train.csv/test.csv don't.

        Args:
            out_dir (Path): Directory to write into.
            full (pd.DataFrame): The full, scaled dataset.
            train_df (pd.DataFrame): The training split.
            test_df (pd.DataFrame): The test split.
        """

        full.to_csv(out_dir / "full.csv", index=True)
        train_df.to_csv(out_dir / "train.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)

    @staticmethod
    def _write_scaler(out_dir: Path, scaler: MinMaxScaler) -> None:
        """
        Pickle the fitted scaler to scaler.pkl.

        Args:
            out_dir (Path): Directory to write into.
            scaler (MinMaxScaler): The fitted scaler to persist.
        """

        joblib.dump(scaler, out_dir / "scaler.pkl")

    @staticmethod
    def _write_encoding_map(out_dir: Path, encoding_map: dict[str, list[str]]) -> None:
        """
        Write encoding_map.json, if there were any categorical columns to encode.

        Args:
            out_dir (Path): Directory to write into.
            encoding_map (dict[str, list[str]]): Feature -> sorted categories.
        """

        if encoding_map:
            (out_dir / "encoding_map.json").write_text(json.dumps(encoding_map))

    @staticmethod
    def _write_feature_map(
        out_dir: Path, feature_cols: list[str], label_col: str
    ) -> None:
        """
        Write feature_map.json: each feature column -> "f<i>", label -> "label".

        Args:
            out_dir (Path): Directory to write into.
            feature_cols (list[str]): Feature columns, in order.
            label_col (str): The label column's name.
        """

        feature_map: dict[str, str] = {
            col: f"f{i}" for i, col in enumerate(feature_cols)
        }
        feature_map[label_col] = "label"
        (out_dir / "feature_map.json").write_text(json.dumps(feature_map))

    @staticmethod
    def _write_details(
        out_dir: Path, feature_cols: list[str], train_df: pd.DataFrame
    ) -> None:
        """
        Write details.csv (feature,name,lb,ub), lb/ub taken from train_df's
        own min/max -- typically inside but not exactly [0, 1], since the
        scaler was fit on the full dataset while this reads only train.

        Args:
            out_dir (Path): Directory to write into.
            feature_cols (list[str]): Feature columns, in order.
            train_df (pd.DataFrame): The training split, for lb/ub.
        """

        details = pd.DataFrame(
            {
                "feature": range(len(feature_cols)),
                "name": feature_cols,
                "lb": [train_df[c].min() for c in feature_cols],
                "ub": [train_df[c].max() for c in feature_cols],
            }
        )
        details.to_csv(out_dir / "details.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="path to the raw CSV")
    parser.add_argument("--name", required=True, help="dataset/<name>/ to write into")
    parser.add_argument(
        "--label", default=None, help="label column (default: last column)"
    )
    parser.add_argument(
        "--categorical",
        default=None,
        help="comma-separated categorical column names (default: auto-detect object "
        "dtype)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args: argparse.Namespace = parser.parse_args()
    categorical_cols: Any | None = (
        args.categorical.split(",") if args.categorical else None
    )

    DatasetBuilder.build_dataset(
        raw_csv=args.raw,
        name=args.name,
        label_col=args.label,
        categorical_cols=categorical_cols,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

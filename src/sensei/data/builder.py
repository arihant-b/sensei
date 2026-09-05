import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

_DATASET_ROOT: Path = Path(__file__).resolve().parents[3] / "dataset"


class DatasetBuilder:
    """
    Build a dataset/<name>/ directory from a raw CSV, with train/test split,
    categorical encoding, and feature scaling. The output is cached to disk for
    later use by loader.py and the Ensense core. Artifacts expected on disk:
        encoding_map.json    raw categorical value -> ordinal index (sorted)
        scaler.pkl           sklearn MinMaxScaler, fit on the FULL encoded feature
                             matrix
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
        Build a dataset from a raw CSV file.

        Args:
            raw_csv (Path): Path to the raw CSV file.
            name (str): Name of the dataset.
            label_col (str | None, optional): Name of the label column. Defaults to
                                              None.
            categorical_cols (list[str] | None, optional): List of categorical column
                                                           names. Defaults to None.
            test_size (float, optional): Proportion of the dataset to include in the
                                         test split. Defaults to 0.2.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """

        df: pd.DataFrame = pd.read_csv(raw_csv)

        label_col, feature_cols, categorical_cols = DatasetBuilder._resolve_columns(
            df, label_col, categorical_cols
        )
        DatasetBuilder._drop_missing_categoricals(df, categorical_cols)

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

        print(
            f"wrote dataset/{name}/ "
            f"({len(feature_cols)} features, {len(categorical_cols)} categorical, "
            f"{len(train_df)} train / {len(test_df)} test rows)"
        )

    @staticmethod
    def _resolve_columns(
        df: pd.DataFrame, label_col: str | None, categorical_cols: list[str] | None
    ) -> tuple[str, list[str], list[str]]:
        """
        Resolve the label column, feature columns, and categorical columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            label_col (str | None): Name of the label column.
            categorical_cols (list[str] | None): List of categorical column names.

        Returns:
            tuple[str, list[str], list[str]]: The resolved label column, feature
                                              columns, and categorical columns.
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
    def _drop_missing_categoricals(
        df: pd.DataFrame, categorical_cols: list[str]
    ) -> None:
        """
        Drop rows with missing values in categorical columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            categorical_cols (list[str]): List of categorical column names.
        """

        for col in categorical_cols:
            n_missing = int(df[col].isna().sum())

            if n_missing > 0:
                print(f"'{col}': {n_missing} missing values -> dropped")
                df[col] = df[col].dropna()

    @staticmethod
    def _encode_categoricals(
        df: pd.DataFrame, categorical_cols: list[str]
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Encode categorical columns as ordinal indices, and return the encoding map.

        Args:
            df (pd.DataFrame): The input DataFrame.
            categorical_cols (list[str]): List of categorical column names.

        Returns:
            tuple[pd.DataFrame, dict[str, list[str]]]: The encoded DataFrame and the
                                                    encoding map.
        """

        encoded: pd.DataFrame = df.copy()
        encoding_map: dict[str, list[str]] = {}

        for col in categorical_cols:
            categories: list[str] = sorted(df[col].unique().tolist())
            encoding_map[col] = categories
            index_of: dict[str, int] = {v: i for i, v in enumerate(categories)}
            encoded[col] = df[col].map(index_of)

        return encoded, encoding_map

    @staticmethod
    def _encode_label(
        df: pd.DataFrame,
        encoded: pd.DataFrame,
        label_col: str,
        encoding_map: dict[str, list[str]],
    ) -> None:
        """
        Encode the label column as ordinal indices, and update the encoding map.

        Args:
            df (pd.DataFrame): The input DataFrame.
            encoded (pd.DataFrame): The DataFrame with encoded categorical columns.
            label_col (str): The name of the label column.
            encoding_map (dict[str, list[str]]): The encoding map to be updated.
        """

        if is_string_dtype(encoded[label_col]):
            label_categories: list[str] = sorted(df[label_col].unique().tolist())
            encoding_map[label_col] = label_categories
            encoded[label_col] = df[label_col].map(
                {v: i for i, v in enumerate(label_categories)}
            )

    @staticmethod
    def _scale_features(
        encoded: pd.DataFrame, feature_cols: list[str], label_col: str
    ) -> tuple[pd.DataFrame, MinMaxScaler]:
        """
        Scale the feature columns to [0, 1] using MinMaxScaler, and return the
        scaled DataFrame and the fitted scaler.

        Args:
            encoded (pd.DataFrame): The DataFrame with encoded categorical columns.
            feature_cols (list[str]): List of feature column names.
            label_col (str): The name of the label column.

        Returns:
            tuple[pd.DataFrame, MinMaxScaler]: The scaled DataFrame and the fitted
                                               scaler.
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
        Split the full DataFrame into training and testing DataFrames.

        Args:
            full (pd.DataFrame): The full DataFrame.
            label_col (str): The name of the label column.
            test_size (float): The proportion of the dataset to include in the test
                               split.
            seed (int): The random seed.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The training and testing DataFrames.
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
        Write the full, training, and testing DataFrames to CSV files.

        Args:
            out_dir (Path): The output directory to write the CSV files to.
            full (pd.DataFrame): The full DataFrame.
            train_df (pd.DataFrame): The training DataFrame.
            test_df (pd.DataFrame): The testing DataFrame.
        """

        full.to_csv(out_dir / "full.csv", index=True)
        train_df.to_csv(out_dir / "train.csv", index=False)
        test_df.to_csv(out_dir / "test.csv", index=False)

    @staticmethod
    def _write_scaler(out_dir: Path, scaler: MinMaxScaler) -> None:
        """
        Write the fitted MinMaxScaler to a pickle file.

        Args:
            out_dir (Path): The output directory to write the pickle file to.
            scaler (MinMaxScaler): The fitted MinMaxScaler.
        """

        joblib.dump(scaler, out_dir / "scaler.pkl")

    @staticmethod
    def _write_encoding_map(out_dir: Path, encoding_map: dict[str, list[str]]) -> None:
        """
        Write the encoding map to a JSON file.

        Args:
            out_dir (Path): The output directory to write the JSON file to.
            encoding_map (dict[str, list[str]]): The encoding map.
        """

        if encoding_map:
            (out_dir / "encoding_map.json").write_text(json.dumps(encoding_map))

    @staticmethod
    def _write_feature_map(
        out_dir: Path, feature_cols: list[str], label_col: str
    ) -> None:
        """
        Write the feature map to a JSON file.

        Args:
            out_dir (Path): The output directory to write the JSON file to.
            feature_cols (list[str]): The list of feature column names.
            label_col (str): The name of the label column.
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
        Write the details to a CSV file.

        Args:
            out_dir (Path): The output directory to write the CSV file to.
            feature_cols (list[str]): The list of feature column names.
            train_df (pd.DataFrame): The training DataFrame.
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

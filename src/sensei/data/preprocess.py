import numpy as np
import pandas as pd
from numpy._typing._array_like import NDArray

from sensei.spec import Spec

_INTEGER_ROUND_TOLERANCE = 1e-6


class Preprocessor:
    """
    Tier B1 preprocessing: make malformed points unrepresentable rather than
    merely forbidden, by acting on the raw data before it's encoded/scaled.

    Raises:
        ValueError: If a column declared as integer_code_features has non-integral
                    values.
    """

    @staticmethod
    def apply(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
        """
        Apply every declared B1 preprocessing step, in order.

        Args:
            df (pd.DataFrame): The raw input DataFrame to preprocess.
            spec (Spec): The dataset specification containing preprocessing
                         instructions.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with redundant columns dropped and
                          integer-coded features cast to integer dtype.
        """

        df = Preprocessor.drop_fd_redundant_columns(df, spec)
        df = Preprocessor.integer_code_features(df, spec)
        return df

    @staticmethod
    def drop_fd_redundant_columns(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
        """
        Drop columns from the DataFrame that are declared as redundant in the dataset
        specification. This is based on the `drop_fd_redundant_columns` list in the
        `preprocessing` section of the spec.

        Args:
            df (pd.DataFrame): The raw input DataFrame from which to drop redundant
                               columns.
            spec (Spec): The dataset specification containing the list of redundant
                         columns to drop.

        Returns:
            pd.DataFrame: The DataFrame with redundant columns dropped.
        """

        cols: list[str] = list(
            filter(
                lambda c: c in df.columns, spec.preprocessing.drop_fd_redundant_columns
            )
        )
        return df.drop(columns=cols)

    @staticmethod
    def integer_code_features(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
        """
        Round and cast to integer every column declared as an integer-coded feature in
        the dataset specification.

        Args:
            df (pd.DataFrame): The raw input DataFrame containing the features to
                               process.
            spec (Spec): The dataset specification containing the list of integer-coded
                         features.

        Raises:
            ValueError: If a column declared as integer_code_features has non-integral
                        values.

        Returns:
            pd.DataFrame: The DataFrame with integer-coded features rounded and cast to
                          integer dtype.
        """

        out: pd.DataFrame = df.copy()

        for col in spec.preprocessing.integer_code_features:
            if col not in out.columns:
                continue

            values: NDArray[np.float64] = np.asarray(out[col], dtype=float)
            rounded: NDArray[np.float64] = np.round(values)
            bad: NDArray[np.bool] = np.abs(values - rounded) > _INTEGER_ROUND_TOLERANCE

            if bad.any():
                bad_values: NDArray[np.float64] = values[bad][:5]
                raise ValueError(
                    f"'{col}' is declared integer_code_features but has "
                    f"non-integral values (e.g. {bad_values.tolist()}) -- this is "
                    f"exactly the malformed-point case B1 exists to catch; fix the "
                    f"source data or remove '{col}' from the spec, don't round it away"
                )

            out[col] = rounded.astype(int)

        return out

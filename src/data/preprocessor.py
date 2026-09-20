"""Data preprocessing and hygiene pipeline for mouse telemetry."""

from typing import Tuple
import numpy as np
import pandas as pd


class TelemetryPreprocessor:
    """
    Cleans raw mouse telemetry events, removes stationary polling duplicates,
    regularizes timestamps, and handles coordinate outlier boundaries.
    """

    def __init__(self, min_dt: float = 1e-4, iqr_factor: float = 3.0):
        self.min_dt = min_dt
        self.iqr_factor = iqr_factor

    def clean_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute full hygiene pipeline on a raw session DataFrame.

        Steps:
        1. Ensure chronological ordering.
        2. Remove identical consecutive records (stationary polling redundancy).
        3. Harmonize zero or negative time deltas.
        4. Detect extreme coordinate outliers.
        """
        if df.empty or len(df) < 2:
            return df.copy()

        clean_df = df.copy().sort_values("client_timestamp").reset_index(drop=True)

        # 1. Prune consecutive identical coordinates with zero time delta
        dt = clean_df["client_timestamp"].diff().fillna(0)
        dx = clean_df["x"].diff().fillna(0)
        dy = clean_df["y"].diff().fillna(0)
        is_duplicate = (dt == 0) & (dx == 0) & (dy == 0)
        clean_df = clean_df.loc[~is_duplicate].reset_index(drop=True)

        if len(clean_df) < 2:
            return clean_df

        # 2. Regularize inter-event time deltas
        dt = clean_df["client_timestamp"].diff().fillna(0)
        # Harmonize negative or microsecond jitter deltas
        dt_regularized = np.where(dt <= 0, self.min_dt, dt)
        clean_df["dt"] = dt_regularized

        # 3. Flag spatial outliers using the 3*IQR rule from paper Section 2.1
        clean_df = self._flag_spatial_outliers(clean_df)

        return clean_df

    def _flag_spatial_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags non-physical coordinate jumps exceeding 3 * IQR bounds."""
        df["is_outlier"] = False
        if len(df) < 10:
            return df

        for coord in ["x", "y"]:
            q1 = df[coord].quantile(0.25)
            q3 = df[coord].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - self.iqr_factor * iqr
                upper = q3 + self.iqr_factor * iqr
                df["is_outlier"] |= (df[coord] < lower) | (df[coord] > upper)

        return df

"""Data loading utilities for Balabit Mouse Dynamics Challenge dataset."""

import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    CANONICAL_FEATURE_ORDER,
    ENGINEERED_FEATURES_PATH,
    METADATA_COLUMNS,
    PUBLIC_LABELS_PATH,
    RAW_TEST_DIR,
    RAW_TRAIN_DIR,
)


def load_engineered_dataset(
    filepath: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the pre-extracted 25-feature dataset.

    Returns:
        features_df: DataFrame containing the 25 numerical behavioral features.
        metadata_df: DataFrame containing session metadata ('user_id', 'session_id', 'split', 'is_illegal').
    """
    path = Path(filepath) if filepath else ENGINEERED_FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Engineered features file not found at: {path}")

    df = pd.read_csv(path)

    # Ensure all canonical features exist
    missing_cols = [col for col in CANONICAL_FEATURE_ORDER if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols}")

    features_df = df[CANONICAL_FEATURE_ORDER].copy()
    metadata_df = df[METADATA_COLUMNS].copy()

    # Fill any subtle NaNs with median of the column
    features_df = features_df.fillna(features_df.median())

    return features_df, metadata_df


def load_raw_session(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single raw mouse session file into a sanitized DataFrame.

    Expected raw columns:
        'record timestamp', 'client timestamp', 'button', 'state', 'x', 'y'
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Session file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    # Standardize column naming
    rename_map = {
        "record timestamp": "record_timestamp",
        "client timestamp": "client_timestamp",
    }
    df = df.rename(columns=rename_map)

    # Convert numeric fields
    for col in ["record_timestamp", "client_timestamp", "x", "y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["client_timestamp", "x", "y"]).sort_values("client_timestamp").reset_index(drop=True)
    return df


def discover_raw_sessions(
    train_dir: Optional[Union[str, Path]] = None,
    test_dir: Optional[Union[str, Path]] = None,
    labels_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Discover all available raw session files across training and test folders.
    Merges with public_labels.csv where available.
    """
    train_path = Path(train_dir) if train_dir else RAW_TRAIN_DIR
    test_path = Path(test_dir) if test_dir else RAW_TEST_DIR
    lbl_path = Path(labels_path) if labels_path else PUBLIC_LABELS_PATH

    records = []

    # 1. Training sessions (all legal: is_illegal = 0)
    if train_path.exists():
        for user_dir in sorted(train_path.iterdir()):
            if user_dir.is_dir():
                user_id = user_dir.name
                for sfile in sorted(user_dir.iterdir()):
                    if sfile.is_file():
                        records.append({
                            "user_id": user_id,
                            "session_id": sfile.name,
                            "filepath": str(sfile),
                            "split": "train",
                            "is_illegal": 0,
                        })

    # 2. Test sessions with labels
    labels_dict = {}
    if lbl_path.exists():
        labels_df = pd.read_csv(lbl_path)
        labels_dict = dict(zip(labels_df["filename"], labels_df["is_illegal"]))

    if test_path.exists():
        for user_dir in sorted(test_path.iterdir()):
            if user_dir.is_dir():
                user_id = user_dir.name
                for sfile in sorted(user_dir.iterdir()):
                    if sfile.is_file():
                        sname = sfile.name
                        is_illegal = labels_dict.get(sname, -1)  # -1 if unlabelled
                        records.append({
                            "user_id": user_id,
                            "session_id": sname,
                            "filepath": str(sfile),
                            "split": "test",
                            "is_illegal": is_illegal,
                        })

    return pd.DataFrame(records)


def simulate_session_stream(
    filepath: Union[str, Path],
    window_sec: float = 10.0,
    stride_sec: float = 2.0,
) -> Generator[Tuple[float, pd.DataFrame], None, None]:
    """
    Stream sliding temporal windows from a raw session file to emulate
    real-time DOM event processing in a web browser.

    Yields:
        (current_timestamp, window_df): The current elapsed timestamp and slice DataFrame.
    """
    df = load_raw_session(filepath)
    if df.empty:
        return

    t0 = df["client_timestamp"].iloc[0]
    t_end = df["client_timestamp"].iloc[-1]
    curr_t = t0 + window_sec

    while curr_t <= t_end + stride_sec:
        window_mask = (df["client_timestamp"] >= curr_t - window_sec) & (
            df["client_timestamp"] <= curr_t
        )
        window_slice = df.loc[window_mask].copy()
        yield (curr_t - t0, window_slice)
        curr_t += stride_sec

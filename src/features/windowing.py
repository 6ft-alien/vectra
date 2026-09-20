"""Sliding window segmentation engine for continuous behavioral biometrics."""

from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.extractor import FeatureExtractor


class SlidingWindowSegmenter:
    """
    Partitions continuous mouse trajectory event streams into overlapping
    evaluation windows for real-time Continuous Authentication.
    """

    def __init__(
        self,
        window_size_sec: float = 10.0,
        stride_sec: float = 2.0,
        min_events: int = 10,
    ):
        self.window_size_sec = window_size_sec
        self.stride_sec = stride_sec
        self.min_events = min_events
        self.extractor = FeatureExtractor()

    def segment_session(
        self, df: pd.DataFrame, max_duration_sec: Optional[float] = None
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Segment a session DataFrame into sliding windows.

        Parameters:
            df: Session DataFrame.
            max_duration_sec: Optional maximum elapsed time in seconds to segment.

        Returns:
            List of tuples: (window_start_sec, window_end_sec, feature_vector).
        """
        if df.empty or len(df) < self.min_events:
            return []

        t = df["client_timestamp"].values.astype(float)
        t0 = t[0]
        t_limit = t0 + max_duration_sec if max_duration_sec is not None else t[-1]
        t_max = min(t[-1], t_limit)

        windows = []
        curr_start = t0

        while curr_start + self.window_size_sec <= t_max + self.stride_sec:
            curr_end = curr_start + self.window_size_sec
            idx_start = int(np.searchsorted(t, curr_start, side="left"))
            idx_end = int(np.searchsorted(t, curr_end, side="right"))

            if (idx_end - idx_start) >= self.min_events:
                w_df = df.iloc[idx_start:idx_end]
                feat_vec = self.extractor.extract_vector(w_df)
                windows.append((curr_start - t0, curr_end - t0, feat_vec))

            curr_start += self.stride_sec

        return windows

    def stream_windows(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[float, float, np.ndarray], None, None]:
        """
        Generator yielding sliding windows one by one to emulate live continuous ingestion.
        """
        for win in self.segment_session(df):
            yield win

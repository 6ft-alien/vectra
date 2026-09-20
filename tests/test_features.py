"""Unit tests for feature extraction and windowing."""

import unittest
import numpy as np
import pandas as pd

from src.config import CANONICAL_FEATURE_ORDER
from src.features.extractor import FeatureExtractor
from src.features.windowing import SlidingWindowSegmenter


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_synthetic_trajectory_extraction(self):
        # Create a synthetic 10-second trajectory (100 Hz, 1000 events)
        t = np.linspace(0, 10, 1000)
        x = 500 + 100 * np.cos(t * 2)
        y = 500 + 100 * np.sin(t * 2)
        buttons = ["NoButton"] * 950 + ["Left"] * 50
        states = ["Move"] * 900 + ["Pressed"] * 50 + ["Drag"] * 50

        df = pd.DataFrame({
            "client_timestamp": t,
            "x": x,
            "y": y,
            "button": buttons,
            "state": states,
        })

        feats = self.extractor.extract_features_from_dataframe(df)

        # 1. Assert all 25 features are present
        for col in CANONICAL_FEATURE_ORDER:
            self.assertIn(col, feats, f"Missing feature: {col}")
            self.assertFalse(np.isnan(feats[col]), f"NaN encountered in {col}")
            self.assertFalse(np.isinf(feats[col]), f"Inf encountered in {col}")

        # 2. Check logical properties
        self.assertEqual(feats["num_events"], 1000)
        self.assertAlmostEqual(feats["duration"], 10.0, places=1)
        self.assertGreater(feats["total_distance"], 0.0)
        self.assertGreater(feats["avg_velocity"], 0.0)
        self.assertGreaterEqual(feats["click_count"], 0)
        self.assertGreaterEqual(feats["drag_ratio"], 0.0)

    def test_degenerated_single_event(self):
        df = pd.DataFrame({
            "client_timestamp": [1.0],
            "x": [100.0],
            "y": [200.0],
            "button": ["NoButton"],
            "state": ["Move"],
        })
        feats = self.extractor.extract_features_from_dataframe(df)
        self.assertEqual(feats["num_events"], 1.0)
        for col in CANONICAL_FEATURE_ORDER:
            self.assertIn(col, feats)

    def test_sliding_window_segmenter(self):
        t = np.linspace(0, 25, 2500)
        x = np.linspace(0, 500, 2500)
        y = np.linspace(0, 500, 2500)
        df = pd.DataFrame({"client_timestamp": t, "x": x, "y": y, "button": ["NoButton"] * 2500, "state": ["Move"] * 2500})

        segmenter = SlidingWindowSegmenter(window_size_sec=10.0, stride_sec=2.0, min_events=10)
        windows = segmenter.segment_session(df)

        self.assertGreater(len(windows), 5)
        for start_t, end_t, vec in windows:
            self.assertAlmostEqual(end_t - start_t, 10.0, places=1)
            self.assertEqual(len(vec), 25)


if __name__ == "__main__":
    unittest.main()

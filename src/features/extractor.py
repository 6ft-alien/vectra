"""Behavioral Biometrics Feature Extraction Engine.

Extracts the 25 kinematic, angular, cadence, and spatial features specified in:
'Behavioral Biometrics for Zero-Trust Web Architectures: Continuous Authentication via Cursor Dynamics'
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import CANONICAL_FEATURE_ORDER


class FeatureExtractor:
    """
    Extracts a 25-dimensional behavioral biometric feature vector from
    preprocessed cursor trajectory events.
    """

    def __init__(self, clip_straightness: float = 50.0, pause_time_thresh: float = 0.5, pause_dist_thresh: float = 5.0):
        self.clip_straightness = clip_straightness
        self.pause_time_thresh = pause_time_thresh
        self.pause_dist_thresh = pause_dist_thresh

    def extract_features_from_dataframe(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract the 25 canonical behavioral features from a session or window slice DataFrame.

        Parameters:
            df: DataFrame containing at least ['client_timestamp', 'x', 'y'].
                Optional columns: ['button', 'state'].

        Returns:
            Dict mapping feature names to numerical values.
        """
        feats: Dict[str, float] = {}

        num_events = len(df)
        feats["num_events"] = float(num_events)

        if num_events < 2:
            # Fallback zero-filled vector for degenerated inputs
            for k in CANONICAL_FEATURE_ORDER:
                if k not in feats:
                    feats[k] = 0.0
            return feats

        # Temporal arrays
        t = df["client_timestamp"].values.astype(float)
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)

        duration = max(t[-1] - t[0], 1e-5)
        feats["duration"] = duration

        # Spatial deltas
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)
        dt = np.clip(dt, 1e-5, None)

        dist = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(dist)
        feats["total_distance"] = float(total_distance)

        # 1. Kinematics: Velocity
        velocity = dist / dt
        feats["avg_velocity"] = float(np.mean(velocity))
        feats["std_velocity"] = float(np.std(velocity))
        # 99th percentile or max
        feats["max_velocity"] = float(np.percentile(velocity, 99) if len(velocity) >= 100 else np.max(velocity))
        feats["median_velocity"] = float(np.median(velocity))

        # 2. Kinematics: Acceleration
        if len(velocity) > 1:
            dt_acc = dt[1:]
            dt_acc = np.clip(dt_acc, 1e-5, None)
            dvel = np.diff(velocity)
            acc = np.abs(dvel / dt_acc)
            feats["avg_acceleration"] = float(np.mean(acc))
            feats["std_acceleration"] = float(np.std(acc))
        else:
            feats["avg_acceleration"] = 0.0
            feats["std_acceleration"] = 0.0

        # 3. Angular Dynamics & Curvature
        angles = np.arctan2(dy, dx)
        if len(angles) > 1:
            dtheta = np.diff(angles)
            # Wrap to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            abs_dtheta = np.abs(dtheta)

            dt_ang = dt[1:]
            dt_ang = np.clip(dt_ang, 1e-5, None)
            ang_vel = abs_dtheta / dt_ang
            feats["avg_angular_velocity"] = float(np.mean(ang_vel))

            dist_curv = dist[1:] + 1e-5
            curvature = abs_dtheta / dist_curv
            feats["avg_curvature"] = float(np.mean(curvature))
        else:
            feats["avg_angular_velocity"] = 0.0
            feats["avg_curvature"] = 0.0

        # 4. Interaction Cadence: Clicks & Drags
        button_col = "button" if "button" in df.columns else None
        state_col = "state" if "state" in df.columns else None

        click_count = 0
        drag_count = 0

        if state_col:
            states = df[state_col].astype(str).str.lower()
            click_count = int(states.isin(["pressed", "down"]).sum())
            drag_count = int(states.isin(["drag"]).sum())
        elif button_col:
            buttons = df[button_col].astype(str)
            is_active = (buttons != "NoButton") & (buttons != "none")
            click_count = int(np.sum(np.diff(is_active.astype(int)) == 1))
            drag_count = int(is_active.sum())

        feats["click_count"] = float(click_count)
        feats["click_rate"] = float(click_count / duration)
        feats["drag_count"] = float(drag_count)
        feats["drag_ratio"] = float(drag_count / num_events)

        # 5. Cognitive Hesitation & Pauses
        # Pause defined as dt > 0.5s and dist < 5.0px
        is_pause = (dt > self.pause_time_thresh) & (dist < self.pause_dist_thresh)
        pause_count = int(np.sum(is_pause))
        feats["pause_count"] = float(pause_count)
        feats["pause_ratio"] = float(pause_count / num_events)

        # 6. Spatial Geometry & Screen Footprint
        feats["x_mean"] = float(np.mean(x))
        feats["y_mean"] = float(np.mean(y))
        feats["x_std"] = float(np.std(x))
        feats["y_std"] = float(np.std(y))
        feats["x_range"] = float(np.ptp(x))
        feats["y_range"] = float(np.ptp(y))

        # Path straightness ratio: total cumulative path / direct Euclidean displacement
        direct_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        straightness = total_distance / (direct_dist + 1e-5)
        feats["straightness"] = float(np.clip(straightness, 1.0, self.clip_straightness))

        feats["avg_sampling_rate"] = float(num_events / duration)

        return feats

    def extract_vector(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features as a 1D NumPy array in canonical order."""
        feats = self.extract_features_from_dataframe(df)
        return np.array([feats[k] for k in CANONICAL_FEATURE_ORDER], dtype=np.float32)

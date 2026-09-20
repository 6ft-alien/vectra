"""Global configuration settings for the Behavioral Biometrics Continuous Authentication framework."""

import os
from pathlib import Path

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN_DIR = PROJECT_ROOT / "training_files"
RAW_TEST_DIR = PROJECT_ROOT / "test_files"
PUBLIC_LABELS_PATH = PROJECT_ROOT / "public_labels.csv"
ENGINEERED_FEATURES_PATH = DATA_DIR / "engineered_features.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# 25 Core Behavioral Biometric Feature Dimensions
KINEMATIC_FEATURES = [
    "avg_velocity",
    "std_velocity",
    "max_velocity",
    "median_velocity",
    "avg_acceleration",
    "std_acceleration",
]

ANGULAR_FEATURES = [
    "avg_angular_velocity",
    "avg_curvature",
]

CADENCE_FEATURES = [
    "click_count",
    "click_rate",
    "drag_count",
    "drag_ratio",
    "pause_count",
    "pause_ratio",
]

SPATIAL_FEATURES = [
    "total_distance",
    "straightness",
    "x_mean",
    "y_mean",
    "x_std",
    "y_std",
    "x_range",
    "y_range",
    "avg_sampling_rate",
]

TEMPORAL_FEATURES = [
    "num_events",
    "duration",
]

# Ensure deterministic order matching engineered_features.csv
CANONICAL_FEATURE_ORDER = [
    "num_events",
    "duration",
    "total_distance",
    "avg_velocity",
    "std_velocity",
    "max_velocity",
    "median_velocity",
    "avg_acceleration",
    "std_acceleration",
    "avg_angular_velocity",
    "avg_curvature",
    "click_count",
    "click_rate",
    "drag_count",
    "drag_ratio",
    "pause_count",
    "pause_ratio",
    "x_mean",
    "y_mean",
    "x_std",
    "y_std",
    "x_range",
    "y_range",
    "straightness",
    "avg_sampling_rate",
]

METADATA_COLUMNS = ["user_id", "session_id", "split", "is_illegal"]

# Target Users in Balabit Dataset
USER_IDS = [
    "user7",
    "user9",
    "user12",
    "user15",
    "user16",
    "user20",
    "user21",
    "user23",
    "user29",
    "user35",
]

# Zero-Trust Policy Configuration
ZERO_TRUST_CONFIG = {
    # Risk Score Thresholds: R_t in [0.0, 1.0]
    "tau_low": 0.35,      # Normal operation: token seamlessly refreshed
    "tau_high": 0.70,     # Severe anomaly: instant token revocation & session termination
    # Moderate risk range: [tau_low, tau_high) triggers Step-Up MFA Challenge
    "ema_alpha": 0.35,    # Exponential moving average smoothing factor
    "window_size_sec": 10.0,
    "window_stride_sec": 2.0,
    "min_events_per_window": 15,
}

# Experiment Reproducibility
RANDOM_STATE = 42

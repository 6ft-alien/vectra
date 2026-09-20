"""Evaluation metrics and plotting subpackage."""

from src.evaluation.metrics import calculate_eer, evaluate_predictions
from src.evaluation.plots import (
    plot_roc_curves,
    plot_det_curves,
    plot_feature_importances,
    plot_continuous_session_timeline,
)

__all__ = [
    "calculate_eer",
    "evaluate_predictions",
    "plot_roc_curves",
    "plot_det_curves",
    "plot_feature_importances",
    "plot_continuous_session_timeline",
]

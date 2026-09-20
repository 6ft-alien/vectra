"""Biometric verification and classification evaluation metrics."""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculate the Equal Error Rate (EER) and the optimal decision threshold.

    EER is the operating point on the ROC curve where:
    False Acceptance Rate (FAR) == False Rejection Rate (FRR).

    Returns:
        eer: The Equal Error Rate (float between 0.0 and 1.0).
        optimal_threshold: The decision threshold yielding the EER.
        fpr: False Positive Rate (FAR) curve.
        fnr: False Negative Rate (FRR) curve (1 - TPR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Avoid degenerated cases
    if len(fpr) < 2:
        return 0.5, 0.5, fpr, fnr

    # Find the intersection of FPR and FNR
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        optimal_threshold = float(interp1d(fpr, thresholds)(eer))
    except Exception:
        # Fallback: discrete minimum absolute difference
        diff = np.abs(fpr - fnr)
        idx = np.argmin(diff)
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        optimal_threshold = float(thresholds[idx])

    return float(eer), float(optimal_threshold), fpr, fnr


def evaluate_predictions(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, Union[float, int]]:
    """
    Compute comprehensive biometric evaluation metrics given ground truth and predicted probabilities.
    """
    y_pred = (y_scores >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_scores))
    eer, optimal_thresh, _, _ = calculate_eer(y_true, y_scores)

    # Predictions at optimal EER threshold
    y_pred_eer = (y_scores >= optimal_thresh).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "roc_auc": auc,
        "eer": eer,
        "optimal_threshold": optimal_thresh,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_at_eer": float(accuracy_score(y_true, y_pred_eer)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": far,
        "frr": frr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

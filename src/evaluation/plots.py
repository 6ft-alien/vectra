"""Publication-quality plotting utilities for behavioral biometrics research."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

from src.config import FIGURES_DIR

# Modern academic styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_roc_curves(
    curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot comparative ROC curves for multiple models.

    Parameters:
        curves_dict: Dict mapping model_name -> (fpr, tpr, roc_auc)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("deep", len(curves_dict))

    for idx, (name, (fpr, tpr, auc)) in enumerate(curves_dict.items()):
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc:.3f})",
            color=colors[idx],
            linewidth=2.0,
        )

    # Reference random guess line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.2, label="Random Guess (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FAR / 1 - Specificity)")
    ax.set_ylabel("True Positive Rate (1 - FRR / Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    out_path = Path(save_path) if save_path else FIGURES_DIR / "roc_curves_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_det_curves(
    det_dict: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot Detection Error Tradeoff (DET) curves: FAR vs FRR.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set2", len(det_dict))

    for idx, (name, (far, frr, eer)) in enumerate(det_dict.items()):
        ax.plot(
            far * 100,
            frr * 100,
            label=f"{name} (EER = {eer*100:.1f}%)",
            color=colors[idx],
            linewidth=2.0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([0.5, 100.0])
    ax.set_ylim([0.5, 100.0])
    ax.set_xlabel("False Acceptance Rate (FAR, %)")
    ax.set_ylabel("False Rejection Rate (FRR, %)")
    ax.set_title("Detection Error Tradeoff (DET) Curve")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    fig.tight_layout()

    out_path = Path(save_path) if save_path else FIGURES_DIR / "det_curves_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Top Behavioral Biometric Feature Importances",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of top predictive features."""
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices][::-1]
    top_scores = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_names, top_scores, color="#2b5c8f", alpha=0.85, edgecolor="#1a3b5c")

    ax.set_xlabel("Gini Feature Importance / Weight")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    # Annotate values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    fig.tight_layout()

    out_path = Path(save_path) if save_path else FIGURES_DIR / "feature_importances.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_continuous_session_timeline(
    timestamps: np.ndarray,
    instant_scores: np.ndarray,
    ema_scores: np.ndarray,
    hijack_time: float,
    tau_low: float = 0.35,
    tau_high: float = 0.70,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot real-time dynamic risk scoring over time, highlighting normal,
    step-up challenge, and instantaneous token revocation regions.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Zone shading
    ax.axhspan(0.0, tau_low, color="#e8f5e9", alpha=0.8, label="Low Risk (Token Refreshed)")
    ax.axhspan(tau_low, tau_high, color="#fff9c4", alpha=0.8, label="Moderate Risk (Step-Up MFA Challenge)")
    ax.axhspan(tau_high, 1.0, color="#ffebee", alpha=0.8, label="Severe Anomaly (Token Revocation)")

    # Instantaneous prediction dots
    ax.scatter(timestamps, instant_scores, color="#78909c", alpha=0.5, s=25, label="Window Prediction P(imposter)")

    # Continuous smoothed EMA risk line
    ax.plot(timestamps, ema_scores, color="#c62828", linewidth=2.8, label="Zero-Trust Dynamic Risk Score R(t)")

    # Hijack event vertical line
    ax.axvline(x=hijack_time, color="#212121", linestyle="--", linewidth=2.0, label=f"Session Hijacked (t = {hijack_time:.0f}s)")

    # Threshold horizontal reference lines
    ax.axhline(y=tau_low, color="#2e7d32", linestyle=":", linewidth=1.5)
    ax.axhline(y=tau_high, color="#c62828", linestyle=":", linewidth=1.5)

    ax.set_xlim([timestamps[0], timestamps[-1]])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Session Elapsed Time (seconds)")
    ax.set_ylabel("Imposter Risk Probability R(t)")
    ax.set_title("Continuous Behavioral Biometrics: Real-Time Zero-Trust Verification Timeline")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()

    out_path = Path(save_path) if save_path else FIGURES_DIR / "session_hijacking_timeline.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig

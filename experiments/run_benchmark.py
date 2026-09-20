"""Comprehensive Model Benchmark on Balabit Mouse Dynamics Challenge Dataset."""

import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CANONICAL_FEATURE_ORDER,
    FIGURES_DIR,
    RANDOM_STATE,
    TABLES_DIR,
)
from src.data.loader import load_engineered_dataset
from src.evaluation.metrics import calculate_eer, evaluate_predictions
from src.evaluation.plots import plot_det_curves, plot_feature_importances, plot_roc_curves
from src.models.anomaly_detector import IsolationForestAnomalyDetector, OneClassSVMAnomalyDetector
from src.models.baselines import get_baseline_models
from src.models.mlp import build_mlp_pipeline


def run_full_benchmark() -> pd.DataFrame:
    print("=" * 80)
    print("BEHAVIORAL BIOMETRICS ZERO-TRUST CONTINUOUS AUTHENTICATION BENCHMARK")
    print("Dataset: Balabit Mouse Dynamics Challenge (25-Dimensional Kinematic Features)")
    print("=" * 80)

    # 1. Load engineered feature dataset
    features_df, metadata_df = load_engineered_dataset()
    X = features_df.values
    y = metadata_df["is_illegal"].values.astype(int)

    print(f"\n[*] Total samples loaded: {len(X)} sessions across {metadata_df['user_id'].nunique()} user cohorts.")
    print(f"[*] Class distribution: Legitimate = {np.sum(y == 0)} ({np.mean(y==0)*100:.1f}%), Imposter = {np.sum(y == 1)} ({np.mean(y==1)*100:.1f}%)")

    # 2. Assemble candidate model suite
    models = get_baseline_models(random_state=RANDOM_STATE)
    models["Multi-Layer Perceptron (MLP)"] = build_mlp_pipeline(random_state=RANDOM_STATE)
    models["One-Class SVM (Unsupervised)"] = OneClassSVMAnomalyDetector(nu=0.1)
    models["Isolation Forest (Unsupervised)"] = IsolationForestAnomalyDetector(random_state=RANDOM_STATE)

    # 3. Cross-Validation setup (Stratified 5-Fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results_records: List[Dict] = []
    roc_curves_dict: Dict = {}
    det_curves_dict: Dict = {}

    rf_feature_importances = None

    print("\n" + "-" * 80)
    print(f"{'Model Architecture':<30} | {'ROC-AUC':<8} | {'EER (%)':<8} | {'Acc (%)':<8} | {'F1-Score':<8} | {'Latency':<8}")
    print("-" * 80)

    for model_name, model in models.items():
        t0 = time.time()
        oof_scores = np.zeros(len(y))

        # Perform 5-fold cross-validation
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Special case for unsupervised one-class detectors: train ONLY on legitimate baseline (y == 0)
            if "One-Class" in model_name or "Isolation Forest" in model_name:
                legit_mask = (y_train == 0)
                model.fit(X_train[legit_mask])
            else:
                model.fit(X_train, y_train)

            # Predict probabilities
            probs = model.predict_proba(X_val)
            # Probability of illegal (class 1)
            oof_scores[val_idx] = probs[:, 1]

        elapsed_time = time.time() - t0

        # Compute full biometric metrics
        metrics = evaluate_predictions(y, oof_scores)
        eer, opt_thresh, fpr, fnr = calculate_eer(y, oof_scores)
        roc_fpr, roc_tpr, _ = roc_curve(y, oof_scores)

        roc_curves_dict[model_name] = (roc_fpr, roc_tpr, metrics["roc_auc"])
        det_curves_dict[model_name] = (fpr, fnr, eer)

        # Store for table
        row = {
            "Model": model_name,
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "EER (%)": f"{eer * 100:.2f}%",
            "Accuracy (%)": f"{metrics['accuracy'] * 100:.2f}%",
            "Accuracy at EER (%)": f"{metrics['accuracy_at_eer'] * 100:.2f}%",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1-Score": f"{metrics['f1_score']:.4f}",
            "FAR (%)": f"{metrics['far'] * 100:.2f}%",
            "FRR (%)": f"{metrics['frr'] * 100:.2f}%",
            "Training Time (s)": f"{elapsed_time:.2f}",
        }
        results_records.append(row)

        print(
            f"{model_name:<30} | {metrics['roc_auc']:<8.4f} | {eer*100:<7.2f}% | "
            f"{metrics['accuracy']*100:<7.2f}% | {metrics['f1_score']:<8.4f} | {elapsed_time:<7.2f}s"
        )

        # Grab feature importances from Random Forest for analysis
        if model_name == "Random Forest":
            rf_model = models["Random Forest"]
            rf_model.fit(X, y)
            rf_feature_importances = rf_model.feature_importances_

    print("-" * 80)

    # 4. Save results to DataFrame
    results_df = pd.DataFrame(results_records)
    csv_path = TABLES_DIR / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved benchmark metrics table to: {csv_path}")

    # Export LaTeX table
    latex_path = TABLES_DIR / "benchmark_results.tex"
    with open(latex_path, "w") as f:
        f.write(results_df.to_latex(index=False, caption="Comparative Performance of Machine Learning Architectures for Continuous Mouse Dynamics Authentication.", label="tab:model_benchmark"))
    print(f"[+] Saved LaTeX table to: {latex_path}")

    # 5. Generate and export publication figures
    print("[*] Generating publication figures...")
    plot_roc_curves(roc_curves_dict, save_path=FIGURES_DIR / "roc_curves_comparison.png")
    plot_det_curves(det_curves_dict, save_path=FIGURES_DIR / "det_curves_comparison.png")

    if rf_feature_importances is not None:
        plot_feature_importances(
            feature_names=CANONICAL_FEATURE_ORDER,
            importances=rf_feature_importances,
            top_n=15,
            save_path=FIGURES_DIR / "feature_importances.png",
        )

    print(f"[+] Saved ROC Curves to: {FIGURES_DIR / 'roc_curves_comparison.png'}")
    print(f"[+] Saved DET Curves to: {FIGURES_DIR / 'det_curves_comparison.png'}")
    print(f"[+] Saved Feature Importances to: {FIGURES_DIR / 'feature_importances.png'}")

    return results_df


if __name__ == "__main__":
    run_full_benchmark()

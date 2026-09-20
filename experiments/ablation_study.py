"""Feature Subset Ablation Study for Behavioral Biometrics."""

from typing import Dict, List
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.config import (
    ANGULAR_FEATURES,
    CADENCE_FEATURES,
    KINEMATIC_FEATURES,
    RANDOM_STATE,
    SPATIAL_FEATURES,
    TABLES_DIR,
)
from src.data.loader import load_engineered_dataset
from src.evaluation.metrics import calculate_eer
from src.models.baselines import get_baseline_models


def run_ablation_study() -> pd.DataFrame:
    print("=" * 80)
    print("FEATURE SUBSET ABLATION STUDY: BEHAVIORAL BIOMETRICS")
    print("Evaluating informational synergy across kinematic, angular, cadence, and spatial dimensions.")
    print("=" * 80)

    features_df, metadata_df = load_engineered_dataset()
    y = metadata_df["is_illegal"].values.astype(int)

    feature_subsets = {
        "Kinematic Only (Vel/Acc)": KINEMATIC_FEATURES,
        "Angular Only (Curv/AngVel)": ANGULAR_FEATURES,
        "Cadence & Pauses Only": CADENCE_FEATURES,
        "Spatial Geometry Only": [f for f in SPATIAL_FEATURES if f in features_df.columns],
        "Kinematic + Cadence": KINEMATIC_FEATURES + CADENCE_FEATURES,
        "All 25 Features (Proposed)": features_df.columns.tolist(),
    }

    results = []

    print("\n" + "-" * 75)
    print(f"{'Feature Subset':<32} | {'Dimensions':<12} | {'ROC-AUC':<10} | {'Std Dev':<8}")
    print("-" * 75)

    for subset_name, col_list in feature_subsets.items():
        valid_cols = [c for c in col_list if c in features_df.columns]
        X_sub = features_df[valid_cols].values

        clf = get_baseline_models(random_state=RANDOM_STATE)["Random Forest"]
        scores = cross_val_score(clf, X_sub, y, cv=5, scoring="roc_auc", n_jobs=-1)

        mean_auc = float(scores.mean())
        std_auc = float(scores.std())

        results.append({
            "Feature Subset": subset_name,
            "Dimensions": len(valid_cols),
            "Mean ROC-AUC": f"{mean_auc:.4f}",
            "Std Dev": f"{std_auc:.4f}",
        })

        print(f"{subset_name:<32} | {len(valid_cols):<12} | {mean_auc:<10.4f} | +/- {std_auc:.4f}")

    print("-" * 75)

    df_out = pd.DataFrame(results)
    out_csv = TABLES_DIR / "ablation_study_results.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[+] Saved ablation study results to: {out_csv}")
    return df_out


if __name__ == "__main__":
    run_ablation_study()

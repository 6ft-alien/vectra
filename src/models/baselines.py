"""Supervised baseline machine learning classifiers for Behavioral Biometrics."""

from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RANDOM_STATE


def get_baseline_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """
    Build and return standard supervised classification pipelines.
    Models requiring feature normalization are wrapped in a StandardScaler pipeline.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Machine (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(
                    C=2.0,
                    kernel="rbf",
                    gamma="scale",
                    random_state=random_state,
                ),
                ensemble=False,
            )),
        ]),
        "Gradient Boosting (GBDT)": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=random_state,
        ),
        "k-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski",
            )),
        ]),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
            )),
        ]),
    }
    return models

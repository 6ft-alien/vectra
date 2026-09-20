"""Unsupervised One-Class Anomaly Detection for Zero-Trust Baseline Enrollment."""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import RANDOM_STATE


class OneClassSVMAnomalyDetector(BaseEstimator, ClassifierMixin):
    """
    One-Class SVM for passive behavioral anomaly detection.
    Trained strictly on legitimate owner baseline sessions.
    Outputs normalized risk scores in [0.0, 1.0].
    """

    def __init__(self, nu: float = 0.08, gamma: str = "scale"):
        self.nu = nu
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.model = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel="rbf")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OneClassSVMAnomalyDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Convert decision function distance to an anomaly risk probability:
        High positive distance -> authentic owner (risk ~ 0).
        High negative distance -> imposter (risk ~ 1).
        """
        scores = self.decision_function(X)
        # Logistic sigmoid inversion
        risk_scores = 1.0 / (1.0 + np.exp(np.clip(scores * 2.0, -15.0, 15.0)))
        return np.column_stack([1.0 - risk_scores, risk_scores])

    def predict(self, X: np.ndarray) -> np.ndarray:
        risk = self.predict_proba(X)[:, 1]
        return (risk >= 0.5).astype(int)


class IsolationForestAnomalyDetector(BaseEstimator, ClassifierMixin):
    """
    Isolation Forest for high-dimensional cursor trajectory anomaly detection.
    """

    def __init__(self, contamination: float = 0.08, n_estimators: int = 150, random_state: int = RANDOM_STATE):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestAnomalyDetector":
        self.model.fit(X)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # score_samples: more negative means more anomalous
        scores = self.model.score_samples(X)
        # Shift and scale to [0, 1]
        # In sklearn, typical score is in [-0.8, -0.3]
        norm_scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-5)
        risk_scores = 1.0 / (1.0 + np.exp(np.clip(norm_scores * 2.0, -15.0, 15.0)))
        return np.column_stack([1.0 - risk_scores, risk_scores])

    def predict(self, X: np.ndarray) -> np.ndarray:
        risk = self.predict_proba(X)[:, 1]
        return (risk >= 0.5).astype(int)

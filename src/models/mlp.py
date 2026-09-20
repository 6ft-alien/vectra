"""Multi-Layer Perceptron (MLP) Neural Network for Continuous Authentication."""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


def build_mlp_pipeline(
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    activation: str = "relu",
    learning_rate_init: float = 0.003,
    alpha: float = 1e-4,
    max_iter: int = 500,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Constructs a standardized, high-performance MLP neural network pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.15,
            random_state=random_state,
        )),
    ])


class ScratchMLP(BaseEstimator, ClassifierMixin):
    """
    Multi-Layer Perceptron implemented from scratch with NumPy.
    Provides complete architectural transparency for scientific inquiry.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 0.01,
        epochs: int = 200,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.weights_: List[np.ndarray] = []
        self.biases_: List[np.ndarray] = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def _relu_deriv(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchMLP":
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # Xavier/He initialization
        w1 = rng.randn(n_features, self.hidden_dim) * np.sqrt(2.0 / n_features)
        b1 = np.zeros((1, self.hidden_dim))
        w2 = rng.randn(self.hidden_dim, 1) * np.sqrt(1.0 / self.hidden_dim)
        b2 = np.zeros((1, 1))

        y_col = y.reshape(-1, 1).astype(float)

        for _ in range(self.epochs):
            # Forward pass
            z1 = np.dot(X, w1) + b1
            a1 = self._relu(z1)
            z2 = np.dot(a1, w2) + b2
            y_hat = self._sigmoid(z2)

            # Backpropagation
            dz2 = (y_hat - y_col) / n_samples
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, w2.T)
            dz1 = da1 * self._relu_deriv(z1)
            dw1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Gradient descent update
            w1 -= self.lr * dw1
            b1 -= self.lr * db1
            w2 -= self.lr * dw2
            b2 -= self.lr * db2

        self.weights_ = [w1, w2]
        self.biases_ = [b1, b2]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z1 = np.dot(X, self.weights_[0]) + self.biases_[0]
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.weights_[1]) + self.biases_[1]
        prob_1 = self._sigmoid(z2).flatten()
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)

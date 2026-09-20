"""Unit tests for models and evaluation metrics."""

import unittest
import numpy as np

from src.evaluation.metrics import calculate_eer, evaluate_predictions
from src.models.anomaly_detector import IsolationForestAnomalyDetector, OneClassSVMAnomalyDetector
from src.models.baselines import get_baseline_models
from src.models.mlp import build_mlp_pipeline, ScratchMLP


class TestModelsAndMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic 50-sample binary problem with 25 features
        self.X = np.random.randn(60, 25)
        self.y = np.array([0] * 30 + [1] * 30)

    def test_baseline_models_fit_and_predict(self):
        models = get_baseline_models(random_state=42)
        for name, model in models.items():
            model.fit(self.X, self.y)
            probs = model.predict_proba(self.X)
            self.assertEqual(probs.shape, (60, 2))
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_mlp_models(self):
        # Sklearn MLP
        mlp_pipe = build_mlp_pipeline(hidden_layer_sizes=(16,), max_iter=20, random_state=42)
        mlp_pipe.fit(self.X, self.y)
        probs = mlp_pipe.predict_proba(self.X)
        self.assertEqual(probs.shape, (60, 2))

        # Scratch MLP
        scratch = ScratchMLP(hidden_dim=8, epochs=10, random_state=42)
        scratch.fit(self.X, self.y)
        probs_s = scratch.predict_proba(self.X)
        self.assertEqual(probs_s.shape, (60, 2))

    def test_anomaly_detectors(self):
        # Trained only on legitimate baseline (class 0)
        X_legit = self.X[self.y == 0]
        oc_svm = OneClassSVMAnomalyDetector(nu=0.1)
        oc_svm.fit(X_legit)
        probs_svm = oc_svm.predict_proba(self.X)
        self.assertEqual(probs_svm.shape, (60, 2))

        iforest = IsolationForestAnomalyDetector(random_state=42)
        iforest.fit(X_legit)
        probs_if = iforest.predict_proba(self.X)
        self.assertEqual(probs_if.shape, (60, 2))

    def test_eer_and_evaluation_metrics(self):
        # Perfect predictions
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
        metrics = evaluate_predictions(y_true, y_scores)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["eer"], 0.0)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

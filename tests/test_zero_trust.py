"""Unit tests for Zero-Trust Risk Engine and Token Lifecycle Manager."""

import unittest
from src.zero_trust.risk_engine import PolicyAction, ZeroTrustRiskEngine
from src.zero_trust.token_manager import TokenState, ZeroTrustTokenManager


class TestZeroTrustComponents(unittest.TestCase):

    def setUp(self):
        self.engine = ZeroTrustRiskEngine(tau_low=0.35, tau_high=0.70, ema_alpha=0.4)
        self.token_mgr = ZeroTrustTokenManager()

    def test_risk_engine_normal_session(self):
        # Stream of benign low-risk probabilities
        for t in range(5):
            score, action = self.engine.update(timestamp_sec=float(t * 2), instant_imposter_prob=0.10)
            self.assertLess(score, 0.35)
            self.assertEqual(action, PolicyAction.ALLOW_REFRESH)
        self.assertFalse(self.engine.is_session_revoked)

    def test_risk_engine_hijacking_transition(self):
        # 1. Normal period
        for t in range(3):
            self.engine.update(timestamp_sec=float(t * 2), instant_imposter_prob=0.08)

        # 2. Moderate risk anomaly
        # With ema_alpha=0.4 and baseline risk=0.08, sending 0.80 yields 0.368 (in [0.35, 0.70))
        score_mod, action_mod = self.engine.update(timestamp_sec=8.0, instant_imposter_prob=0.80)
        self.assertEqual(action_mod, PolicyAction.STEP_UP_MFA)

        # 3. Severe attack spikes (consecutive windows push smoothed EMA above tau_high = 0.70)
        self.engine.update(timestamp_sec=10.0, instant_imposter_prob=0.98)
        score_high, action_high = self.engine.update(timestamp_sec=12.0, instant_imposter_prob=0.98)
        self.assertEqual(action_high, PolicyAction.REVOKE_TOKEN)
        self.assertTrue(self.engine.is_session_revoked)

    def test_token_lifecycle(self):
        token = self.token_mgr.issue_token(user_id="user7", session_id="test_sess_1")
        is_valid, _ = self.token_mgr.verify_token(token)
        self.assertTrue(is_valid)

        # Process benign window
        action, _ = self.token_mgr.process_telemetry_window(token, 2.0, 0.1)
        self.assertEqual(action, PolicyAction.ALLOW_REFRESH)
        is_valid, _ = self.token_mgr.verify_token(token)
        self.assertTrue(is_valid)

        # Process consecutive attack spikes until EMA risk crosses 0.70
        self.token_mgr.process_telemetry_window(token, 4.0, 0.95)
        self.token_mgr.process_telemetry_window(token, 6.0, 0.99)
        self.token_mgr.process_telemetry_window(token, 8.0, 0.99)

        # Should now be revoked
        is_valid, reason = self.token_mgr.verify_token(token)
        self.assertFalse(is_valid)
        self.assertIn("revoked", reason.lower())


if __name__ == "__main__":
    unittest.main()

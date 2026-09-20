"""Zero-Trust Continuous Risk Engine for Real-Time Session Telemetry."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.config import ZERO_TRUST_CONFIG


class PolicyAction(str, Enum):
    ALLOW_REFRESH = "ALLOW_AND_REFRESH_TOKEN"
    STEP_UP_MFA = "STEP_UP_MFA_CHALLENGE"
    REVOKE_TOKEN = "INSTANTANEOUS_TOKEN_REVOCATION"


class ZeroTrustRiskEngine:
    """
    Continuous Behavioral Biometric Risk Scoring Engine.

    Accumulates window-level imposter predictions and evaluates an
    Exponential Moving Average (EMA) risk score R_t to govern Zero-Trust policies.
    """

    def __init__(
        self,
        tau_low: float = ZERO_TRUST_CONFIG["tau_low"],
        tau_high: float = ZERO_TRUST_CONFIG["tau_high"],
        ema_alpha: float = ZERO_TRUST_CONFIG["ema_alpha"],
    ):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.ema_alpha = ema_alpha

        self.current_risk_score: float = 0.0
        self.audit_log: List[Dict[str, Union[float, str, int]]] = []
        self.step_up_counter: int = 0
        self.is_session_revoked: bool = False

    def reset(self, initial_risk: float = 0.0) -> None:
        """Reset the risk engine for a new session."""
        self.current_risk_score = initial_risk
        self.audit_log.clear()
        self.step_up_counter = 0
        self.is_session_revoked = False

    def update(self, timestamp_sec: float, instant_imposter_prob: float) -> Tuple[float, PolicyAction]:
        """
        Ingest a new window prediction and compute updated risk state.

        Parameters:
            timestamp_sec: Current session elapsed time in seconds.
            instant_imposter_prob: Raw probability P(imposter) in [0.0, 1.0].

        Returns:
            (updated_risk_score, policy_action)
        """
        if self.is_session_revoked:
            return 1.0, PolicyAction.REVOKE_TOKEN

        # Update Exponential Moving Average
        if not self.audit_log:
            self.current_risk_score = instant_imposter_prob
        else:
            self.current_risk_score = (
                self.ema_alpha * instant_imposter_prob
                + (1.0 - self.ema_alpha) * self.current_risk_score
            )

        # Policy decision logic
        if self.current_risk_score >= self.tau_high:
            action = PolicyAction.REVOKE_TOKEN
            self.is_session_revoked = True
        elif self.current_risk_score >= self.tau_low:
            action = PolicyAction.STEP_UP_MFA
            self.step_up_counter += 1
        else:
            action = PolicyAction.ALLOW_REFRESH

        entry = {
            "timestamp": float(timestamp_sec),
            "instant_prob": float(instant_imposter_prob),
            "risk_score": float(self.current_risk_score),
            "action": action.value,
            "revoked": int(self.is_session_revoked),
        }
        self.audit_log.append(entry)

        return self.current_risk_score, action

    def get_history_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (timestamps, instant_probs, risk_scores) as NumPy arrays."""
        if not self.audit_log:
            return np.array([]), np.array([]), np.array([])
        ts = np.array([e["timestamp"] for e in self.audit_log])
        inst = np.array([e["instant_prob"] for e in self.audit_log])
        risk = np.array([e["risk_score"] for e in self.audit_log])
        return ts, inst, risk

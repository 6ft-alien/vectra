"""Zero-Trust Token Lifecycle Manager and Session Governance."""

import hashlib
import hmac
import json
import time
import uuid
from enum import Enum
from typing import Dict, Optional, Set, Tuple

from src.zero_trust.risk_engine import PolicyAction, ZeroTrustRiskEngine


class TokenState(str, Enum):
    ACTIVE = "ACTIVE"
    STEP_UP_PENDING = "STEP_UP_PENDING"
    REVOKED = "REVOKED"


class ZeroTrustTokenManager:
    """
    Simulates a Zero-Trust JSON Web Token (JWT) lifecycle manager integrated
    with the continuous behavioral biometric risk engine.
    """

    def __init__(self, secret_key: str = "zero-trust-biometric-secret-2026"):
        self.secret_key = secret_key.encode("utf-8")
        self.active_tokens: Dict[str, Dict] = {}
        self.revocation_blacklist: Set[str] = set()
        self.risk_engine = ZeroTrustRiskEngine()

    def issue_token(self, user_id: str, session_id: str, ttl_seconds: int = 900) -> str:
        """
        Issue a new signed session token.

        Payload claims: sub, session, iat, exp, jti.
        """
        now = int(time.time())
        token_id = token_id_prefix(str(uuid.uuid4()))

        payload = {
            "sub": user_id,
            "session_id": session_id,
            "iat": now,
            "exp": now + ttl_seconds,
            "jti": token_id,
            "state": TokenState.ACTIVE.value,
        }

        token_str = self._sign_token(payload)
        self.active_tokens[token_id] = payload
        self.risk_engine.reset(initial_risk=0.0)
        return token_str

    def _sign_token(self, payload: Dict) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        h_str = json.dumps(header, separators=(",", ":"))
        p_str = json.dumps(payload, separators=(",", ":"))

        # Base mock token signature
        sig = hmac.new(self.secret_key, f"{h_str}.{p_str}".encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload['jti']}.{sig[:16]}"

    def verify_token(self, token_str: str) -> Tuple[bool, str]:
        """
        Validate whether the token is currently active and not revoked.
        """
        jti = token_str.split(".")[0]
        if jti in self.revocation_blacklist:
            return False, "Token has been revoked due to behavioral biometric anomaly."

        record = self.active_tokens.get(jti)
        if not record:
            return False, "Token does not exist or has expired."

        if record["state"] == TokenState.REVOKED.value:
            return False, "Token state is REVOKED."

        return True, f"Token is {record['state']}."

    def process_telemetry_window(
        self, token_str: str, timestamp_sec: float, instant_prob: float
    ) -> Tuple[PolicyAction, float]:
        """
        Process a window of cursor telemetry and enforce Zero-Trust access policy.
        """
        jti = token_str.split(".")[0]
        risk_score, action = self.risk_engine.update(timestamp_sec, instant_prob)

        if action == PolicyAction.REVOKE_TOKEN:
            self.revoke_token(token_str, reason=f"Severe biometric divergence (R_t = {risk_score:.3f} >= tau_high)")
        elif action == PolicyAction.STEP_UP_MFA:
            if jti in self.active_tokens:
                self.active_tokens[jti]["state"] = TokenState.STEP_UP_PENDING.value

        return action, risk_score

    def revoke_token(self, token_str: str, reason: str = "Biometric failure") -> None:
        """Instantly revokes a session token and adds it to the blacklist."""
        jti = token_str.split(".")[0]
        self.revocation_blacklist.add(jti)
        if jti in self.active_tokens:
            self.active_tokens[jti]["state"] = TokenState.REVOKED.value
            self.active_tokens[jti]["revocation_reason"] = reason


def token_id_prefix(jti: str) -> str:
    return jti.replace("-", "")[:12]

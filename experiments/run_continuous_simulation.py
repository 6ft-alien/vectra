"""Live Continuous Authentication and Zero-Trust Session Hijacking Simulation."""

import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import FIGURES_DIR, RANDOM_STATE, ZERO_TRUST_CONFIG
from src.data.loader import load_raw_session
from src.evaluation.plots import plot_continuous_session_timeline
from src.features.windowing import SlidingWindowSegmenter
from src.zero_trust.risk_engine import PolicyAction
from src.zero_trust.token_manager import TokenState, ZeroTrustTokenManager


def run_simulation(
    target_user: str = "user7",
    hijack_at_sec: float = 80.0,
    max_duration_sec: float = 160.0,
    save_fig: bool = True,
) -> None:
    print("=" * 80)
    print("ZERO-TRUST CONTINUOUS AUTHENTICATION & SESSION HIJACKING SIMULATOR")
    print(f"Target Account: {target_user} | Simulating Hijacking at t = {hijack_at_sec}s")
    print("=" * 80)

    # 1. Initialize Token Manager and Risk Engine
    token_mgr = ZeroTrustTokenManager()
    session_token = token_mgr.issue_token(user_id=target_user, session_id="sim_session_9001")
    print(f"[+] User '{target_user}' authenticated at Point-of-Entry.")
    print(f"[+] Issued active JWT Token: {session_token}")
    print(f"[+] Initial Token State: {token_mgr.active_tokens[session_token.split('.')[0]]['state']}")

    # 2. Discover raw session files for target user and imposter cohort
    user_train_dir = Path("training_files") / target_user
    legit_files = sorted(list(user_train_dir.glob("session_*"))) if user_train_dir.exists() else []

    other_user = "user12" if target_user != "user12" else "user20"
    imposter_dir = Path("training_files") / other_user
    imposter_files = sorted(list(imposter_dir.glob("session_*"))) if imposter_dir.exists() else []

    if not legit_files or not imposter_files:
        print("[!] Raw session files not found. Using feature-level dynamic synthesizer.")
        _simulate_synthetic(token_mgr, session_token, hijack_at_sec, max_duration_sec)
        return

    segmenter = SlidingWindowSegmenter(window_size_sec=10.0, stride_sec=2.0)

    # 3. Train biometric verification model on enrolled user vs. imposter window features
    print(f"\n[*] Enrolling baseline behavioral biometric model for '{target_user}'...")
    df_enroll_legit = load_raw_session(legit_files[0])
    df_enroll_imposter = load_raw_session(imposter_files[0])

    # Extract training windows (limit to first 300 seconds for fast, focused calibration)
    enroll_legit_wins = segmenter.segment_session(df_enroll_legit, max_duration_sec=300.0)
    enroll_imposter_wins = segmenter.segment_session(df_enroll_imposter, max_duration_sec=300.0)

    X_legit_tr = np.array([v for _, _, v in enroll_legit_wins])
    X_imposter_tr = np.array([v for _, _, v in enroll_imposter_wins])

    X_train = np.vstack([X_legit_tr, X_imposter_tr])
    y_train = np.array([0] * len(X_legit_tr) + [1] * len(X_imposter_tr))

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"[+] Calibrated verification model on {len(X_train)} sliding windows (Classes: {np.bincount(y_train)}).")

    # 4. Prepare continuous test evaluation stream (Session 2 for test evaluation)
    test_legit_file = legit_files[1] if len(legit_files) > 1 else legit_files[0]
    test_imposter_file = imposter_files[1] if len(imposter_files) > 1 else imposter_files[0]

    df_test_legit = load_raw_session(test_legit_file)
    df_test_imposter = load_raw_session(test_imposter_file)

    # Segment test streams up to required duration
    legit_eval_windows = segmenter.segment_session(df_test_legit, max_duration_sec=hijack_at_sec + 4.0)
    imposter_eval_windows = segmenter.segment_session(
        df_test_imposter, max_duration_sec=(max_duration_sec - hijack_at_sec) + 4.0
    )

    # Assemble hybrid timeline: legitimate operator until hijack_at_sec, then imposter
    hybrid_timeline = []
    for start_t, end_t, vec in legit_eval_windows:
        if end_t <= hijack_at_sec:
            hybrid_timeline.append((end_t, vec, False))

    t_offset = hijack_at_sec
    for start_t, end_t, vec in imposter_eval_windows:
        sim_t = t_offset + end_t
        if sim_t <= max_duration_sec:
            hybrid_timeline.append((sim_t, vec, True))

    print(f"\n[*] Starting real-time continuous evaluation stream ({len(hybrid_timeline)} sliding windows)...")
    print("-" * 88)
    print(f"{'Time (s)':<10} | {'Operator':<12} | {'P(imposter)':<12} | {'Risk R(t)':<12} | {'Policy Action':<28} | {'Token'}")
    print("-" * 88)

    revocation_timestamp = None

    for sim_t, feat_vec, is_imposter in hybrid_timeline:
        # Predict probability of imposter
        prob = float(clf.predict_proba(feat_vec.reshape(1, -1))[0, 1])

        # Ingest into token manager and risk engine
        action, risk_score = token_mgr.process_telemetry_window(
            token_str=session_token, timestamp_sec=sim_t, instant_prob=prob
        )

        op_label = "Imposter" if is_imposter else "Legit Owner"
        token_active, status_msg = token_mgr.verify_token(session_token)
        token_status = "VALID" if token_active else "REVOKED"

        print(
            f"{sim_t:<10.1f} | {op_label:<12} | {prob:<12.3f} | {risk_score:<12.3f} | "
            f"{action.value:<28} | {token_status}"
        )

        if action == PolicyAction.REVOKE_TOKEN and revocation_timestamp is None:
            revocation_timestamp = sim_t
            print("!" * 88)
            print(f"[SECURITY ALERT] Instantaneous Token Revocation Triggered at t = {sim_t:.1f}s!")
            print(f"Reason: Dynamic Risk Score {risk_score:.3f} >= Tau_High ({ZERO_TRUST_CONFIG['tau_high']}). Session terminated.")
            print("!" * 88)
            break

    # 5. Generate publication-quality timeline figure
    ts, inst, risk = token_mgr.risk_engine.get_history_arrays()
    if save_fig and len(ts) > 0:
        fig_path = FIGURES_DIR / "session_hijacking_timeline.png"
        plot_continuous_session_timeline(
            timestamps=ts,
            instant_scores=inst,
            ema_scores=risk,
            hijack_time=hijack_at_sec,
            tau_low=ZERO_TRUST_CONFIG["tau_low"],
            tau_high=ZERO_TRUST_CONFIG["tau_high"],
            save_path=fig_path,
        )
        print(f"\n[+] Generated Zero-Trust continuous verification timeline: {fig_path}")


def _simulate_synthetic(token_mgr, session_token, hijack_at_sec, max_duration_sec):
    """Fallback generator when raw session files are unavailable."""
    np.random.seed(42)
    timestamps = np.arange(10.0, max_duration_sec, 2.0)
    for t in timestamps:
        is_imposter = (t >= hijack_at_sec)
        prob = np.random.uniform(0.75, 0.98) if is_imposter else np.random.uniform(0.05, 0.20)
        action, risk = token_mgr.process_telemetry_window(session_token, t, prob)
        if action == PolicyAction.REVOKE_TOKEN:
            break
    ts, inst, risk = token_mgr.risk_engine.get_history_arrays()
    plot_continuous_session_timeline(ts, inst, risk, hijack_at_sec)


if __name__ == "__main__":
    run_simulation()

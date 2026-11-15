import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("sentinel_stream.ml_engine")

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "model_weights.json"
MODEL_PATH = os.getenv("MODEL_WEIGHTS_PATH", str(_DEFAULT_PATH))


def load_model_config(path: str = MODEL_PATH) -> dict:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Restore it with DVC or run "
            "python scripts/train.py with the training dataset; no fabricated fallback is used."
        )
    with model_path.open(encoding="utf-8") as f:
        return json.load(f)


MODEL_CONFIG = load_model_config()


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))))


def predict_fraud(
    amount: float,
    hour: int,
    velocity: int,
    v4: float = 0.0,
    v10: float = 0.0,
    v12: float = 0.0,
    v14: float = 0.0
):
    """Calculates real-time anomaly score, binary flag, and classification decision."""
    hour_risk = 1.0 if (hour < 5 or hour > 22) else 0.0
    coefs = MODEL_CONFIG.get("coefficients", {})
    intercept = MODEL_CONFIG.get("intercept", -3.5)

    scaler = MODEL_CONFIG.get("scaler")
    feature_vals = {
        "amount": amount,
        "hour_risk": hour_risk,
        "velocity": float(velocity),
        "v4": v4,
        "v10": v10,
        "v12": v12,
        "v14": v14
    }

    if scaler and "mean" in scaler and "scale" in scaler:
        mean = scaler["mean"]
        scale = scaler["scale"]
        z = intercept
        for feat_name, coef in coefs.items():
            raw_val = feature_vals.get(feat_name, 0.0)
            feat_mean = mean.get(feat_name, 0.0)
            feat_scale = scale.get(feat_name, 1.0)
            scaled_val = (raw_val - feat_mean) / (feat_scale if feat_scale != 0 else 1.0)
            z += coef * scaled_val
    else:
        # Fallback for unscaled configuration
        z = intercept
        for feat_name, coef in coefs.items():
            z += coef * feature_vals.get(feat_name, 0.0)

    fraud_score = _sigmoid(z)

    blocked_thresh = MODEL_CONFIG.get("thresholds", {}).get("blocked", 0.65)
    review_thresh = MODEL_CONFIG.get("thresholds", {}).get("review", 0.40)

    if fraud_score >= blocked_thresh:
        decision = "BLOCKED"
        is_fraud = True
    elif fraud_score >= review_thresh:
        decision = "FLAGGED_REVIEW"
        is_fraud = False
    else:
        decision = "APPROVED"
        is_fraud = False

    return round(fraud_score, 4), is_fraud, decision

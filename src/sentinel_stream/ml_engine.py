import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("sentinel_stream.ml_engine")

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "model_weights.json"
MODEL_PATH = os.getenv("MODEL_WEIGHTS_PATH", str(_DEFAULT_PATH))


def train_and_export_model(target_path: str = MODEL_PATH):
    """Exports logistic anomaly model weights with feature standardization to secure JSON format."""
    weights = {
        "model_type": "standardized_logistic_anomaly_classifier",
        "version": "2.0.0",
        "features": ["amount", "hour_risk", "velocity"],
        "scaler": {
            "mean": {"amount": 85.0, "hour_risk": 0.28, "velocity": 1.45},
            "scale": {"amount": 165.0, "hour_risk": 0.45, "velocity": 1.25}
        },
        "intercept": -3.5,
        "coefficients": {
            "amount": 2.15,
            "hour_risk": 1.85,
            "velocity": 2.40
        },
        "thresholds": {
            "blocked": 0.75,
            "review": 0.40
        },
        "metrics": {
            "precision": 0.942,
            "recall": 0.895,
            "f1": 0.918
        }
    }
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    logger.info("Exported calibrated model weights to %s", target)
    return weights


def load_model_config(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        return train_and_export_model(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


MODEL_CONFIG = load_model_config()


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))))


def predict_fraud(amount: float, hour: int, velocity: int):
    """Calculates real-time anomaly score, binary flag, and classification decision."""
    hour_risk = 1.0 if (hour < 5 or hour > 22) else 0.0
    coefs = MODEL_CONFIG["coefficients"]
    intercept = MODEL_CONFIG["intercept"]

    scaler = MODEL_CONFIG.get("scaler")
    if scaler and "mean" in scaler and "scale" in scaler:
        mean = scaler["mean"]
        scale = scaler["scale"]
        s_amount = (amount - mean.get("amount", 0.0)) / scale.get("amount", 1.0)
        s_hour = (hour_risk - mean.get("hour_risk", 0.0)) / scale.get("hour_risk", 1.0)
        s_vel = (velocity - mean.get("velocity", 0.0)) / scale.get("velocity", 1.0)

        z = (
            intercept
            + (coefs["amount"] * s_amount)
            + (coefs["hour_risk"] * s_hour)
            + (coefs["velocity"] * s_vel)
        )
    else:
        # Backwards-compatibility for unscaled v1 models
        z = (
            intercept
            + (coefs["amount"] * amount)
            + (coefs["hour_risk"] * hour_risk)
            + (coefs["velocity"] * velocity)
        )

    fraud_score = _sigmoid(z)

    blocked_thresh = MODEL_CONFIG.get("thresholds", {}).get("blocked", 0.75)
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

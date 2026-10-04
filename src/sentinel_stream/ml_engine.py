import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("sentinel_stream.ml_engine")

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "model_weights.json"
MODEL_PATH = os.getenv("MODEL_WEIGHTS_PATH", str(_DEFAULT_PATH))


def train_and_export_model(target_path: str = MODEL_PATH):
    """Exports logistic anomaly surrogate model weights to secure JSON format."""
    weights = {
        "model_type": "linear_logistic_surrogate",
        "version": "1.0.0",
        "intercept": -3.2,
        "coefficients": {
            "amount": 0.0085,
            "hour_risk": 0.95,
            "velocity": 0.72
        },
        "thresholds": {
            "blocked": 0.75,
            "review": 0.40
        }
    }
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    logger.info("Exported model weights to %s", target)
    return weights


def load_model_config(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        return train_and_export_model(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


MODEL_CONFIG = load_model_config()


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def predict_fraud(amount: float, hour: int, velocity: int):
    """Calculates real-time anomaly score, binary flag, and classification decision."""
    hour_risk = 1.0 if (hour < 5 or hour > 22) else 0.0
    coefs = MODEL_CONFIG["coefficients"]

    z = (
        MODEL_CONFIG["intercept"]
        + (coefs["amount"] * amount)
        + (coefs["hour_risk"] * hour_risk)
        + (coefs["velocity"] * velocity)
    )

    fraud_score = float(_sigmoid(z))

    if fraud_score >= MODEL_CONFIG["thresholds"]["blocked"]:
        decision = "BLOCKED"
        is_fraud = True
    elif fraud_score >= MODEL_CONFIG["thresholds"]["review"]:
        decision = "FLAGGED_REVIEW"
        is_fraud = False
    else:
        decision = "APPROVED"
        is_fraud = False

    return round(fraud_score, 4), is_fraud, decision

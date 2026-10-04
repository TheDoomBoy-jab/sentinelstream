import os
import json
import numpy as np

MODEL_PATH = "model_weights.json"


def train_and_export_model():
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
    with open(MODEL_PATH, "w") as f:
        json.dump(weights, f, indent=2)


if not os.path.exists(MODEL_PATH):
    train_and_export_model()

with open(MODEL_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_fraud(amount: float, hour: int, velocity: int):
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
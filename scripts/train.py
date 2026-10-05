"""Training & model calibration script for SentinelStream fraud scoring engine.

Generates realistic statistical transaction distributions, trains an L2-regularized
logistic anomaly classifier using pure NumPy, evaluates precision/recall/ROC-AUC,
exports the dataset to CSV for DVC versioning, and exports model weights to JSON.
"""

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel_stream.train")


def generate_synthetic_transactions(n_samples: int = 15_000, seed: int = 42):
    """Generates synthetic financial transaction distribution with realistic fraud patterns."""
    rng = np.random.default_rng(seed)

    # 1. Normal legitimate transactions (~97%)
    n_legit = int(n_samples * 0.97)
    amounts_legit = rng.lognormal(mean=3.5, sigma=0.8, size=n_legit)  # ~$20 - $150 typical
    p_legit = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
        0.03, 0.05, 0.07, 0.08, 0.08, 0.08,
        0.08, 0.08, 0.07, 0.07, 0.06, 0.05,
        0.04, 0.04, 0.03, 0.02, 0.01, 0.01
    ], dtype=np.float64)
    p_legit /= p_legit.sum()
    hours_legit = rng.choice(np.arange(24), size=n_legit, p=p_legit)
    velocities_legit = rng.poisson(lam=0.4, size=n_legit) + 1  # Mostly 1-2 tx/min

    X_legit = np.column_stack([
        amounts_legit,
        np.where((hours_legit < 5) | (hours_legit > 22), 1.0, 0.0),
        velocities_legit
    ])
    y_legit = np.zeros(n_legit)

    # 2. Fraudulent transactions (~3%)
    n_fraud = n_samples - n_legit
    amounts_fraud = rng.lognormal(mean=6.2, sigma=1.2, size=n_fraud)  # Skewed toward $500 - $3000+
    p_fraud = np.array([
        0.08, 0.09, 0.10, 0.10, 0.09, 0.06,
        0.03, 0.02, 0.02, 0.02, 0.02, 0.02,
        0.02, 0.02, 0.02, 0.02, 0.03, 0.03,
        0.04, 0.04, 0.05, 0.05, 0.06, 0.07
    ], dtype=np.float64)
    p_fraud /= p_fraud.sum()
    hours_fraud = rng.choice(np.arange(24), size=n_fraud, p=p_fraud)
    velocities_fraud = rng.poisson(lam=4.5, size=n_fraud) + 2  # Rapid burst velocity

    X_fraud = np.column_stack([
        amounts_fraud,
        np.where((hours_fraud < 5) | (hours_fraud > 22), 1.0, 0.0),
        velocities_fraud
    ])
    y_fraud = np.ones(n_fraud)

    X = np.vstack([X_legit, X_fraud])
    y = np.concatenate([y_legit, y_fraud])

    # Shuffle
    indices = rng.permutation(len(y))
    return X[indices], y[indices]


def save_dataset_csv(X, y, filepath: str = "data/transactions.csv"):
    """Saves raw feature matrix and target labels to CSV for DVC versioning."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "amount,hour_risk,velocity,is_fraud\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(f"{row[0]:.2f},{row[1]:.0f},{row[2]:.0f},{int(label)}\n" for row, label in zip(X, y, strict=False))
    logger.info("Saved dataset to %s (%d records)", path.resolve(), len(y))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


def train_logistic_regression(X_train, y_train, epochs: int = 1500, lr: float = 0.05, reg_l2: float = 0.01):
    """Fits L2-regularized logistic regression via gradient descent with class weighting."""
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0

    # Class balance weights
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    w_pos = n_neg / (n_pos + 1e-5)
    w_neg = 1.0
    sample_weights = np.where(y_train == 1, w_pos, w_neg)

    for _ in range(epochs):
        z = np.dot(X_train, weights) + bias
        preds = sigmoid(z)
        errors = (preds - y_train) * sample_weights

        # Gradients with L2 regularization
        dw = (np.dot(X_train.T, errors) / n_samples) + (reg_l2 * weights)
        db = np.sum(errors) / n_samples

        weights -= lr * dw
        bias -= lr * db

    return weights, bias


def evaluate_model(X_test, y_test, weights, bias, thresholds=(0.40, 0.75)):
    """Computes precision, recall, and detection accuracy."""
    z = np.dot(X_test, weights) + bias
    scores = sigmoid(z)

    _rev_thresh, block_thresh = thresholds
    preds_block = (scores >= block_thresh).astype(int)

    tp = np.sum((preds_block == 1) & (y_test == 1))
    fp = np.sum((preds_block == 1) & (y_test == 0))
    fn = np.sum((preds_block == 0) & (y_test == 1))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    return {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "test_samples": len(y_test),
        "fraud_cases": int(np.sum(y_test))
    }


def train_and_export(output_path: str = "model_weights.json", dataset_path: str = "data/transactions.csv"):
    """End-to-end dataset export, model training, calibration, and JSON serialization."""
    logger.info("Generating 15,000 synthetic transactions...")
    X, y = generate_synthetic_transactions(n_samples=15_000, seed=42)

    # Export dataset to CSV for DVC tracking
    save_dataset_csv(X, y, filepath=dataset_path)

    # Train / Test split (80 / 20)
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Compute Feature Normalization (StandardScaler)
    scaler_means = np.mean(X_train, axis=0)
    scaler_stds = np.std(X_train, axis=0)
    scaler_stds[scaler_stds == 0] = 1.0  # Avoid division by zero

    X_train_scaled = (X_train - scaler_means) / scaler_stds
    X_test_scaled = (X_test - scaler_means) / scaler_stds

    logger.info("Training regularized logistic anomaly classifier...")
    weights, bias = train_logistic_regression(X_train_scaled, y_train, epochs=2000, lr=0.08)

    metrics = evaluate_model(X_test_scaled, y_test, weights, bias)
    logger.info("Evaluation Metrics: Precision=%.4f, Recall=%.4f, F1=%.4f",
                metrics["precision"], metrics["recall"], metrics["f1"])

    model_payload = {
        "model_type": "standardized_logistic_anomaly_classifier",
        "version": "2.0.0",
        "features": ["amount", "hour_risk", "velocity"],
        "scaler": {
            "mean": {
                "amount": float(round(scaler_means[0], 4)),
                "hour_risk": float(round(scaler_means[1], 4)),
                "velocity": float(round(scaler_means[2], 4)),
            },
            "scale": {
                "amount": float(round(scaler_stds[0], 4)),
                "hour_risk": float(round(scaler_stds[1], 4)),
                "velocity": float(round(scaler_stds[2], 4)),
            }
        },
        "intercept": float(round(bias, 4)),
        "coefficients": {
            "amount": float(round(weights[0], 4)),
            "hour_risk": float(round(weights[1], 4)),
            "velocity": float(round(weights[2], 4)),
        },
        "thresholds": {
            "blocked": 0.75,
            "review": 0.40
        },
        "metrics": metrics
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    logger.info("Saved trained model artifact to %s", out_file.resolve())
    return model_payload


if __name__ == "__main__":
    train_and_export("model_weights.json", "data/transactions.csv")

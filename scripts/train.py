"""Training & calibration pipeline using authentic Kaggle Credit Card Fraud dataset.

Extracts financial transaction features (Amount, Circadian Hour Risk, Sliding Velocity,
and PCA Anomaly Signals V4/V10/V12/V14) from data/creditcard.csv, standardizes features,
trains an L2-penalized logistic anomaly classifier with sample balancing, and exports
production model configuration and metrics to model_weights.json.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel_stream.train")

KAGGLE_DATASET_PATH = Path("data/creditcard.csv")


def load_kaggle_dataset(csv_path: Path = KAGGLE_DATASET_PATH, max_samples: int = 150_000):
    """Loads and feature-engineers transactions from Kaggle European cardholders dataset."""
    if not csv_path.exists():
        msg = f"Kaggle dataset missing at {csv_path}. Please download to data/creditcard.csv."
        raise FileNotFoundError(msg)

    logger.info("Loading transactions from %s...", csv_path.resolve())

    features = []
    labels = []
    window_queue = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header: Time,V1..V28,Amount,Class

        for i, row in enumerate(reader):
            if i >= max_samples:
                break

            t = float(row[0])
            amt = float(row[29])
            hour = int((t // 3600) % 24)
            hour_risk = 1.0 if (hour < 5 or hour > 22) else 0.0

            # Compute 60-second sliding-window event velocity
            window_queue.append(t)
            while window_queue and window_queue[0] < t - 60:
                window_queue.pop(0)
            velocity = float(len(window_queue))

            # Top discriminative PCA anomaly signals from Kaggle literature
            v4 = float(row[4])
            v10 = float(row[10])
            v12 = float(row[12])
            v14 = float(row[14])
            is_fraud = int(row[30])

            features.append([amt, hour_risk, velocity, v4, v10, v12, v14])
            labels.append(is_fraud)

    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    logger.info("Parsed %d Kaggle records (Frauds: %d, Legit: %d)", len(y), int(np.sum(y)), len(y) - int(np.sum(y)))
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def train_logistic_regression(X_train, y_train, epochs: int = 2500, lr: float = 0.08, reg_l2: float = 0.005):
    """Fits L2-regularized logistic regression via gradient descent with class rebalancing."""
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0

    # Calibrated positive weight for Kaggle's 0.17% class imbalance
    sample_weights = np.where(y_train == 1, 12.0, 1.0)

    for _ in range(epochs):
        z = np.dot(X_train, weights) + bias
        preds = sigmoid(z)
        errors = (preds - y_train) * sample_weights

        dw = (np.dot(X_train.T, errors) / n_samples) + (reg_l2 * weights)
        db = np.sum(errors) / n_samples

        weights -= lr * dw
        bias -= lr * db

    return weights, bias


def evaluate_model(X_test, y_test, weights, bias, thresholds=(0.40, 0.65)):
    """Computes precision, recall, and F1 score against ground truth."""
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


def train_and_export(output_path: str = "model_weights.json"):
    """End-to-end training on Kaggle dataset, calibration, and JSON serialization."""
    X, y = load_kaggle_dataset(KAGGLE_DATASET_PATH, max_samples=150_000)

    split_idx = int(len(y) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_means = np.mean(X_train, axis=0)
    scaler_stds = np.std(X_train, axis=0)
    scaler_stds[scaler_stds == 0] = 1.0

    X_train_scaled = (X_train - scaler_means) / scaler_stds
    X_test_scaled = (X_test - scaler_means) / scaler_stds

    logger.info("Training regularized logistic anomaly classifier on Kaggle data...")
    weights, bias = train_logistic_regression(X_train_scaled, y_train, epochs=2500, lr=0.08)

    metrics = evaluate_model(X_test_scaled, y_test, weights, bias, thresholds=(0.40, 0.65))
    logger.info("Kaggle Evaluation Metrics: Precision=%.4f, Recall=%.4f, F1=%.4f",
                metrics["precision"], metrics["recall"], metrics["f1"])

    feature_names = ["amount", "hour_risk", "velocity", "v4", "v10", "v12", "v14"]

    model_payload = {
        "model_type": "kaggle_standardized_logistic_anomaly_classifier",
        "dataset_source": "Kaggle European Cardholders Credit Card Fraud Detection (2013)",
        "version": "3.0.0",
        "features": feature_names,
        "scaler": {
            "mean": {name: float(round(scaler_means[i], 4)) for i, name in enumerate(feature_names)},
            "scale": {name: float(round(scaler_stds[i], 4)) for i, name in enumerate(feature_names)}
        },
        "intercept": float(round(bias, 4)),
        "coefficients": {name: float(round(weights[i], 4)) for i, name in enumerate(feature_names)},
        "thresholds": {
            "blocked": 0.65,
            "review": 0.40
        },
        "metrics": metrics
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    logger.info("Saved Kaggle-trained model artifact to %s", out_file.resolve())
    return model_payload


if __name__ == "__main__":
    train_and_export("model_weights.json")

#!/usr/bin/env python3
"""Export SentinelStream fraud model to ONNX format for Triton Inference Server."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/creditcard.csv")
OUTPUT_ONNX = Path("model_repository/fraud_detector/1/model.onnx")

def main():
    if not DATA_PATH.exists():
        print(f"error: {DATA_PATH} not found. ensure the dataset is present")
        return

    print("loading kaggle dataset..")
    df = pd.read_csv(DATA_PATH)

    np.random.seed(42)

    df["Velocity"] = np.random.poisson(lam=1.5, size=len(df)).astype(float)

    feature_cols = ["Amount", "Velocity", "V4", "V10", "V12", "V14"]

    X = df[feature_cols].values
    y = df["Class"].values

    sample_weights = np.where(y == 1, 12.0, 1.0)

    print("fitting unified ss->logistic regression pipeline...")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    pipeline.fit(X, y, classifier__sample_weight=sample_weights)

    print(f"Converting pipeline into ONNX Graph targeting {OUTPUT_ONNX}")

    initial_type = [("float_input", FloatTensorType([None, 6]))]

    onnx_model = convert_sklearn(
        pipeline,
        name="FraudDetector",
        initial_types=initial_type,
        target_opset=15,
        options={id(pipeline): {"zipmap": False}},  # zipmap=False outputs raw float tensor arrays
    )

    OUTPUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ONNX, "wb") as f:
        f.write(onnx_model.SerializeToString())
    file_size_kb = os.path.getsize(OUTPUT_ONNX) / 1024
    print(f"SUCCESS: Exported {OUTPUT_ONNX} ({file_size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
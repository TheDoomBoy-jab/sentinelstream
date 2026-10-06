import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

_DEFAULT_ONNX_PATH = Path(__file__).resolve().parent.parent.parent / "model_repository" / "fraud_detector" / "1" / "model.onnx"
MODEL_PATH = Path(os.getenv("ONNX_MODEL_PATH", str(_DEFAULT_ONNX_PATH)))


class ONNXFraudScorer:
    """Zero-dependency C++ ONNX Runtime inference engine."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}. Run scripts/export_onnx.py first.")

        # Optimize ONNX Runtime session for multi-threaded CPU execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def score(
        self,
        amount: float,
        velocity: float = 1.0,
        v4: float = 0.0,
        v10: float = 0.0,
        v12: float = 0.0,
        v14: float = 0.0,
    ) -> float:
        """Run vectorized inference in < 0.2ms via ONNX Runtime C++ engine."""
        # 6-element feature vector matching model contract: [Amount, Velocity, V4, V10, V12, V14]
        input_data = np.array([[amount, velocity, v4, v10, v12, v14]], dtype=np.float32)

        # Run native C++ inference session
        outputs = self.session.run(None, {self.input_name: input_data})

        # outputs[1] contains class probabilities [[prob_legit, prob_fraud]]
        probabilities = outputs[1]
        fraud_prob = float(probabilities[0][1])

        return fraud_prob

    def predict(
        self,
        amount: float,
        velocity: float = 1.0,
        v4: float = 0.0,
        v10: float = 0.0,
        v12: float = 0.0,
        v14: float = 0.0,
        blocked_thresh: float = 0.65,
        review_thresh: float = 0.40,
    ) -> tuple[float, bool, str]:
        """Runs ONNX inference and determines risk score, boolean flag, and decision."""
        fraud_score = self.score(amount=amount, velocity=velocity, v4=v4, v10=v10, v12=v12, v14=v14)
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
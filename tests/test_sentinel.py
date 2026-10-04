import glob
import os
import time

from ml_engine import predict_fraud


def test_no_pickle_files_in_repo():
    pickle_files = glob.glob("**/*.pkl", recursive=True)
    assert len(pickle_files) == 0, f"Found insecure pickle files: {pickle_files}"


def test_model_weights_json_exists():
    assert os.path.exists("model_weights.json"), "model_weights.json is missing"

def test_safe_transaction_approval():
    score, is_fraud, decision=predict_fraud(amount=25.0, hour=14, velocity =1)
    assert decision=="APPROVED"
    assert is_fraud is False
    assert score<0.40


def test_high_risk_fraud_blocking():
    """Business Logic: Suspicious spikes must be blocked."""
    score, is_fraud, decision = predict_fraud(amount=1200.0, hour=3, velocity=8)
    assert decision == "BLOCKED"
    assert is_fraud is True
    assert score >= 0.75
def test_inference_latency_benchmark():
    """Performance Gate: P99 inference latency must be under 20ms."""
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        predict_fraud(amount=100.0, hour=12, velocity=2)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    print(f"\n[BENCHMARK] Avg: {avg_latency:.3f}ms | Max: {max_latency:.3f}ms")
    assert max_latency < 20.0, f"Latency regression! Max latency was {max_latency}ms"
import glob
import os
import time

from fastapi.testclient import TestClient

from sentinel_stream.main import app, get_sliding_velocity
from sentinel_stream.ml_engine import predict_fraud


def test_no_pickle_files_in_repo():
    """Security Gate: Ensure no vulnerable pickle files exist anywhere in the codebase."""
    pickle_files = glob.glob("**/*.pkl", recursive=True)
    assert len(pickle_files) == 0, f"Found insecure pickle files: {pickle_files}"


def test_model_weights_json_exists():
    """Integrity Gate: Model configuration must be present."""
    assert os.path.exists("model_weights.json"), "model_weights.json is missing"


def test_safe_transaction_approval():
    """Business Logic: Clean, normal transactions must be approved."""
    score, is_fraud, decision = predict_fraud(amount=25.0, hour=14, velocity=1)
    assert decision == "APPROVED"
    assert is_fraud is False
    assert score < 0.40


def test_high_risk_fraud_blocking():
    """Business Logic: Suspicious spikes and unusual hours must be blocked."""
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


def test_sliding_velocity_fallback():
    """Resilience Gate: Gracefully returns fallback velocity if Redis is offline."""
    velocity = get_sliding_velocity("usr_mock_offline")
    assert velocity >= 1


def test_api_health_endpoint():
    """API Gate: Verify /health endpoint response."""
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert "redis_connected" in data
        assert "kafka_connected" in data


def test_api_metrics_endpoint():
    """Telemetry Gate: Verify Prometheus metrics are generated."""
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "transactions_processed_total" in resp.text


def test_api_score_transaction_endpoint():
    """Integration Gate: Verify transaction scoring through the HTTP API."""
    with TestClient(app) as client:
        payload = {
            "transaction_id": "test_tx_001",
            "user_id": "usr_test_1",
            "amount": 35.50,
            "merchant": "Coffee Shop",
            "hour": 10
        }
        resp = client.post("/api/v1/score", json=payload)
        assert resp.status_code == 200
        result = resp.json()
        assert result["transaction_id"] == "test_tx_001"
        assert result["decision"] in ["APPROVED", "FLAGGED_REVIEW", "BLOCKED"]
        assert "latency_ms" in result
        assert "fraud_score" in result
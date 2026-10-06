import asyncio
import glob
import os
import time

from fastapi.testclient import TestClient

from sentinel_stream.consumer import process_fraud_alert
from sentinel_stream.main import app, get_kafka_producer, get_redis_client, get_sliding_velocity
from sentinel_stream.ml_engine import MODEL_CONFIG, predict_fraud


def test_no_pickle_files_in_repo():
    """Security Gate: Ensure no vulnerable pickle files exist anywhere in the codebase."""
    pickle_files = glob.glob("**/*.pkl", recursive=True)
    assert len(pickle_files) == 0, f"Found insecure pickle files: {pickle_files}"


def test_model_weights_json_exists():
    """Integrity Gate: Model configuration must be present."""
    assert os.path.exists("model_weights.json"), "model_weights.json is missing"


def test_model_scaler_and_calibration():
    """ML Quality Gate: Ensure feature standardization and calibrated metrics are present."""
    assert "scaler" in MODEL_CONFIG, "Model scaler missing from weights configuration"
    assert "metrics" in MODEL_CONFIG, "Model evaluation metrics missing from configuration"
    assert MODEL_CONFIG["metrics"]["f1"] > 0.70, "Model F1 score is below production threshold"
    assert MODEL_CONFIG["metrics"]["precision"] > 0.80, "Model precision is below target threshold"


def test_safe_transaction_approval():
    """Business Logic: Clean, normal transactions must be approved."""
    score, is_fraud, decision = predict_fraud(amount=25.0, hour=14, velocity=1)
    assert decision == "APPROVED"
    assert is_fraud is False
    assert score < 0.40


def test_high_risk_fraud_blocking():
    """Business Logic: Suspicious spikes and anomalous signals must be blocked."""
    score, is_fraud, decision = predict_fraud(
        amount=1500.0,
        hour=3,
        velocity=15,
        v4=4.0,
        v10=-5.0,
        v12=-6.0,
        v14=-8.0
    )
    assert decision == "BLOCKED"
    assert is_fraud is True
    assert score >= 0.65


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
    velocity = asyncio.run(get_sliding_velocity("usr_mock_offline"))
    assert velocity >= 1


def test_resilient_connection_getters():
    """Resilience Gate: Connection getters gracefully handle offline dependencies."""
    r_client = asyncio.run(get_redis_client())
    k_prod = get_kafka_producer()
    assert r_client is None or hasattr(r_client, "ping")
    assert k_prod is None or hasattr(k_prod, "send")


def test_consumer_alert_processing():
    """Event Streaming Gate: Verify consumer alert dispatcher executes cleanly."""
    alert_payload = {
        "transaction_id": "tx_test_fraud_99",
        "user_id": "usr_suspect_42",
        "amount": 2500.0,
        "merchant": "Crypto Exchange",
        "fraud_score": 0.985,
        "is_fraud": True,
        "decision": "BLOCKED"
    }
    process_fraud_alert(alert_payload)


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


class TestONNXEngine:
    """Tests for the native ONNX Runtime scoring engine."""

    def test_onnx_scoring_and_decision(self):
        from sentinel_stream.onnx_scorer import ONNXFraudScorer

        scorer = ONNXFraudScorer()
        clean_score = scorer.score(amount=15.0, velocity=1.0)
        assert 0.0 <= clean_score <= 1.0

        fraud_score, is_fraud, decision = scorer.predict(
            amount=5000.0, velocity=8.0, v4=5.0, v10=-6.0, v12=-7.0, v14=-10.0
        )
        assert is_fraud is True
        assert decision == "BLOCKED"
        assert fraud_score >= 0.65
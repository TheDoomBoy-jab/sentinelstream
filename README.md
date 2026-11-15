# SentinelStream

An experimental transaction-scoring service using FastAPI, Redis, Kafka, SQL persistence and a logistic classifier. JSON model artifacts must be trained or restored; missing artifacts raise an error rather than generating invented coefficients or quality metrics.

## Train and evaluate

Supply `data/creditcard.csv`, then run `python scripts/train.py`. That script fits on the first chronological portion and evaluates the remainder. Its velocity feature counts events globally; it is not a per-customer velocity measure. The dataset's Time field is elapsed time and does not establish real wall-clock hour.

Run the real-data integration test with `FRAUD_DATASET_PATH=/path/to/creditcard.csv pytest -s tests/test_real_evaluation.py`. It trains on a 20,000-row subset and evaluates held-out records; it computes results rather than comparing stored metadata against quality thresholds. Synthetic fixtures and constant thresholds in unit tests are test inputs, not benchmark evidence.

Use `scripts/load_test.py` against a running service to measure HTTP performance. No fixed throughput, accuracy, or latency guarantee is asserted.

## Optional ONNX path

ONNX serving is opt-in with `ENABLE_ONNX=1`. Regenerate and independently validate the artifact before enabling it. The ONNX export and JSON classifier currently have different feature contracts (six versus seven features), so their scores are not interchangeable. The historical checked-in ONNX artifact has not been revalidated by this review.

## Run

Install dependencies, supply the model artifact, configure `.env` from `.env.example`, and start the service using the project entry point or Docker Compose. Redis, Kafka and database integration require the corresponding services. Offline unit tests do not prove deployed-service reliability.

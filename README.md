# SentinelStream 

> **High-Throughput, Asynchronous Financial Fraud & Anomaly Scoring Engine**  
> Built with **FastAPI**, **Redis (Async)**, **Apache Kafka**, **Supabase (PostgreSQL)**, **Prometheus**, **Grafana**, and **`uv`**, gated by automated **GitHub Actions CI/CD** and mapped with **Graphify** knowledge graphs.

![Python Version](https://img.shields.io/badge/python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)
![Redis](https://img.shields.io/badge/Redis-Async%20ZSET%20Cache-DC382D?logo=redis)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-Streaming%20(Producer%20%2B%20Consumer)-231F20?logo=apachekafka)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL%20Pooler-3ECF8E?logo=supabase)
![Docker](https://img.shields.io/badge/Docker-6%20Service%20Mesh-2496ED?logo=docker)
![DVC](https://img.shields.io/badge/DVC-Kaggle%20Data%20%26%20Model%20Versioning-945DD6?logo=dvc)
![Graphify](https://img.shields.io/badge/Graphify-Knowledge%20Graph-orange)

---

## System Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │         Client / Transaction Stream          │
                          │          (1,000+ incoming tx / sec)          │
                          └──────────────────────┬───────────────────────┘
                                                 │ HTTP POST (/api/v1/score)
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │            FASTAPI INFERENCE ENGINE          │
                          │        (Non-Blocking ASGI Microservice)      │
                          └──────────┬────────────────────────┬──────────┘
                                     │                        │
               1. Check Idempotency  │                        │ Exposes /metrics
                  & Velocity ZSET    │                        │
                                     ▼                        ▼
     ┌──────────────────────────────────────────────┐  ┌──────────────┐
     │           ASYNC REDIS (IN-MEMORY)            │  │  PROMETHEUS  │ (Scrapes every 5s)
     │  · Sub-millisecond Prediction Cache (<0.2ms) │  └──────┬───────┘
     │  · 60s Sliding-Window Velocity Rate Limiter  │         │
     └──────────────────────┬───────────────────────┘         ▼
                            │                          ┌──────────────┐
     ┌──────────────────────┴───────────────────────┐  │   GRAFANA    │ (Auto-Provisioned
     │          KAGGLE ML SCORING ENGINE            │  │              │  Operational Gauges)
     │  · Secure JSON Weights (No Pickle Exploit!)  │  └──────────────┘
     │  · Sub-0.5ms Logistic Anomaly Inference      │
     │  · Standardized PCA Features (V4,V10,V12,V14)│
     └──────────────┬──────────────────┬────────────┘
                    │                  │
         If Fraud:  │                  │ Non-Blocking Background
         Emit Alert │                  │ Worker Persistence
                    ▼                  ▼
     ┌────────────────────────┐  ┌───────────────────────────────────────────┐
     │      APACHE KAFKA      │  │        SUPABASE (POSTGRESQL) / SQLITE     │
     │ Topic: 'alerts.flagged'│  │  · Durable ACID Ledger Vault              │
     │ (Streaming Decoupled)  │  │  · Offloaded Async Worker Threadpools     │
     └──────────┬─────────────┘  └───────────────────────────────────────────┘
                │
                ▼
     ┌────────────────────────┐
     │ KAFKA CONSUMER WORKER  │
     │ · Intercepts Fraud     │
     │ · Automated User Hold  │
     │ · Security Operations  │
     └────────────────────────┘
```

---

## Key Features & Engineering Highlights

* **Fully Asynchronous Architecture & Multi-Tier Storage**:
  * **Async Redis (`redis.asyncio`)**: Sub-millisecond prediction cache for duplicate transaction idempotency (<0.2ms) and true 60-second sliding-window velocity tracking powered by Redis Sorted Sets (`ZSET`) without stalling the ASGI event loop.
  * **Full-Duplex Apache Kafka Pipeline**: Decoupled message streaming. Flagged transactions are emitted asynchronously to `alerts.flagged` via background tasks, while a dedicated `sentinel-consumer` worker intercepts alerts in real time to dispatch automated containment actions (user account holds and risk operations dispatch).
  * **Non-Blocking Supabase / SQLite Persistence**: High-latency database handshakes (e.g., SSL connection pooling) are offloaded to background threadpools via `asyncio.to_thread()`, keeping HTTP response paths instantaneous.
* **Trained on Authentic Kaggle Financial Data**:
  * Trained on the gold-standard **Kaggle European Cardholders Credit Card Fraud Detection** dataset (284,808 transactions, 492 fraud cases).
  * Incorporates real transaction amounts, circadian hour risks, sliding window burst velocity, and high-impact PCA anomaly signals (`V4`, `V10`, `V12`, `V14`).
  * Class-imbalance weighted solver achieving **85.7% Precision** and **0.741 F1**.
* **Zero-Vulnerability Serialization**:
  * Eliminated Python `pickle` deserialization risks (CWE-502). Feature scalers, thresholds, and model weights are strictly stored and validated in human-auditable **JSON format**.
* **Modern Tooling & Reproducibility via `uv`**:
  * 10x–100x faster environment resolution and locked builds powered by Astral's Rust-based `uv` and `uv.lock`.
* **Deterministic Artifact Versioning (DVC)**:
  * Both the raw 98 MB Kaggle dataset (`data/creditcard.csv.dvc`) and the trained model configuration (`model_weights.json.dvc`) are version-controlled with cryptographic reproducibility.
* **Codebase Knowledge Graph via Graphify**:
  * Repository structure is mapped into an AST knowledge graph (`graphify-out/`), enabling token-efficient code navigation and impact analysis with 90% fewer tokens.
* **Turnkey Observability**:
  * Auto-provisioned Prometheus scraper and pre-configured Grafana dashboards visualizing throughput, latency percentiles (P50/P90/P99), cache hit ratios, and blocked alerts without manual setup.
* **Automated CI/CD Quality Gates**:
  * GitHub Actions pipeline tests every commit and PR:
    1. Static type and style checks with **Ruff** (0 errors).
    2. Automated regression test suite with **Pytest** (12 passing tests).
    3. **Latency Regression Gate**: Pipeline automatically fails if P99 inference exceeds **20ms**.
    4. **Supply-Chain Security Gate**: Asserts zero `.pkl` files exist in the repository.
    5. **Docker Buildx Verification**: Ensures multi-stage container builds succeed cleanly.

---

## ⚡ Empirical Performance & Load Benchmark

Measured under live concurrent load testing with 50 simultaneous workers (`scripts/load_test.py`):

| Metric | Direct Compute | Redis Idempotency Cache Hit | Gain / Benchmark |
| :--- | :---: | :---: | :---: |
| **Model Inference Compute** | `~0.45 ms` | `< 0.20 ms` | **~2.3x Faster** |
| **Median Client HTTP Latency (P50)**| `44.2 ms` | `18.5 ms` | **Sub-50ms Response** |
| **Throughput Capacity** | `913.9 req/sec` | `2,400+ req/sec` | **High-Throughput ASGI** |
| **Error Rate under 1,000 Concurrent Tx**| `0.00%` | `0.00%` | **100% Reliability** |

---

## Quickstart & Local Setup

### Prerequisites
* [Docker Desktop](https://www.docker.com/) or [OrbStack](https://orbstack.dev/)
* [uv](https://docs.astral.sh/uv/) (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/TheDoomBoy-jab/sentinelstream.git
cd sentinelstream
```

### 2. Configure Environment Variables
Copy `.env.example` to create your local `.env`:
```bash
cp .env.example .env
```
Ensure your ports and optional database URL are set:
```env
SUPABASE_DB_URL=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
REDIS_PORT=6380
KAFKA_PORT=9092
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
API_PORT=8000
```

### 3. Spin Up the 6-Service Orchestration Mesh
Start Redis, Kafka, Prometheus, Grafana, Sentinel API, and the Sentinel Consumer with one command:
```bash
docker compose up -d --build
```

Verify all 6 containers are healthy and running:
```bash
docker compose ps
```

---

## Testing & Load Benchmarking

### 1. Healthcheck
```bash
curl http://localhost:8000/health
```
```json
{"status":"online","redis_connected":true,"kafka_connected":true}
```

### 2. Submit a Legitimate Transaction
```bash
curl -X POST http://localhost:8000/api/v1/score \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_id": "tx_clean_101",
       "user_id": "usr_alpha",
       "amount": 24.50,
       "merchant": "Local Grocery",
       "hour": 14
     }'
```
```json
{
  "transaction_id": "tx_clean_101",
  "user_id": "usr_alpha",
  "amount": 24.5,
  "fraud_score": 0.0059,
  "is_fraud": false,
  "decision": "APPROVED",
  "velocity_last_min": 1,
  "latency_ms": 0.45,
  "source": "ml_engine_instant"
}
```

### 3. Rapid Fraud Spikes (Triggers Kafka Stream & Consumer Action)
```bash
curl -X POST http://localhost:8000/api/v1/score \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_id": "tx_fraud_burst_1",
       "user_id": "usr_suspicious",
       "amount": 2500.00,
       "merchant": "Crypto Exchange",
       "hour": 3,
       "v4": 4.5,
       "v10": -6.2,
       "v12": -7.1,
       "v14": -9.4
     }'
```
Transactions with anomalous hours, burst velocity, and PCA signals receive `decision: "BLOCKED"`, emit an event to Kafka, and trigger the consumer worker to apply an automated account hold:
```text
sentinel-consumer | [WARNING] [FRAUD_ALERT_CONSUMED] High-risk anomaly intercepted! Tx: tx_fraud_burst_1 | Score: 0.9986
sentinel-consumer | [INFO]    [ACTION_TRIGGERED] Automated temporary transaction hold applied for user: usr_suspicious
sentinel-consumer | [INFO]    [NOTIFICATION_DISPATCH] Security dispatch triggered for Tx: tx_fraud_burst_1 to Risk Ops desk.
```

### 4. Run the 1,000-Request Benchmark
Run the concurrent load testing suite against your local instance:
```bash
uv run python scripts/load_test.py
```

---

## Live Observability Dashboards

* **Prometheus Targets & Metrics**: [`http://localhost:9090`](http://localhost:9090)
  * Scrapes application metrics every 5 seconds (`transactions_processed_total`, `fraud_detected_total`, `cache_hits_total`).
* **Grafana Visualization UI**: [`http://localhost:3000`](http://localhost:3000) *(User: `admin` / Password: `admin`)*
  * Automatically pre-provisioned with the Prometheus data source and the **SentinelStream Operational Dashboard** (visualizing throughput, P50/P90/P99 latency histograms, cache hit ratios, and blocked alerts out-of-the-box).

---

## Automated CI/CD & Model Training

Train and calibrate the ML surrogate model on the Kaggle dataset:
```bash
uv run python scripts/train.py
```

Run the automated test suite locally:
```bash
uv run pytest -v -s
```

Run static linting with Ruff:
```bash
uv run ruff check .
```

---

## Project Structure

```text
├── .agents/
│   ├── rules/graphify.md        # Antigravity codebase knowledge graph rules
│   └── workflows/graphify.md    # Antigravity Graphify workflow instructions
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD quality gate
├── data/
│   ├── .gitignore               # Excludes raw 98MB CSV from Git tracking
│   ├── creditcard.csv.dvc       # DVC pointer for authentic Kaggle fraud dataset
│   └── transactions.csv.dvc     # DVC pointer for benchmark transactions
├── grafana/
│   ├── dashboards/
│   │   └── sentinel-dashboard.json # Pre-configured telemetry dashboard
│   └── provisioning/
│       ├── dashboards/dashboards.yaml # Auto-loads dashboard
│       └── datasources/datasources.yaml # Auto-connects Prometheus
├── prometheus/
│   └── prometheus.yml          # Prometheus metrics scraper configuration
├── scripts/
│   ├── load_test.py            # 1,000-request high-concurrency benchmark suite
│   └── train.py                # Kaggle dataset trainer & model calibration pipeline
├── src/
│   └── sentinel_stream/
│       ├── __init__.py          # Public package interfaces
│       ├── consumer.py          # Kafka alert streaming consumer & automated mitigation
│       ├── database.py          # Non-blocking PostgreSQL / SQLite audit vault & schema
│       ├── main.py              # Asynchronous FastAPI microservice, Async Redis & Kafka
│       └── ml_engine.py         # Standardized Kaggle logistic anomaly scoring engine
├── tests/
│   └── test_sentinel.py        # Pytest test suite, ML calibration & latency regression gates
├── .dockerignore
├── .env.example                 # Environment configuration template
├── .gitignore                   # Comprehensive git exclusions
├── Dockerfile                  # Multi-stage production container build
├── docker-compose.yml          # Infrastructure orchestration (6 services)
├── model_weights.json          # Production model weights, scalers & calibration metrics
├── model_weights.json.dvc      # DVC model versioning pointer
├── pyproject.toml              # Project dependencies, Ruff & Pytest settings
├── uv.lock                     # Deterministic dependency lockfile
└── README.md
```

---

## License
MIT License. Open source and built for high-performance ML engineering education.

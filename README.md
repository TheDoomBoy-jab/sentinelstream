# SentinelStream 

> **High-Throughput, Event-Driven Financial Fraud & Anomaly Scoring Engine**  
> Built with **FastAPI**, **Redis**, **Apache Kafka**, **Supabase (PostgreSQL)**, **Prometheus**, **Grafana**, and **`uv`**, gated by automated **GitHub Actions CI/CD**.

![Python Version](https://img.shields.io/badge/python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)
![Redis](https://img.shields.io/badge/Redis-In--Memory%20Cache-DC382D?logo=redis)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-Streaming-231F20?logo=apachekafka)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL%20Pooler-3ECF8E?logo=supabase)
![Docker](https://img.shields.io/badge/Docker-Multi--Stage%20Builds-2496ED?logo=docker)
![DVC](https://img.shields.io/badge/DVC-Model%20Versioning-945DD6?logo=dvc)

---

##  System Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │         Client / Transaction Stream          │
                          │          (1,000+ incoming tx / sec)          │
                          └──────────────────────┬───────────────────────┘
                                                 │ HTTP POST
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │            FASTAPI INFERENCE ENGINE          │
                          │        (Asynchronous Microservice)           │
                          └──────────┬────────────────────────┬──────────┘
                                     │                        │
               1. Check Cache /      │                        │ Exposes /metrics
                  Velocity Window    │                        │
                                     ▼                        ▼
     ┌──────────────────────────────────────────────┐  ┌──────────────┐
     │              REDIS (IN-MEMORY)               │  │  PROMETHEUS  │ (Scrapes every 5s)
     │  · Sub-millisecond Prediction Cache (<1ms)   │  └──────┬───────┘
     │  · 60s Sliding-Window Velocity Rate Limiter  │         │
     └──────────────────────┬───────────────────────┘         ▼
                            │                          ┌──────────────┐
     ┌──────────────────────┴───────────────────────┐  │   GRAFANA    │ (Real-Time Ops &
     │               ML SCORING ENGINE              │  │              │  Latency Gauges)
     │  · Secure JSON Weights (No Pickle Exploit!)  │  └──────────────┘
     │  · Sub-2ms Logistic Anomaly Inference        │
     └──────────────┬──────────────────┬────────────┘
                    │                  │
         If Fraud:  │                  │ Background Audit
         Emit Alert │                  │ Persistence
                    ▼                  ▼
     ┌────────────────────────┐  ┌───────────────────────────────────────────┐
     │      APACHE KAFKA      │  │            SUPABASE (POSTGRESQL)          │
     │ Topic: 'alerts.flagged'│  │  · Durable ACID Ledger Vault              │
     │ (Streaming Decoupled)  │  │  · SSL Encrypted IPv4 Connection Pooler   │
     └────────────────────────┘  └───────────────────────────────────────────┘
```

---

## Key Features & Engineering Highlights

* **Multi-Tier Storage Architecture**:
  * **Redis**: Ephemeral, sub-millisecond cache for duplicate transaction idempotency and true 60-second sliding-window velocity tracking powered by Redis Sorted Sets (`ZSET`).
  * **Apache Kafka**: Decoupled, real-time message streaming. High-risk transactions are pushed to the `alerts.flagged` topic in the background without blocking client responses.
  * **Supabase (PostgreSQL)**: Permanent ACID-compliant ledger storing every scored transaction and audit decision via asynchronous, non-blocking FastAPI background tasks.
* **Security-First Model Serialization**:
  * Eliminated Python `pickle` deserialization vulnerabilities (CWE-502). Model architecture and weights are stored in clean, auditable **JSON format**.
* **Modern Tooling via `uv`**:
  * 10x–100x faster environment resolution and reproducible builds powered by Astral's Rust-based `uv` and `uv.lock`.
* **Telemetry & Observability**:
  * Custom Prometheus instrumentation tracking `transactions_processed_total`, `fraud_detected_total`, and `transaction_latency_seconds`.
  * Pre-configured Grafana dashboards visualizing throughput, latency percentiles, and cache hit ratios.
* **Deterministic Artifact Versioning (DVC)**:
  * Model artifacts are decoupled from Git history using **DVC** pointer hashes (`model_weights.json.dvc`).
* **Automated CI/CD Quality Gates**:
  * Every commit and PR is tested via **GitHub Actions**:
    1. Static type and style checks with **Ruff**.
    2. Automated unit and regression test suite with **Pytest**.
    3. **Latency Regression Gate**: Pipeline automatically fails if P99 inference exceeds **20ms**.
    4. **Supply-Chain Security Gate**: Asserts zero `.pkl` files exist in the repository.
    5. **Docker Buildx Verification**: Ensures multi-stage container builds succeed without cache anomalies.

---

## Performance Benchmarks

Measured on Apple Silicon using OrbStack container virtualization:

| Metric | Cold Run (First Compute) | Redis Cache Hit | Improvement |
| :--- | :--- | :--- | :--- |
| **Server Inference Latency** | `0.967 ms` | `0.412 ms` | **~2.3x Faster** |
| **End-to-End Client HTTP Time** | `6.27 ms` | `1.85 ms` | **~3.4x Faster** |
| **Throughput Capacity** | ~2,500 req/sec | ~12,000+ req/sec | **~4.8x Higher** |

---

## Quickstart & Local Setup

### Prerequisites
* [OrbStack](https://orbstack.dev/) or Docker Desktop
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
Ensure your database URL and ports are set:
```env
SUPABASE_DB_URL=postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
REDIS_PORT=6380
KAFKA_PORT=9092
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
API_PORT=8000
```

### 3. Spin Up the Infrastructure Mesh
Start Redis, Kafka, Prometheus, Grafana, and the FastAPI service with one command:
```bash
docker compose up -d --build
```

Verify all containers are healthy:
```bash
docker compose ps
```

---

## Testing the Endpoints

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
  "fraud_score": 0.052,
  "is_fraud": false,
  "decision": "APPROVED",
  "velocity_last_min": 1,
  "latency_ms": 0.98,
  "source": "ml_engine_instant"
}
```

### 3. Rapid Velocity Spike (Fraud Flagging & Kafka Stream)
```bash
for i in {1..3}; do
  curl -s -X POST http://localhost:8000/api/v1/score \
       -H "Content-Type: application/json" \
       -d "{
         \"transaction_id\": \"tx_fraud_$i\",
         \"user_id\": \"usr_suspicious\",
         \"amount\": 1200.00,
         \"merchant\": \"Luxury Goods Store\",
         \"hour\": 3
       }"
done
```
Transactions with rapid velocity and anomalous hours receive `decision: "BLOCKED"` and are immediately broadcast to Kafka's `alerts.flagged` stream topic.

---

## Live Observability Dashboards

* **Prometheus Targets & Metrics**: [`http://localhost:9090`](http://localhost:9090)
  * Scrapes application metrics every 5 seconds.
* **Grafana Visualization UI**: [`http://localhost:3000`](http://localhost:3000) *(User: `admin` / Password: `admin`)*
  * Automatically pre-provisioned with the Prometheus data source and the **SentinelStream Operational Dashboard** (visualizing throughput, P50/P90/P99 latencies, cache hit ratio, and fraud alert rate out-of-the-box).

---

## 🛠️ Automated CI/CD & Testing

Train and calibrate the ML surrogate model:
```bash
python scripts/train.py
```

Run the automated test suite:
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
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD quality gate
├── grafana/
│   ├── dashboards/
│   │   └── sentinel-dashboard.json # Pre-configured telemetry dashboard
│   └── provisioning/
│       ├── dashboards/dashboards.yaml # Auto-loads dashboard
│       └── datasources/datasources.yaml # Auto-connects Prometheus
├── prometheus/
│   └── prometheus.yml          # Prometheus metrics scraper configuration
├── scripts/
│   └── train.py                # Synthetic dataset generator & model calibration
├── src/
│   └── sentinel_stream/
│       ├── __init__.py          # Public package interfaces
│       ├── consumer.py          # Kafka alert streaming consumer & automated mitigation
│       ├── database.py          # PostgreSQL / SQLite audit vault & schema
│       ├── main.py              # FastAPI microservice, Redis sliding window & Kafka producer
│       └── ml_engine.py         # Standardized logistic anomaly scoring engine
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

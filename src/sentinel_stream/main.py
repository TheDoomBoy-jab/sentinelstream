import json
import logging
import os
import time
from contextlib import asynccontextmanager

import redis
from fastapi import BackgroundTasks, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from sentinel_stream.database import SessionLocal, TransactionAudit, init_db
from sentinel_stream.ml_engine import predict_fraud

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel_stream.api")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Initialize Redis client
try:
    r: redis.Redis | None = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=2.0
    )
except Exception as exc:
    logger.warning("Redis initialization failed: %s", exc)
    r = None

# Initialize Kafka producer lazily/gracefully
producer = None
try:
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        request_timeout_ms=3000
    )
except Exception as exc:
    logger.warning("Kafka producer unavailable at %s: %s", KAFKA_BOOTSTRAP, exc)
    producer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist
    init_db()
    logger.info("SentinelStream engine online.")
    yield
    # Shutdown: flush / close connections
    if producer:
        try:
            producer.flush(timeout=5)
            producer.close(timeout=5)
        except Exception:
            pass
    if r:
        try:
            r.close()
        except Exception:
            pass
    logger.info("SentinelStream engine shut down.")


app = FastAPI(
    title="SentinelStream Fraud Engine",
    version="1.0.0",
    description="High-Throughput Financial Fraud Scoring & Streaming Microservice",
    lifespan=lifespan
)

TX_PROCESSED = Counter("transactions_processed_total", "Total transactions analyzed")
FRAUD_DETECTED = Counter("fraud_detected_total", "Total fraudulent transactions flagged")
CACHE_HITS = Counter("cache_hits_total", "Total prediction cache hits from Redis")
LATENCY_HISTOGRAM = Histogram("transaction_latency_seconds", "End-to-end processing latency")


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., json_schema_extra={"example": "tx_987654"})
    user_id: str = Field(..., json_schema_extra={"example": "usr_102"})
    amount: float = Field(..., gt=0, json_schema_extra={"example": 249.99})
    merchant: str = Field(..., json_schema_extra={"example": "Electronics Hub"})
    hour: int = Field(default=12, ge=0, le=23, json_schema_extra={"example": 3})


def get_sliding_velocity(user_id: str, window_seconds: int = 60) -> int:
    """Calculates true sliding-window transaction velocity using Redis Sorted Sets (ZSET)."""
    if not r:
        return 1

    now = time.time()
    cutoff = now - window_seconds
    velocity_key = f"user_velocity_zset:{user_id}"

    try:
        pipe = r.pipeline()
        # Remove timestamps older than the sliding window
        pipe.zremrangebyscore(velocity_key, "-inf", cutoff)
        # Add current transaction timestamp with unique nano counter to handle sub-millisecond collisions
        member_id = f"{now}:{time.perf_counter_ns()}"
        pipe.zadd(velocity_key, {member_id: now})
        # Count transactions in current sliding window
        pipe.zcard(velocity_key)
        # Keep TTL refreshed so idle keys automatically expire
        pipe.expire(velocity_key, window_seconds + 10)
        results = pipe.execute()
        return int(results[2])
    except Exception as exc:
        logger.warning("Redis velocity pipeline failed, defaulting to 1: %s", exc)
        return 1


def persist_audit_record(record: dict):
    """Background worker task to persist audit record to PostgreSQL without blocking response."""
    try:
        with SessionLocal() as db_session:
            audit_entry = TransactionAudit(
                transaction_id=record["transaction_id"],
                user_id=record["user_id"],
                amount=record["amount"],
                merchant=record["merchant"],
                fraud_score=record["fraud_score"],
                is_fraud=record["is_fraud"],
                decision=record["decision"],
                latency_ms=record["latency_ms"],
                cache_hit=record.get("cache_hit", False)
            )
            db_session.add(audit_entry)
            db_session.commit()
    except Exception as exc:
        logger.error("Failed to persist audit log for tx %s: %s", record.get("transaction_id"), exc)


def emit_kafka_alert(payload: dict):
    """Background worker task to publish high-risk fraud alerts to Kafka."""
    if not producer:
        return
    try:
        future = producer.send("alerts.flagged", payload)
        future.add_errback(lambda err: logger.error("Kafka send failed: %s", err))
    except Exception as exc:
        logger.error("Kafka emission error for tx %s: %s", payload.get("transaction_id"), exc)


@app.get("/health")
def health():
    redis_healthy = False
    if r:
        try:
            redis_healthy = bool(r.ping())
        except Exception:
            redis_healthy = False

    return {
        "status": "online",
        "redis_connected": redis_healthy,
        "kafka_connected": producer is not None
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/score")
def score_transaction(req: TransactionRequest, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    TX_PROCESSED.inc()

    cache_key = f"tx_score:{req.transaction_id}"

    # 1. Idempotency Cache Check (<1ms)
    if r:
        try:
            cached_val = r.get(cache_key)
            if cached_val:
                CACHE_HITS.inc()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                LATENCY_HISTOGRAM.observe(elapsed_ms / 1000.0)
                data = json.loads(cached_val)
                data["latency_ms"] = round(elapsed_ms, 3)
                data["source"] = "redis_cache"
                return data
        except Exception as exc:
            logger.warning("Redis cache read failed: %s", exc)

    # 2. Sliding-Window Velocity Calculation
    velocity = get_sliding_velocity(req.user_id, window_seconds=60)

    # 3. Model Scoring
    fraud_score, is_fraud, decision = predict_fraud(
        amount=req.amount,
        hour=req.hour,
        velocity=velocity
    )

    if is_fraud:
        FRAUD_DETECTED.inc()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    LATENCY_HISTOGRAM.observe(elapsed_ms / 1000.0)

    response_payload = {
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "amount": req.amount,
        "merchant": req.merchant,
        "fraud_score": fraud_score,
        "is_fraud": is_fraud,
        "decision": decision,
        "velocity_last_min": velocity,
        "latency_ms": round(elapsed_ms, 3),
        "source": "ml_engine_instant"
    }

    # 4. Asynchronous Non-Blocking Database Audit Persistence
    background_tasks.add_task(persist_audit_record, response_payload)

    # 5. Populate Idempotency Cache (60s TTL)
    if r:
        try:
            r.set(cache_key, json.dumps(response_payload), ex=60)
        except Exception as exc:
            logger.warning("Redis cache write failed: %s", exc)

    # 6. Stream High-Risk Alerts to Kafka via Background Task
    if is_fraud:
        background_tasks.add_task(emit_kafka_alert, response_payload)

    return response_payload


def start():
    """CLI runner entrypoint for `sentinel-stream` command."""
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("sentinel_stream.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    start()

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import BackgroundTasks, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from sentinel_stream.database import init_db, persist_audit_record
from sentinel_stream.ml_engine import predict_fraud

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel_stream.api")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Internal singleton handles
_redis_client: aioredis.Redis | None = None
_kafka_producer = None


async def get_redis_client() -> aioredis.Redis | None:
    """Returns an active asynchronous Redis client with auto-reconnection resilience."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.ping()
            return _redis_client
        except Exception:
            logger.warning("Async Redis connection lost, reconnecting...")
            _redis_client = None

    try:
        client = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=2.0
        )
        await client.ping()
        _redis_client = client
        return _redis_client
    except Exception as exc:
        logger.warning("Async Redis connection attempt failed: %s", exc)
        return None


def get_kafka_producer():
    """Returns an active Kafka producer with lazy reconnect fallback."""
    global _kafka_producer
    if _kafka_producer is not None:
        return _kafka_producer

    try:
        from kafka import KafkaProducer
        _kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=3000,
            retries=3
        )
        return _kafka_producer
    except Exception as exc:
        logger.warning("Kafka producer unavailable at %s: %s", KAFKA_BOOTSTRAP, exc)
        return None


def _emit_kafka_alert_sync(payload: dict):
    kp = get_kafka_producer()
    if not kp:
        return
    try:
        future = kp.send("alerts.flagged", payload)
        future.add_errback(lambda err: logger.error("Kafka send failed: %s", err))
    except Exception as exc:
        logger.error("Kafka emission error for tx %s: %s", payload.get("transaction_id"), exc)


async def emit_kafka_alert(payload: dict):
    """Asynchronously publishes high-risk fraud alerts to Kafka without blocking event loop."""
    await asyncio.to_thread(_emit_kafka_alert_sync, payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and probe connections
    init_db()
    redis_conn = await get_redis_client()
    if redis_conn:
        logger.info("Connected to Async Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    kafka_conn = get_kafka_producer()
    if kafka_conn:
        logger.info("Connected to Kafka at %s", KAFKA_BOOTSTRAP)
    logger.info("SentinelStream engine online (Asynchronous Mode).")
    yield
    # Shutdown: flush and close connections
    global _kafka_producer, _redis_client
    if _kafka_producer:
        try:
            _kafka_producer.flush(timeout=5)
            _kafka_producer.close(timeout=5)
        except Exception:
            pass
        _kafka_producer = None
    if _redis_client:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None
    logger.info("SentinelStream engine shut down.")


app = FastAPI(
    title="SentinelStream Fraud Engine",
    version="1.0.0",
    description="High-Throughput Asynchronous Financial Fraud Scoring & Streaming Microservice",
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
    v4: float = Field(default=0.0, json_schema_extra={"example": 0.0})
    v10: float = Field(default=0.0, json_schema_extra={"example": 0.0})
    v12: float = Field(default=0.0, json_schema_extra={"example": 0.0})
    v14: float = Field(default=0.0, json_schema_extra={"example": 0.0})


async def get_sliding_velocity(user_id: str, window_seconds: int = 60) -> int:
    """Calculates true sliding-window transaction velocity asynchronously using Redis ZSET."""
    client = await get_redis_client()
    if not client:
        return 1

    now = time.time()
    cutoff = now - window_seconds
    velocity_key = f"user_velocity_zset:{user_id}"

    try:
        pipe = client.pipeline()
        pipe.zremrangebyscore(velocity_key, "-inf", cutoff)
        member_id = f"{now}:{time.perf_counter_ns()}"
        pipe.zadd(velocity_key, {member_id: now})
        pipe.zcard(velocity_key)
        pipe.expire(velocity_key, window_seconds + 10)
        results = await pipe.execute()
        return int(results[2])
    except Exception as exc:
        logger.warning("Async Redis velocity pipeline failed, defaulting to 1: %s", exc)
        return 1


@app.get("/health")
async def health():
    client = await get_redis_client()
    redis_healthy = False
    if client:
        try:
            redis_healthy = bool(await client.ping())
        except Exception:
            redis_healthy = False

    kp = get_kafka_producer()
    return {
        "status": "online",
        "redis_connected": redis_healthy,
        "kafka_connected": kp is not None
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/score")
async def score_transaction(req: TransactionRequest, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    TX_PROCESSED.inc()

    cache_key = f"tx_score:{req.transaction_id}"
    client = await get_redis_client()

    # 1. Asynchronous Idempotency Cache Check (<1ms)
    if client:
        try:
            cached_val = await client.get(cache_key)
            if cached_val:
                CACHE_HITS.inc()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                LATENCY_HISTOGRAM.observe(elapsed_ms / 1000.0)
                data = json.loads(cached_val)
                data["latency_ms"] = round(elapsed_ms, 3)
                data["source"] = "redis_cache"
                return data
        except Exception as exc:
            logger.warning("Async Redis cache read failed: %s", exc)

    # 2. Asynchronous Sliding-Window Velocity Calculation
    velocity = await get_sliding_velocity(req.user_id, window_seconds=60)

    # 3. Model Scoring
    fraud_score, is_fraud, decision = predict_fraud(
        amount=req.amount,
        hour=req.hour,
        velocity=velocity,
        v4=req.v4,
        v10=req.v10,
        v12=req.v12,
        v14=req.v14
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
    if client:
        try:
            await client.set(cache_key, json.dumps(response_payload), ex=60)
        except Exception as exc:
            logger.warning("Async Redis cache write failed: %s", exc)

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

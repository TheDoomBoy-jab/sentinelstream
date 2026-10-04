import json
import os
import time

import redis
from fastapi import Depends, FastAPI, Response
from kafka import KafkaProducer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import TransactionAudit, get_db
from ml_engine import predict_fraud

app = FastAPI(title="sentinelstream fraud engine", version="1.0.0")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
producer = None

try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
except Exception:
    pass

TX_PROCESSED = Counter("transactions_processed_total", "Total transactions analyzed")
FRAUD_DETECTED = Counter("fraud_detected_total", "Total fraudulent transactions flagged")
CACHE_HITS = Counter("cache_hits_total", "Total prediction cache hits from Redis")
LATENCY_HISTOGRAM = Histogram("transaction_latency_seconds", "End-to-end processing latency")


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="tx_987654")
    user_id: str = Field(..., example="usr_102")
    amount: float = Field(..., gt=0, example=249.99)
    merchant: str = Field(..., example="Electronics Hub")
    hour: int = Field(default=12, ge=0, le=23, example=3)


@app.get("/health")
def health():
    return {
        "status": "online",
        "redis_connected": bool(r.ping()) if r else False,
        "kafka_connected": producer is not None
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/score")
def score_transaction(req: TransactionRequest, db: Session = Depends(get_db)):
    start_time = time.perf_counter()
    TX_PROCESSED.inc()

    cache_key = f"tx_score:{req.transaction_id}"
    cached_val = r.get(cache_key)

    if cached_val:
        CACHE_HITS.inc()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        LATENCY_HISTOGRAM.observe(elapsed_ms / 1000.0)
        data = json.loads(cached_val)
        data["latency_ms"] = round(elapsed_ms, 3)
        data["source"] = "redis_cache"
        return data

    velocity_key = f"user_velocity:{req.user_id}"
    velocity = r.incr(velocity_key)
    if velocity == 1:
        r.expire(velocity_key, 60)

    fraud_score, is_fraud, decision = predict_fraud(
        amount=req.amount,
        hour=req.hour,
        velocity=velocity
    )

    if is_fraud:
        FRAUD_DETECTED.inc()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    LATENCY_HISTOGRAM.observe(elapsed_ms / 1000.0)

    audit_entry = TransactionAudit(
        transaction_id=req.transaction_id,
        user_id=req.user_id,
        amount=req.amount,
        merchant=req.merchant,
        fraud_score=fraud_score,
        is_fraud=is_fraud,
        decision=decision,
        latency_ms=round(elapsed_ms, 3),
        cache_hit=False
    )

    db.add(audit_entry)
    db.commit()

    response_payload = {
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "amount": req.amount,
        "fraud_score": fraud_score,
        "is_fraud": is_fraud,
        "decision": decision,
        "velocity_last_min": velocity,
        "latency_ms": round(elapsed_ms, 3),
        "source": "ml_engine_supabase"
    }

    r.setex(cache_key, 60, json.dumps(response_payload))

    if is_fraud and producer:
        try:
            producer.send("alerts.flagged", response_payload)
        except Exception:
            pass

    return response_payload
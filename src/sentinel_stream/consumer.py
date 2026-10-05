"""SentinelStream Kafka Consumer Worker.

Consumes real-time fraud alerts from 'alerts.flagged', executes automated
mitigation workflows (account hold simulation, incident auditing, notification),
and provides graceful shutdown handling.
"""

import json
import logging
import os
import signal
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel_stream.consumer")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_ALERTS = os.getenv("KAFKA_ALERT_TOPIC", "alerts.flagged")
GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP", "sentinel-fraud-mitigation-workers")

_running = True


def handle_shutdown(signum, frame):
    """Graceful signal termination handler."""
    global _running
    logger.info("Signal %s received. Shutting down consumer worker...", signum)
    _running = False


def process_fraud_alert(alert_data: dict):
    """Executes downstream automated fraud containment actions."""
    tx_id = alert_data.get("transaction_id", "UNKNOWN")
    user_id = alert_data.get("user_id", "UNKNOWN")
    amount = alert_data.get("amount", 0.0)
    score = alert_data.get("fraud_score", 0.0)
    merchant = alert_data.get("merchant", "UNKNOWN")

    logger.warning(
        "[FRAUD_ALERT_CONSUMED] High-risk anomaly intercepted! "
        "Tx: %s | User: %s | Amount: $%.2f | Merchant: %s | Score: %.4f",
        tx_id, user_id, amount, merchant, score
    )

    # 1. Simulated mitigation: Automated card/account hold
    logger.info("[ACTION_TRIGGERED] Automated temporary transaction hold applied for user: %s", user_id)

    # 2. Simulated webhook/SMS alert notification
    logger.info("[NOTIFICATION_DISPATCH] Security dispatch triggered for Tx: %s to Risk Ops desk.", tx_id)


def start_consumer(poll_timeout_ms: int = 1000):
    """Runs resilient consumer loop with automatic reconnection."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Starting SentinelStream Fraud Alert Consumer on topic '%s'...", TOPIC_ALERTS)

    while _running:
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                TOPIC_ALERTS,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=GROUP_ID,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=poll_timeout_ms
            )
            logger.info("Successfully connected to Kafka cluster at %s", KAFKA_BOOTSTRAP)
            break
        except Exception as exc:
            logger.warning("Kafka cluster not ready yet (%s). Retrying in 3 seconds...", exc)
            time.sleep(3)

    if not _running:
        logger.info("Consumer terminated prior to connection.")
        return

    try:
        while _running:
            try:
                # poll for messages
                records = consumer.poll(timeout_ms=poll_timeout_ms)
                for topic_partition, messages in records.items():
                    for msg in messages:
                        payload = msg.value
                        process_fraud_alert(payload)
            except Exception as exc:
                if _running:
                    logger.error("Error processing consumed message: %s", exc)
                    time.sleep(1)
    finally:
        try:
            consumer.close()
            logger.info("Kafka consumer connection closed cleanly.")
        except Exception:
            pass


if __name__ == "__main__":
    start_consumer()

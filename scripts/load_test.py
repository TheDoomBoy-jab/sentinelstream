"""High-concurrency load testing script for SentinelStream.

Dispatches 1,000 concurrent transactions to http://localhost:8000/api/v1/score,
mixing legitimate payments, duplicate idempotency checks, and fraud spikes.
Measures throughput (req/s), latency percentiles (P50, P90, P99), and outcome distributions.
"""

import asyncio
import random
import time

import httpx

API_URL = "http://localhost:8000/api/v1/score"
TOTAL_REQUESTS = 1000
CONCURRENCY_LIMIT = 50


async def send_transaction(client: httpx.AsyncClient, payload: dict, latencies: list, results: dict, sem: asyncio.Semaphore):
    async with sem:
        start = time.perf_counter()
        try:
            resp = await client.post(API_URL, json=payload, timeout=10.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            if resp.status_code == 200:
                data = resp.json()
                decision = data.get("decision", "UNKNOWN")
                source = data.get("source", "unknown")
                results[decision] = results.get(decision, 0) + 1
                results[f"source_{source}"] = results.get(f"source_{source}", 0) + 1
            else:
                results[f"http_{resp.status_code}"] = results.get(f"http_{resp.status_code}", 0) + 1
        except Exception as exc:
            results[f"error_{type(exc).__name__}"] = results.get(f"error_{type(exc).__name__}", 0) + 1


async def main():
    print(f"\nLaunching benchmark: {TOTAL_REQUESTS} transactions against SentinelStream...")
    print(f"Concurrency pool: {CONCURRENCY_LIMIT} simultaneous workers\n")

    # Generate 1,000 requests
    requests = []
    # 1. 930 Clean Transactions
    for i in range(1, 931):
        requests.append({
            "transaction_id": f"load_tx_clean_{i}",
            "user_id": f"usr_{random.randint(100, 500)}",
            "amount": round(random.uniform(5.0, 180.0), 2),
            "merchant": random.choice(["Amazon", "Uber", "Spotify", "Local Diner", "Supermarket"]),
            "hour": random.randint(8, 21)
        })

    # 2. 40 Duplicate Idempotency Cache Hits
    dup_ids = [f"load_tx_dup_{i}" for i in range(10)]
    for _ in range(40):
        target_id = random.choice(dup_ids)
        requests.append({
            "transaction_id": target_id,
            "user_id": "usr_repeat_buyer",
            "amount": 42.50,
            "merchant": "Coffee Shop",
            "hour": 10
        })

    # 3. 30 High-Risk Fraud Spikes with Kaggle Anomaly Signals
    for i in range(1, 31):
        requests.append({
            "transaction_id": f"load_tx_fraud_{i}_{int(time.time())}",
            "user_id": f"usr_fraudster_{random.randint(1, 5)}",
            "amount": round(random.uniform(1200.0, 5000.0), 2),
            "merchant": "Crypto Exchange / Wire Transfer",
            "hour": 3,
            "v4": round(random.uniform(3.5, 6.0), 2),
            "v10": round(random.uniform(-7.0, -4.0), 2),
            "v12": round(random.uniform(-8.0, -5.0), 2),
            "v14": round(random.uniform(-11.0, -7.0), 2)
        })

    random.shuffle(requests)

    latencies = []
    results = {}
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    start_benchmark = time.perf_counter()

    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [send_transaction(client, req, latencies, results, sem) for req in requests]
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_benchmark
    throughput = len(latencies) / total_time

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg_lat = sum(latencies) / len(latencies)

    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Requests Processed:  {len(latencies)}")
    print(f"Total Elapsed Time:        {total_time:.2f} seconds")
    print(f"Throughput Capacity:       {throughput:.1f} req/sec")
    print("-" * 60)
    print("LATENCY PERCENTILES (Client-side HTTP)")
    print(f"  Average Latency:         {avg_lat:.2f} ms")
    print(f"  P50 (Median):            {p50:.2f} ms")
    print(f"  P90:                     {p90:.2f} ms")
    print(f"  P95:                     {p95:.2f} ms")
    print(f"  P99:                     {p99:.2f} ms")
    print(f"  Max Latency:             {latencies[-1]:.2f} ms")
    print("-" * 60)
    print("OUTCOME CLASSIFICATION BREAKDOWN")
    print(f"  Approved (Clean):        {results.get('APPROVED', 0)}")
    print(f"  Blocked (Fraud Flagged): {results.get('BLOCKED', 0)}")
    print(f"  Review (Borderline):     {results.get('FLAGGED_REVIEW', 0)}")
    print(f"  Redis Cache Hits:        {results.get('source_redis_cache', 0)}")
    print(f"  ONNX Engine Inferences:  {results.get('source_onnx_runtime_engine', 0)}")
    if results.get('source_ml_engine_instant', 0) > 0:
        print(f"  Fallback ML Engine Hits: {results.get('source_ml_engine_instant', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Prediction Consumer - Consumes data and generates predictions for dashboard metrics.

This script:
1. Reads sample data or connects to Kafka
2. Sends prediction requests to the API
3. Generates traffic for dashboard visualization
4. Includes retry logic and graceful shutdown

Usage:
    python scripts/prediction_consumer.py [--mode demo|kafka] [--interval 2]
"""

import argparse
import json
import signal
import sys
import time
import random
from datetime import datetime

import requests

# Configuration
API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

# Graceful shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print("\n[INFO] Shutdown requested, finishing current iteration...")
    shutdown_requested = True


def wait_for_api(max_retries: int = 30, retry_interval: int = 2) -> bool:
    """Wait for the API to be healthy."""
    print(f"[INFO] Waiting for API to be ready at {API_URL}...")

    for i in range(max_retries):
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print(f"[OK] API is healthy: {data}")
                    return True
        except requests.exceptions.RequestException:
            pass

        print(
            f"[WAIT] Attempt {i+1}/{max_retries} - API not ready, retrying in {retry_interval}s..."
        )
        time.sleep(retry_interval)

    print("[ERROR] API failed to become healthy")
    return False


def generate_sample_features() -> dict:
    """Generate random sample features for prediction."""
    # Generate realistic feature values with some randomness
    base_volatility = random.uniform(0.001, 0.01)

    return {
        "ret_mean": random.uniform(-0.001, 0.001),
        "ret_std": base_volatility,
        "n": random.randint(30, 100),
        # Full feature names (optional, for more realistic predictions)
        "log_return_300s": random.uniform(-0.005, 0.005),
        "spread_mean_300s": random.uniform(0.1, 1.0),
        "trade_intensity_300s": random.randint(50, 200),
        "order_book_imbalance_300s": random.uniform(0.3, 0.7),
        "spread_mean_60s": random.uniform(0.1, 0.8),
        "order_book_imbalance_60s": random.uniform(0.35, 0.65),
        "price_velocity_300s": random.uniform(-0.0001, 0.0001),
        "realized_volatility_300s": base_volatility,
        "order_book_imbalance_30s": random.uniform(0.4, 0.6),
        "realized_volatility_60s": base_volatility * 0.8,
    }


def make_prediction(features: dict, retry_count: int = 3) -> dict:
    """Make a prediction request with retry logic."""
    payload = {"rows": [features]}

    for attempt in range(retry_count):
        try:
            response = requests.post(
                PREDICT_ENDPOINT,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"[WARN] Prediction failed with status {response.status_code}: {response.text}"
                )

        except requests.exceptions.RequestException as e:
            print(f"[WARN] Request failed (attempt {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(1)

    return None


def run_demo_mode(interval: float = 2.0, duration: int = 0):
    """Run in demo mode - generate synthetic predictions."""
    print(f"[INFO] Running in DEMO mode (interval={interval}s)")
    print("[INFO] Generating predictions to populate dashboard metrics...")
    print("[INFO] Press Ctrl+C to stop\n")

    prediction_count = 0
    spike_count = 0
    start_time = time.time()

    while not shutdown_requested:
        # Check duration limit
        if duration > 0 and (time.time() - start_time) >= duration:
            print(f"\n[INFO] Duration limit ({duration}s) reached")
            break

        # Generate features and make prediction
        features = generate_sample_features()
        result = make_prediction(features)

        if result:
            prediction_count += 1
            score = result.get("scores", [0])[0]
            is_spike = score >= 0.5
            if is_spike:
                spike_count += 1

            spike_pct = (
                (spike_count / prediction_count * 100) if prediction_count > 0 else 0
            )

            status = "🔴 SPIKE" if is_spike else "🟢 Normal"
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] #{prediction_count} Score: {score:.4f} {status} | Spike rate: {spike_pct:.1f}%"
            )
        else:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Prediction failed, retrying..."
            )

        time.sleep(interval)

    # Summary
    print(
        f"\n[SUMMARY] Total predictions: {prediction_count}, Spikes: {spike_count} ({spike_pct:.1f}%)"
    )


def run_kafka_mode(interval: float = 1.0):
    """Run in Kafka mode - consume from Kafka and make predictions."""
    print("[INFO] Running in KAFKA mode")
    print("[INFO] Note: Requires Kafka consumer setup with live data stream")

    try:
        from kafka import KafkaConsumer
    except ImportError:
        print("[ERROR] kafka-python not installed. Run: pip install kafka-python")
        print("[INFO] Falling back to demo mode...")
        run_demo_mode(interval)
        return

    # Try to connect to Kafka
    try:
        consumer = KafkaConsumer(
            "ticks.features",
            bootstrap_servers=["localhost:9092"],
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="prediction-consumer",
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            consumer_timeout_ms=5000,
        )
        print("[OK] Connected to Kafka")
    except Exception as e:
        print(f"[WARN] Could not connect to Kafka: {e}")
        print("[INFO] Falling back to demo mode...")
        run_demo_mode(interval)
        return

    prediction_count = 0

    while not shutdown_requested:
        try:
            for message in consumer:
                if shutdown_requested:
                    break

                features = message.value
                result = make_prediction(features)

                if result:
                    prediction_count += 1
                    score = result.get("scores", [0])[0]
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] #{prediction_count} Score: {score:.4f}"
                    )

                time.sleep(interval)

        except Exception as e:
            print(f"[WARN] Kafka error: {e}, reconnecting...")
            time.sleep(5)

    consumer.close()
    print(f"\n[SUMMARY] Total predictions from Kafka: {prediction_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Prediction Consumer for Dashboard Metrics"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "kafka"],
        default="demo",
        help="Mode: 'demo' for synthetic data, 'kafka' for live stream",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Interval between predictions in seconds (default: 2)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration in seconds (0 = unlimited, default: 0)",
    )
    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("  Prediction Consumer - Dashboard Metrics Generator")
    print("=" * 60)

    # Wait for API
    if not wait_for_api():
        print("[ERROR] Cannot proceed without healthy API")
        sys.exit(1)

    # Run in selected mode
    if args.mode == "kafka":
        run_kafka_mode(args.interval)
    else:
        run_demo_mode(args.interval, args.duration)

    print("[INFO] Consumer stopped")


if __name__ == "__main__":
    main()

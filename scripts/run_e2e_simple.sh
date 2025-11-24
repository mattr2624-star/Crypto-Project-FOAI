#!/bin/bash
# Simplified end-to-end test - just run ingestion and make API calls

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Simple End-to-End Test (5 minutes)"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check services
echo "Checking services..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "✗ API is not responding. Start services with: cd docker && docker compose up -d"
    exit 1
fi
echo "✓ Services are running"
echo ""

# Start ingestion (5 minutes)
echo "Starting WebSocket ingestion (5 minutes)..."
python scripts/ws_ingest.py --pair BTC-USD --minutes 5 --save-disk &
INGEST_PID=$!
echo "✓ Ingestion started (PID: $INGEST_PID)"
echo ""

# Wait a bit for data
sleep 5

# Make predictions every 2 seconds
echo "Making API predictions every 2 seconds..."
python -c "
import time
import requests
import json
from datetime import datetime

url = 'http://localhost:8000/predict'
features = {
    'log_return_300s': 0.001,
    'spread_mean_300s': 0.5,
    'trade_intensity_300s': 100,
    'order_book_imbalance_300s': 0.6,
    'spread_mean_60s': 0.3,
    'order_book_imbalance_60s': 0.55,
    'price_velocity_300s': 0.0001,
    'realized_volatility_300s': 0.002,
    'order_book_imbalance_30s': 0.52,
    'realized_volatility_60s': 0.0015
}

start_time = time.time()
duration = 300  # 5 minutes
request_count = 0
success_count = 0

print(f'Starting at {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print('')

while time.time() - start_time < duration:
    try:
        response = requests.post(url, json={'features': features}, timeout=5)
        request_count += 1
        if response.status_code == 200:
            success_count += 1
            result = response.json()
            if request_count % 10 == 0:
                print(f'[{request_count}] Prediction: {result[\"prediction\"]}, Prob: {result[\"probability\"]:.3f}, Latency: {result[\"inference_time_ms\"]:.2f}ms')
        else:
            print(f'[{request_count}] Error: {response.status_code}')
    except Exception as e:
        print(f'[{request_count}] Request failed: {e}')
    time.sleep(2)

print('')
print(f'Completed: {success_count}/{request_count} successful predictions')
" &
PREDICTIONS_PID=$!
echo "✓ Prediction generator started (PID: $PREDICTIONS_PID)"
echo ""

echo "=========================================="
echo "Pipeline is running!"
echo "=========================================="
echo ""
echo "Monitor at:"
echo "  • Grafana: http://localhost:3000"
echo "  • Prometheus: http://localhost:9090"
echo "  • API Metrics: http://localhost:8000/metrics"
echo ""
echo "Running for 5 minutes..."
echo "Press Ctrl+C to stop"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping processes..."
    kill $INGEST_PID 2>/dev/null || true
    kill $PREDICTIONS_PID 2>/dev/null || true
    echo "✓ Done"
}

trap cleanup EXIT INT TERM

# Wait 5 minutes
sleep 300

cleanup
echo ""
echo "Test complete! Check Grafana dashboard for metrics."


#!/bin/bash
# Retrain all models with new features and dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Retraining All Models with New Features"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if MLflow is running
if ! curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "⚠️  MLflow is not running. Starting services..."
    cd docker
    docker compose up -d mlflow
    cd ..
    echo "Waiting for MLflow to start..."
    sleep 5
fi

# Features file to use
FEATURES_FILE="data/processed/features_replay.parquet"

# Check if features file exists
if [ ! -f "$FEATURES_FILE" ]; then
    echo "❌ Features file not found: $FEATURES_FILE"
    exit 1
fi

echo "Using features file: $FEATURES_FILE"
echo ""

# Verify features file has labels
python -c "
import pandas as pd
df = pd.read_parquet('$FEATURES_FILE')
if 'volatility_spike' not in df.columns:
    print('❌ Features file missing volatility_spike labels')
    print('Run: python scripts/add_labels.py --features $FEATURES_FILE')
    exit(1)
print(f'✓ Features file has {len(df)} rows with {df[\"volatility_spike\"].mean():.2%} spike rate')
"

echo ""
echo "Training all models..."
echo ""

# Train all models
python models/train.py \
    --features "$FEATURES_FILE" \
    --models baseline logistic xgboost random_forest \
    --mlflow-uri http://localhost:5001

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "View results in MLflow: http://localhost:5001"
echo ""
echo "Models saved to:"
echo "  - models/artifacts/baseline/"
echo "  - models/artifacts/logistic_regression/"
echo "  - models/artifacts/xgboost/"
echo "  - models/artifacts/random_forest/"
echo ""


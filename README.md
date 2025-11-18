# Crypto Pipeline (cp)

This is a cryptocurrency data pipeline with Kafka streaming, MLflow tracking, and training modules.

## Structure
- `crypto-trainer/`: Training logic
- `crypto-producer/`: Kafka producer
- `crypto-consumer/`: Kafka consumer
- `crypto-stream/`: Streaming logic
- `mlflow/`: MLflow server Docker setup

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "midprice": 100000.0,
        "spread": 1.2,
        "trade_intensity": 15,
        "volatility_30s": 0.0003
      }
    ]
  }'

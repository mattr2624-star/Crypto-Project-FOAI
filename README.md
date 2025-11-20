\# Real-Time Crypto Volatility Service



Thin-slice real-time AI service that predicts short-horizon volatility spikes on BTC/USD, with Kafka + FastAPI + MLflow + Prometheus + Grafana + Evidently.



---



\## Quickstart (≤10-line setup)



1\. Clone the repo and `cd` into it.

2\. Copy `.env.example` → `.env` and adjust if needed.

3\. Start services:  

&nbsp;  ```bash

&nbsp;  docker compose up -d --build





🚀 Real-Time Crypto Volatility Detection — Thin Slice Prototype (Week 4)

A real-time AI service that detects short-horizon volatility spikes in Bitcoin using streaming market data, a Gradient Boosting classifier, and MLOps tooling (FastAPI, Kafka, MLflow, Grafana, Prometheus, Evidently).

Status: Working Thin Slice (Replay Mode) — /predict live, services deployed via Docker Compose.

📌 Architecture Overview
┌────────────┐      ┌─────────────┐      ┌─────────┐      ┌──────────────┐
│ Coinbase   │◄────►│ Kafka       │◄────►│ FastAPI │◄────►│ Prometheus    │
│ (Live/repl)│      │ (Streaming) │      │ Model   │      │ Grafana       │
└────────────┘      └─────────────┘      └─────────┘      └──────────────┘
│
▼
MLflow Registry



→ In Week 4, we run in replay mode using a local pkl model instead of the Registry.

🛠 Run the System (One Command)
docker compose up -d



⏱ Starts:

Kafka + Zookeeper

Model Server (FastAPI)

MLflow Tracking Server

Prometheus + Grafana

Drift Monitor container (future use)

🧪 Test the Model API
curl -X POST http://localhost:8000/predict   
-H "Content-Type: application/json"   
-d '{"rows":\[{"midprice":68000.5,"spread":1.2,"trade\_intensity":14,"volatility\_30s":0.02}]}'



Example Response:

{
"volatility\_score": 0.93,
"model\_name": "crypto-vol-ml",
"model\_version": "latest",
"model\_variant": "ml",
"latency\_ms": 0.0
}

Health Check
curl http://localhost:8000/health

Model Metadata
curl http://localhost:8000/version

📈 Monitoring \& Model Tracking
Tool	URL	Purpose
MLflow	http://localhost:5000
Track experiments + artifacts
Grafana	http://localhost:3000
Model dashboards \& latency (future)
Prometheus	http://localhost:9090
Metrics scraped from API
FastAPI Docs	http://localhost:8000/docs
Live schema \& testing

Default Grafana credentials: admin / admin

📦 Model Artifact

The thin slice uses a locally mounted pickle:

/app/models/gbm\_volatility.pkl



This file contains:

trained classifier

feature names

volatility spike threshold

💡 Later weeks will migrate this to MLflow Model Registry and introduce MODEL\_VARIANT rollback.

📁 Key Repository Files
File	Purpose
docker-compose.yaml	Orchestrates all services
server/model\_loader.py	Loads + aligns features from pickle
server/model\_server.py	FastAPI inference app
team\_charter.md	Roles \& responsibilities
selection\_rationale.md	Why we chose this model
Architecture Diagram.png	System flow
🎯 Next Steps (Week 5 Preview)

Upcoming MLOps enhancements:

GitHub Actions CI (lint + replay test)

Load testing (100 request burst)

Retry \& reconnection logic for Kafka

.env.example + secret sanitization

Basic observability /metrics endpoint

Ready when you are: Say “Begin Week 5 CI Setup”

👥 Team Roles

Defined in team\_charter.md. Each member owns one major subsystem:
📌 Kafka • FastAPI • Models • Monitoring • Deployment

📄 Model Selection Strategy

Gradient Boosting chosen for:

superior PR-AUC on tail volatility events

robust to outliers

interpretable feature importances

Full reasoning in: selection\_rationale.md


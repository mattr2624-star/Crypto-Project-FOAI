Operations Runbook
# 🔧 Runbook — Crypto Volatility Real-Time API

## 🚀 Startup
```bash
docker compose up -d


Verify the API:

curl http://localhost:8000/health

🔁 Restart a Component
docker compose restart <service>


Example:

docker compose restart model-server

🔄 Refresh Drift Report
docker compose exec model-server python /app/scripts/drift_summary.py


Output saved to: reports/drift_report.html

🔁 Rollback / Switch Model Variant

Edit docker-compose.yaml or override via shell env:
Linux/Mac

MODEL_VARIANT=baseline docker compose restart model-server


Windows PowerShell

$Env:MODEL_VARIANT="baseline"
docker compose restart model-server


Confirm variant:

curl http://localhost:8000/version

🩺 Troubleshooting
Symptom	Fix
API won’t start	docker compose restart model-server
Kafka errors	docker compose restart kafka
Grafana blank dashboards	Ensure Prometheus is running, then restart Grafana
Model errors	Check /models/*.pkl exists inside container
Drift summary empty	Regenerate current data & run summary again
📦 Shutdown
docker compose down


---

# 🎥 `docs/demo_script.md` — **8-Minute Demo Script**

```markdown
# 🎬 Demo Checklist — Crypto Volatility AI Service

### 1) Intro (20s)
- “Welcome — we’re demonstrating a real-time crypto volatility scoring system.”
- “It streams data, scores risk, tracks models, and monitors drift + latency.”

---

### 2) One-Command Startup (40s)
Run:
```bash
docker compose up -d


Show containers running:

docker compose ps

3) Working API Thin Slice (1 min)

Call health:

curl http://localhost:8000/health


Call predict:

curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"rows":[{"ret_mean":0.01,"ret_std":0.02,"n":30}]}'

4) Live Metrics Monitoring (1.5 min)

Open Grafana: http://localhost:3000

Show p50 / p95 latency

Show request counter increasing

Show Kafka lag & exporter metrics

5) Failure & Recovery Test (1.5 min)

Simulate failure:

docker compose stop kafka


Try predict call → expected fail.

Bring Kafka back:

docker compose start kafka
docker compose restart model-server


Call predict again → success.

6) Safe Rollback to Baseline (1.5 min)
$Env:MODEL_VARIANT="baseline"  # Windows
docker compose restart model-server


Confirm:

curl http://localhost:8000/version


Call /predict again → note new model_name.

7) Drift & Data Quality (1 min)

Regenerate + summarize drift:

docker compose exec model-server python /app/scripts/drift_summary.py


Open: reports/drift_report.html

Describe drift in feature distributions

8) Closing (30s)

“We achieved latency <800ms p95 and safe rollback.”

“Alerts + drift help maintain reliability.”

“Final repo includes SLOs, CI pipeline, dashboards, and runbook.”


---

# 🏁 Short Public README (≤10 lines)

```markdown
# ⚡ Crypto Volatility AI Service

Predicts short-horizon crypto volatility in real time using FastAPI, Kafka, MLflow, Prometheus, and Evidently.

### 🚀 Quick Start
```bash
docker compose up -d
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"rows":[{"ret_mean":0.01,"ret_std":0.02,"n":30}]}'

📊 Monitoring

Grafana: http://localhost:3000
Prometheus: http://localhost:9090


---

### 🎉 All Deliverables Complete!
You now have everything required for:
✔ Week 4  
✔ Week 5  
✔ Week 6  
✔ Week 7 demo & submission

Would you like a **tagged release template** (GitHub versioning + release notes)?  
Reply **`release`** if yes. 🚀
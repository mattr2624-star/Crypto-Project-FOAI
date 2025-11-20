Project: Crypto Volatility Prediction API
Version: v1.0 (Real–Time Architecture)

1) 🎯 Service Scope
Component	Description
FastAPI Model Server	Scores short-horizon crypto volatility
POST /predict	Predicts volatility probability using the selected model
**`MODEL_VARIANT={ml	baseline
Monitoring & Drift	Metrics via Prometheus; drift via Evidently
2) 🟢 Availability SLO
Target	Description	SLI Source
≥ 99.0% availability	API must respond without crashes or server error (5xx)	Prometheus: up, http_server_requests_total
Failure Budget

Allowed downtime per month at 99.0%: ≤ 7 hrs 18 min

Alert Trigger

🔴 Critical: up == 0 for 1 minute
🟠 Warning: rate(http_requests_total{code=~"5.."}[5m]) > 0.02

3) ⚡ Latency SLO
Metric	Objective	Alert Threshold
p95 ≤ 800 ms	End-to-end scoring	Trigger alert if p95 > 800ms for 10 min
p50 ≤ 100 ms	Expected median latency	Trigger if p50 > 200ms for 10 min
Measurement

Computed via Prometheus histogram http_request_duration_seconds_bucket
Grafana panel: “Model Latency (p50/p95)”

4) ❌ Error Rate SLO
Metric	Objective	Alert Trigger
< 1% 5xx errors	Reliability target	If rate(5xx) > 1% for 15 min
< 2% model failures	Prediction failures (bad_input, model not loaded)	Any spike above 2%
5) 📉 Drift Detection SLO
SLO	Description	Trigger
Drift must be checked every 24h	Scheduled Evidently comparison (reference.csv vs current.csv)	Missed scheduled report
Statistical drift < 0.3 (mean p-value across features)	If drift score > 0.3, rollback model	Action required

👉 Drift failure response: switch to MODEL_VARIANT=baseline and notify via Slack/email (future roadmap).

6) 🔄 Rollback SLO
Requirement	Measurement
Rollback must take < 5 minutes	Toggle MODEL_VARIANT + container restart
No more than 5% prediction degradation after rollback	Compare PR-AUC from MLflow

💡 Rollback Command

export MODEL_VARIANT=baseline
docker compose restart model-server

7) 📊 SLI Measurement (How We Calculate)
Latency SLI Example
SLI = (count(latency < 800ms) / total_requests) * 100

Availability SLI Example
SLI = 1 - (5xx responses / total requests)

Drift SLI Example
SLI = (1 - drift_score)

📌 Summary Table
Category	SLO	SLI Source
Availability	≥ 99.0% uptime	Prometheus up
Latency	p95 ≤ 800ms, p50 ≤ 100ms	Prometheus histogram
Error Rate	< 1% 5xx	Prometheus metrics
Drift	Score < 0.3	Evidently HTML + metrics
Rollback	< 5 min to baseline	Manual toggle
🏁 Status (Current Observed)
Metric	Current	Status
p50	~16ms	🟢
p95	~32ms	🟢
Availability	100%	🟢
Drift	TBD (report scheduled)	🟡
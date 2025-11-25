# Demo Checklist

## 8-Minute Demo Script for Real-Time Crypto AI Service

**Duration:** 8 minutes  
**Format:** Screen recording with narration

---

## Pre-Demo Setup (Before Recording)

- [ ] Ensure Docker Desktop is running
- [ ] All services stopped (`docker compose down` in docker folder)
- [ ] Terminal windows ready
- [ ] Browser tabs ready for:
  - http://localhost:8000/docs (API Swagger)
  - http://localhost:5001 (MLflow)
  - http://localhost:3000 (Grafana)
  - http://localhost:9090 (Prometheus)

---

## Demo Script

### Part 1: System Startup (1.5 minutes)

**Narration:** "Let me show you how to start the entire system with a single command."

```powershell
# Navigate to docker folder
cd C:\cp\docker

# Start all services
docker compose up -d

# Show services starting
docker compose ps
```

**Show:**
- [ ] One-command startup
- [ ] All 6 services starting (Kafka, Zookeeper, MLflow, API, Prometheus, Grafana)
- [ ] Health status becoming "healthy"

**Key Points:**
- "Notice all services start automatically with dependencies"
- "The API waits for Kafka to be healthy before starting"

---

### Part 2: API Demonstration (2 minutes)

**Narration:** "Now let's test the prediction API."

```powershell
# Test health endpoint
curl http://localhost:8000/health

# Test version endpoint  
curl http://localhost:8000/version

# Make a prediction (Assignment API format)
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
```

**Show:**
- [ ] Health check returns healthy status
- [ ] Version shows model info and variant
- [ ] Prediction returns scores in assignment format
- [ ] Show Swagger UI at http://localhost:8000/docs

**Key Points:**
- "The API follows the assignment contract exactly"
- "Notice the model_variant field - this enables rollback"

---

### Part 3: Monitoring Dashboard (1.5 minutes)

**Narration:** "Let's look at our monitoring setup."

**Show:**
- [ ] Grafana dashboard at http://localhost:3000 (admin/admin123)
- [ ] Request rate graph
- [ ] Latency graph (p50/p95/p99)
- [ ] Error rate graph
- [ ] Prometheus at http://localhost:9090

**Key Points:**
- "We track p50, p95, and p99 latency"
- "Our SLO target is p95 under 800ms - we're achieving 91ms"
- "Error rate is monitored for SLO compliance"

---

### Part 4: Failure Recovery (1.5 minutes)

**Narration:** "Let me demonstrate failure recovery."

```powershell
# Stop the API to simulate failure
docker compose stop api

# Show health check fails
curl http://localhost:8000/health  # Should fail

# Restart the API
docker compose start api

# Wait and verify recovery
Start-Sleep -Seconds 10
curl http://localhost:8000/health  # Should succeed
```

**Show:**
- [ ] API becomes unavailable
- [ ] Restart command
- [ ] System recovers automatically
- [ ] Health check passes again

**Key Points:**
- "The system can recover from failures gracefully"
- "Docker Compose handles service dependencies"

---

### Part 5: Model Rollback (1.5 minutes)

**Narration:** "Now let me show the model rollback feature."

```powershell
# Check current model variant
curl http://localhost:8000/version

# Rollback to baseline model
$env:MODEL_VARIANT="baseline"
docker compose up -d api

# Verify rollback
Start-Sleep -Seconds 5
curl http://localhost:8000/version  # Should show baseline

# Make prediction with baseline
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'

# Switch back to ML model
$env:MODEL_VARIANT="ml"
docker compose up -d api
```

**Show:**
- [ ] Current model is ML variant
- [ ] Switch to baseline with environment variable
- [ ] Version endpoint confirms baseline
- [ ] Predictions still work with baseline
- [ ] Switch back to ML model

**Key Points:**
- "MODEL_VARIANT toggle allows instant rollback"
- "No code changes needed - just environment variable"
- "Critical for production safety"

---

## Post-Demo Commands

```powershell
# Show final status
docker compose ps

# Show load test results
python scripts/load_test.py --requests 100

# Clean up (optional)
docker compose down
```

---

## Key Metrics to Mention

| Metric | Value | Notes |
|--------|-------|-------|
| p95 Latency | 91.17ms | 88.6% better than 800ms target |
| Success Rate | 100% | From load test |
| Throughput | 121.92 req/s | Handles burst traffic |
| PR-AUC (ML) | 0.9859 | 9.5x better than baseline |
| PR-AUC (Baseline) | 0.1039 | Available for rollback |

---

## Troubleshooting During Demo

### If API doesn't start:
```powershell
docker compose logs api
docker compose restart api
```

### If Kafka is unhealthy:
```powershell
docker compose restart kafka
Start-Sleep -Seconds 20
docker compose restart api
```

### If predictions fail:
```powershell
# Check model is loaded
curl http://localhost:8000/health
# Restart API
docker compose restart api
```

---

## Recording Tips

1. **Resolution:** 1920x1080
2. **Font size:** Large enough to read
3. **Speak clearly:** Explain what each command does
4. **Pause:** Let viewers see the output
5. **Highlight:** Point out key metrics and features

---

**Demo Video:** [Insert YouTube/Loom link here after recording]

**Last Updated:** November 25, 2025


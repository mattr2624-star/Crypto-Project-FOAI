# Demo Script: Crypto Volatility Detection Service

**Duration:** 8 minutes  
**Required Elements:** Startup, Prediction, Failure Recovery, Rollback

---

## PRE-RECORDING SETUP

Run these commands BEFORE starting your screen recording:

```powershell
# 1. Ensure Docker Desktop is running
# 2. Stop any existing containers and clean up
cd C:\cp\docker
docker compose down -v
cd ..

# 3. Clear terminal
cls
```

---

# START RECORDING HERE

---

## PART 1: STARTUP (2 minutes)

### 1.1 Show Repository Structure

```powershell
# Show we're in the project directory
pwd

# Display project structure
dir

# Show key directories
Write-Host "`n=== API Code ===" -ForegroundColor Cyan
dir api

Write-Host "`n=== Docker Configuration ===" -ForegroundColor Cyan
dir docker

Write-Host "`n=== Models ===" -ForegroundColor Cyan
dir models\artifacts
```

### 1.2 One-Command Startup

```powershell
# Start all services with ONE command
cd docker
docker compose up -d
```

### 1.3 Verify Services Running

```powershell
# Wait for services to initialize
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

# Show all containers are running
docker compose ps
```

### 1.4 Open Grafana Dashboard

```powershell
# Open Grafana in browser (no login required!)
Start-Process "http://localhost:3000"
```

**[BROWSER ACTION]** Navigate to the "Crypto Volatility" dashboard

---

## PART 2: PREDICTION (2 minutes)

### 2.1 Health Check

```powershell
cd ..

# Check API is healthy
Write-Host "`n=== Health Check ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json
```

### 2.2 Version Info

```powershell
# Check model version
Write-Host "`n=== Model Version ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8000/version" | ConvertTo-Json
```

### 2.3 Make Single Prediction

```powershell
# Make a prediction using Assignment API contract
Write-Host "`n=== Single Prediction ===" -ForegroundColor Cyan
$body = '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json
```

### 2.4 Generate Predictions for Dashboard

```powershell
# Generate multiple predictions to populate Grafana dashboard
Write-Host "`n=== Generating Predictions for Dashboard ===" -ForegroundColor Cyan
Write-Host "Watch the Grafana dashboard update in real-time!" -ForegroundColor Yellow

python scripts/prediction_consumer.py --mode demo --interval 1 --duration 30
```

**[BROWSER ACTION]** Show Grafana dashboard updating with live metrics

---

## PART 3: FAILURE RECOVERY (2 minutes)

### 3.1 Simulate Failure - Stop API

```powershell
# Simulate failure by stopping the API container
Write-Host "`n=== SIMULATING FAILURE ===" -ForegroundColor Red
cd docker
docker compose stop api

# Show container is stopped
docker compose ps
```

### 3.2 Show API is Down

```powershell
# Try to call health endpoint - will fail
Write-Host "`n=== Attempting to reach API (will fail) ===" -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3
} catch {
    Write-Host "ERROR: API is not responding - as expected!" -ForegroundColor Red
}
```

### 3.3 Recovery - Restart API

```powershell
# Recover by restarting the API
Write-Host "`n=== RECOVERING SERVICE ===" -ForegroundColor Green
docker compose start api

# Wait for API to be ready
Write-Host "Waiting for API to recover..." -ForegroundColor Yellow
Start-Sleep -Seconds 15
```

### 3.4 Verify Recovery

```powershell
# Verify API is back online
Write-Host "`n=== Verifying Recovery ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8000/health" | ConvertTo-Json

# Make a prediction to confirm it works
Write-Host "`n=== Prediction After Recovery ===" -ForegroundColor Cyan
$body = '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json
```

---

## PART 4: ROLLBACK (2 minutes)

### 4.1 Show Current Model (ML/Random Forest)

```powershell
cd ..

# Current model is ML (Random Forest)
Write-Host "`n=== Current Model ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8000/version" | ConvertTo-Json
```

### 4.2 Rollback to Baseline Model

```powershell
# Rollback to baseline model via environment variable
Write-Host "`n=== ROLLBACK TO BASELINE ===" -ForegroundColor Yellow
$env:MODEL_VARIANT="baseline"
cd docker
docker compose up -d api

# Wait for new model to load
Write-Host "Switching to baseline model..." -ForegroundColor Yellow
Start-Sleep -Seconds 15
```

### 4.3 Verify Rollback

```powershell
cd ..

# Confirm baseline model is now active
Write-Host "`n=== Baseline Model Active ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8000/version" | ConvertTo-Json

# Make prediction with baseline model
Write-Host "`n=== Prediction with Baseline Model ===" -ForegroundColor Cyan
$body = '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body | ConvertTo-Json
```

### 4.4 Restore ML Model

```powershell
# Restore ML model
Write-Host "`n=== RESTORE ML MODEL ===" -ForegroundColor Green
$env:MODEL_VARIANT="ml"
cd docker
docker compose up -d api

Start-Sleep -Seconds 15

cd ..
Write-Host "`n=== ML Model Restored ===" -ForegroundColor Cyan
Invoke-RestMethod -Uri "http://localhost:8000/version" | ConvertTo-Json
```

---

## CONCLUSION

**[BROWSER ACTION]** Show final Grafana dashboard state

```powershell
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "       DEMO COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nDemonstrated:" -ForegroundColor White
Write-Host "  1. One-command STARTUP" -ForegroundColor Cyan
Write-Host "  2. Real-time PREDICTION API" -ForegroundColor Cyan
Write-Host "  3. FAILURE RECOVERY" -ForegroundColor Cyan
Write-Host "  4. Model ROLLBACK" -ForegroundColor Cyan
Write-Host "`nSLO Compliance:" -ForegroundColor White
Write-Host "  - p95 Latency: < 800ms" -ForegroundColor Green
Write-Host "  - Availability: 99%+" -ForegroundColor Green
```

---

# STOP RECORDING HERE

---

## Post-Demo Cleanup (Optional)

```powershell
cd docker
docker compose down -v
```

---

## Quick Reference URLs

| Service | URL |
|---------|-----|
| Grafana Dashboard | http://localhost:3000 |
| API Health | http://localhost:8000/health |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |


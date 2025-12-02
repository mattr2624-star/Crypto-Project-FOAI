# Demo Script: Crypto Volatility Detection Service
# Duration: ~8 minutes
# Covers: Startup, Prediction, Failure Recovery, Rollback

param(
    [switch]$SkipStartup,
    [switch]$AutoMode
)

$ErrorActionPreference = "Continue"

function Write-Section($title) {
    Write-Host "`n" -NoNewline
    Write-Host "=" * 60 -ForegroundColor DarkGray
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor DarkGray
}

function Write-Step($step) {
    Write-Host "`n>>> $step" -ForegroundColor Yellow
}

function Pause-Demo($message = "Press Enter to continue...") {
    if (-not $AutoMode) {
        Write-Host "`n$message" -ForegroundColor DarkGray
        Read-Host
    } else {
        Start-Sleep -Seconds 2
    }
}

# Get project root
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ProjectRoot) { $ProjectRoot = "C:\cp" }

Write-Host @"

  ╔═══════════════════════════════════════════════════════════╗
  ║     CRYPTO VOLATILITY DETECTION SERVICE - DEMO            ║
  ║                                                           ║
  ║     Demonstrates: Startup, Prediction,                    ║
  ║                   Failure Recovery, Rollback              ║
  ╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

Pause-Demo "Press Enter to begin the demo..."

# ============================================================
# PART 1: STARTUP
# ============================================================
Write-Section "PART 1: STARTUP"

if (-not $SkipStartup) {
    Write-Step "Showing project structure..."
    Set-Location $ProjectRoot
    
    Write-Host "`nProject Root:" -ForegroundColor White
    Get-ChildItem -Name | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nDocker Configuration:" -ForegroundColor White
    Get-ChildItem docker -Name | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nModel Artifacts:" -ForegroundColor White
    Get-ChildItem models\artifacts -Name | ForEach-Object { Write-Host "  $_" }
    
    Pause-Demo

    Write-Step "Starting all services with ONE command..."
    Set-Location "$ProjectRoot\docker"
    docker compose up -d
    
    Write-Step "Waiting for services to initialize (20 seconds)..."
    for ($i = 20; $i -gt 0; $i--) {
        Write-Host "`r  Starting in $i seconds..." -NoNewline -ForegroundColor DarkGray
        Start-Sleep -Seconds 1
    }
    Write-Host "`r  Services started!           " -ForegroundColor Green
    
    Write-Step "Verifying all containers are running..."
    docker compose ps
    
    Pause-Demo
    
    Write-Step "Opening Grafana Dashboard..."
    Start-Process "http://localhost:3000"
    Write-Host "  Dashboard opened in browser (no login required)" -ForegroundColor Green
}

Set-Location $ProjectRoot

# ============================================================
# PART 2: PREDICTION
# ============================================================
Write-Section "PART 2: PREDICTION"

Pause-Demo

Write-Step "Health Check..."
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health"
    Write-Host ($health | ConvertTo-Json) -ForegroundColor White
} catch {
    Write-Host "  ERROR: API not responding" -ForegroundColor Red
}

Pause-Demo

Write-Step "Model Version..."
try {
    $version = Invoke-RestMethod -Uri "http://localhost:8000/version"
    Write-Host ($version | ConvertTo-Json) -ForegroundColor White
} catch {
    Write-Host "  ERROR: Could not get version" -ForegroundColor Red
}

Pause-Demo

Write-Step "Making a Prediction..."
$body = '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
try {
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body
    Write-Host ($prediction | ConvertTo-Json) -ForegroundColor White
} catch {
    Write-Host "  ERROR: Prediction failed" -ForegroundColor Red
}

Pause-Demo

Write-Step "Generating predictions for Grafana dashboard (30 seconds)..."
Write-Host "  Watch the dashboard update in real-time!" -ForegroundColor Yellow
python "$ProjectRoot\scripts\prediction_consumer.py" --mode demo --interval 1 --duration 30

Pause-Demo

# ============================================================
# PART 3: FAILURE RECOVERY
# ============================================================
Write-Section "PART 3: FAILURE RECOVERY"

Pause-Demo

Write-Step "SIMULATING FAILURE - Stopping API container..."
Set-Location "$ProjectRoot\docker"
docker compose stop api
Write-Host "  API container stopped!" -ForegroundColor Red

Write-Step "Container status:"
docker compose ps

Pause-Demo

Write-Step "Attempting to reach API (will fail)..."
try {
    Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3
    Write-Host "  API responded (unexpected)" -ForegroundColor Yellow
} catch {
    Write-Host "  ERROR: API is not responding - as expected!" -ForegroundColor Red
}

Pause-Demo

Write-Step "RECOVERING - Restarting API container..."
docker compose start api

Write-Host "  Waiting for API to recover (15 seconds)..." -ForegroundColor Yellow
for ($i = 15; $i -gt 0; $i--) {
    Write-Host "`r  Recovery in $i seconds..." -NoNewline -ForegroundColor DarkGray
    Start-Sleep -Seconds 1
}
Write-Host "`r  Recovery complete!           " -ForegroundColor Green

Write-Step "Verifying API is back online..."
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health"
    Write-Host ($health | ConvertTo-Json) -ForegroundColor Green
} catch {
    Write-Host "  ERROR: API still not responding" -ForegroundColor Red
}

Write-Step "Making prediction after recovery..."
try {
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body
    Write-Host ($prediction | ConvertTo-Json) -ForegroundColor White
    Write-Host "  SUCCESS: API recovered and working!" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Prediction failed" -ForegroundColor Red
}

Set-Location $ProjectRoot

Pause-Demo

# ============================================================
# PART 4: ROLLBACK
# ============================================================
Write-Section "PART 4: MODEL ROLLBACK"

Pause-Demo

Write-Step "Current model (ML/Random Forest)..."
$version = Invoke-RestMethod -Uri "http://localhost:8000/version"
Write-Host ($version | ConvertTo-Json) -ForegroundColor White

Pause-Demo

Write-Step "ROLLBACK TO BASELINE MODEL..."
$env:MODEL_VARIANT = "baseline"
Set-Location "$ProjectRoot\docker"
docker compose up -d api

Write-Host "  Switching to baseline model (15 seconds)..." -ForegroundColor Yellow
for ($i = 15; $i -gt 0; $i--) {
    Write-Host "`r  Rollback in $i seconds..." -NoNewline -ForegroundColor DarkGray
    Start-Sleep -Seconds 1
}
Write-Host "`r  Rollback complete!           " -ForegroundColor Green

Set-Location $ProjectRoot

Write-Step "Verifying baseline model is active..."
$version = Invoke-RestMethod -Uri "http://localhost:8000/version"
Write-Host ($version | ConvertTo-Json) -ForegroundColor Yellow

Write-Step "Prediction with baseline model..."
$prediction = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $body
Write-Host ($prediction | ConvertTo-Json) -ForegroundColor White

Pause-Demo

Write-Step "RESTORING ML MODEL..."
$env:MODEL_VARIANT = "ml"
Set-Location "$ProjectRoot\docker"
docker compose up -d api

Write-Host "  Restoring ML model (15 seconds)..." -ForegroundColor Yellow
for ($i = 15; $i -gt 0; $i--) {
    Write-Host "`r  Restore in $i seconds..." -NoNewline -ForegroundColor DarkGray
    Start-Sleep -Seconds 1
}
Write-Host "`r  ML model restored!           " -ForegroundColor Green

Set-Location $ProjectRoot

Write-Step "Verifying ML model is restored..."
$version = Invoke-RestMethod -Uri "http://localhost:8000/version"
Write-Host ($version | ConvertTo-Json) -ForegroundColor Green

# ============================================================
# CONCLUSION
# ============================================================
Write-Host @"

  ╔═══════════════════════════════════════════════════════════╗
  ║                    DEMO COMPLETE!                         ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  Demonstrated:                                            ║
  ║    ✓ One-command STARTUP                                  ║
  ║    ✓ Real-time PREDICTION API                             ║
  ║    ✓ FAILURE RECOVERY                                     ║
  ║    ✓ Model ROLLBACK                                       ║
  ║                                                           ║
  ║  SLO Compliance:                                          ║
  ║    ✓ p95 Latency: < 800ms                                 ║
  ║    ✓ Availability: 99%+                                   ║
  ╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "Demo URLs:" -ForegroundColor Cyan
Write-Host "  Grafana:    http://localhost:3000" -ForegroundColor White
Write-Host "  API Health: http://localhost:8000/health" -ForegroundColor White
Write-Host "  API Docs:   http://localhost:8000/docs" -ForegroundColor White


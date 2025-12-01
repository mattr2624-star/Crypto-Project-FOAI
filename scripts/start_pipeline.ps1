# ============================================================================
# Crypto Volatility Pipeline - Quick Start Script (Windows)
# ============================================================================
# This script starts all services and begins generating predictions
# to populate the Grafana dashboard with live metrics.
#
# Usage: .\scripts\start_pipeline.ps1 [-DemoDuration 60]
# ============================================================================

param(
    [int]$DemoDuration = 60  # Default 60 seconds of demo predictions
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Step { param($msg) Write-Host "`n[Step] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

Write-Host @"

============================================================
  Crypto Volatility Pipeline - Quick Start (Windows)
============================================================

"@ -ForegroundColor Cyan

# Get project directory
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Step 1: Check Docker
Write-Step "Checking Docker..."

try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker not found" }
    Write-Success "Docker installed: $dockerVersion"
} catch {
    Write-Err "Docker not found. Please install Docker Desktop."
    exit 1
}

try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker not running" }
    Write-Success "Docker daemon is running"
} catch {
    Write-Err "Docker daemon not running. Please start Docker Desktop."
    exit 1
}

# Step 2: Start services
Write-Step "Starting Docker services..."

Push-Location "$ProjectDir\docker"
try {
    # Stop existing containers
    docker compose down --remove-orphans 2>&1 | Out-Null
    
    # Start fresh
    docker compose up -d --build 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Failed to start services" }
    
    Write-Success "Services started"
} catch {
    Write-Err "Failed to start Docker services: $_"
    Pop-Location
    exit 1
}
Pop-Location

# Step 3: Wait for API
Write-Step "Waiting for API to be ready..."

$maxRetries = 30
$retryCount = 0
$apiReady = $false

while ($retryCount -lt $maxRetries -and -not $apiReady) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.status -eq "healthy") {
            $apiReady = $true
            Write-Success "API is healthy"
        }
    } catch {
        $retryCount++
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
}

if (-not $apiReady) {
    Write-Host ""
    Write-Warn "API health check timed out, but continuing..."
}

# Step 4: Generate predictions
Write-Step "Generating predictions for dashboard..."

$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
}

$consumerJob = $null
if ($pythonCmd) {
    # Check for requests module
    try {
        & $pythonCmd -c "import requests" 2>&1 | Out-Null
    } catch {
        Write-Host "Installing requests module..." -ForegroundColor Yellow
        & $pythonCmd -m pip install requests -q 2>&1 | Out-Null
    }
    
    # Run prediction consumer in background
    Write-Host "Running predictions for $DemoDuration seconds..." -ForegroundColor Green
    $consumerJob = Start-Job -ScriptBlock {
        param($python, $projectDir, $duration)
        Set-Location $projectDir
        & $python scripts/prediction_consumer.py --mode demo --interval 2 --duration $duration
    } -ArgumentList $pythonCmd, $ProjectDir, $DemoDuration
    
    Start-Sleep -Seconds 3
} else {
    Write-Warn "Python not found, skipping prediction generation"
}

# Open Grafana dashboard
Write-Step "Opening Grafana dashboard..."
$dashboardUrl = "http://localhost:3000/d/crypto-volatility-api"
Start-Process $dashboardUrl

# Summary
Write-Host @"

============================================================
  Pipeline Started Successfully!
============================================================

  Services running:
    * Grafana Dashboard:  http://localhost:3000 (no login required)
    * API Documentation:  http://localhost:8000/docs
    * Prometheus:         http://localhost:9090
    * MLflow:             http://localhost:5001

  Commands:
    * Stop services:      cd docker; docker compose down
    * View logs:          cd docker; docker compose logs -f
    * More predictions:   python scripts/prediction_consumer.py --mode demo

"@ -ForegroundColor Green

# Wait for consumer job if running
if ($consumerJob) {
    Write-Host "Waiting for demo predictions to complete..." -ForegroundColor Yellow
    Wait-Job $consumerJob -Timeout ($DemoDuration + 30) | Out-Null
    Receive-Job $consumerJob
    Remove-Job $consumerJob -Force
    Write-Success "Demo predictions complete. Dashboard should show metrics."
}


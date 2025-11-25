#Requires -Version 5.1
<#
.SYNOPSIS
    End-to-end setup script for Real-Time Crypto Volatility Detection Service on Windows.

.DESCRIPTION
    This script will:
    1. Check prerequisites (Docker, Git)
    2. Clone or update the repository
    3. Build and start all services
    4. Verify the API is working
    5. Open Grafana dashboard

.PARAMETER SkipClone
    Skip cloning if you already have the repo locally.

.PARAMETER ProjectPath
    Path where to clone/run the project. Default: $env:USERPROFILE\crypto-volatility

.EXAMPLE
    .\setup-windows.ps1
    
.EXAMPLE
    .\setup-windows.ps1 -SkipClone -ProjectPath "C:\MyProjects\crypto"
#>

param(
    [switch]$SkipClone,
    [string]$ProjectPath = "$env:USERPROFILE\crypto-volatility"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Banner
Write-Host @"

  ____                  _        __     __    _       _   _ _ _ _         
 / ___|_ __ _   _ _ __ | |_ ___  \ \   / /__ | | __ _| |_(_) (_) |_ _   _ 
| |   | '__| | | | '_ \| __/ _ \  \ \ / / _ \| |/ _` | __| | | | __| | | |
| |___| |  | |_| | |_) | || (_) |  \ V / (_) | | (_| | |_| | | | |_| |_| |
 \____|_|   \__, | .__/ \__\___/    \_/ \___/|_|\__,_|\__|_|_|_|\__|\__, |
            |___/|_|                                                |___/ 
            
        Real-Time AI Service - Windows Setup Script
        Repository: https://github.com/mattr2624-star/Crypto-Project-FOAI

"@ -ForegroundColor Magenta

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
Write-Step "Checking Prerequisites"

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker installed: $dockerVersion"
    } else {
        throw "Docker not found"
    }
} catch {
    Write-Error "Docker is not installed or not in PATH"
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}

# Check Docker is running
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker daemon not running"
    }
    Write-Success "Docker daemon is running"
} catch {
    Write-Error "Docker daemon is not running"
    Write-Host "Please start Docker Desktop and try again" -ForegroundColor Yellow
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker compose version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker Compose installed: $composeVersion"
    } else {
        throw "Docker Compose not found"
    }
} catch {
    Write-Error "Docker Compose is not available"
    Write-Host "Please ensure Docker Desktop is up to date" -ForegroundColor Yellow
    exit 1
}

# Check Git (only if not skipping clone)
if (-not $SkipClone) {
    try {
        $gitVersion = git --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Git installed: $gitVersion"
        } else {
            throw "Git not found"
        }
    } catch {
        Write-Error "Git is not installed or not in PATH"
        Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
        exit 1
    }
}

# ============================================================================
# STEP 2: Clone or Update Repository
# ============================================================================
Write-Step "Setting Up Repository"

$repoUrl = "https://github.com/mattr2624-star/Crypto-Project-FOAI.git"

if (-not $SkipClone) {
    if (Test-Path $ProjectPath) {
        Write-Warning "Project directory exists: $ProjectPath"
        $response = Read-Host "Delete and re-clone? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Write-Host "Removing existing directory..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $ProjectPath
        } else {
            Write-Host "Pulling latest changes..." -ForegroundColor Yellow
            Push-Location $ProjectPath
            git pull origin master 2>&1 | Out-Null
            Pop-Location
        }
    }
    
    if (-not (Test-Path $ProjectPath)) {
        Write-Host "Cloning repository..." -ForegroundColor Yellow
        git clone $repoUrl $ProjectPath 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to clone repository"
            exit 1
        }
    }
    Write-Success "Repository ready at: $ProjectPath"
} else {
    if (-not (Test-Path $ProjectPath)) {
        Write-Error "Project path does not exist: $ProjectPath"
        exit 1
    }
    Write-Success "Using existing project at: $ProjectPath"
}

# Change to project directory
Set-Location $ProjectPath
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Gray

# ============================================================================
# STEP 3: Stop Any Existing Containers
# ============================================================================
Write-Step "Cleaning Up Existing Containers"

Push-Location "$ProjectPath\docker"
try {
    docker compose down --remove-orphans 2>&1 | Out-Null
    Write-Success "Cleaned up any existing containers"
} catch {
    Write-Warning "No existing containers to clean up"
}
Pop-Location

# ============================================================================
# STEP 4: Build and Start Services
# ============================================================================
Write-Step "Building and Starting Services"

Push-Location "$ProjectPath\docker"

Write-Host "Building Docker images (this may take a few minutes)..." -ForegroundColor Yellow
docker compose build --no-cache 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build Docker images"
    exit 1
}
Write-Success "Docker images built successfully"

Write-Host "Starting services..." -ForegroundColor Yellow
docker compose up -d 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start services"
    exit 1
}
Write-Success "Services started"

Pop-Location

# ============================================================================
# STEP 5: Wait for Services to be Ready
# ============================================================================
Write-Step "Waiting for Services to Initialize"

$maxRetries = 30
$retryCount = 0
$apiReady = $false

Write-Host "Waiting for API to be ready..." -ForegroundColor Yellow

while ($retryCount -lt $maxRetries -and -not $apiReady) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.status -eq "healthy") {
            $apiReady = $true
        }
    } catch {
        $retryCount++
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
}

Write-Host ""

if ($apiReady) {
    Write-Success "API is healthy and ready!"
} else {
    Write-Warning "API health check timed out, but services may still be starting"
    Write-Host "Check logs with: docker compose -f docker/compose.yaml logs api" -ForegroundColor Yellow
}

# ============================================================================
# STEP 6: Verify API Endpoints
# ============================================================================
Write-Step "Verifying API Endpoints"

# Test /health
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Success "/health endpoint: $($health.status)"
} catch {
    Write-Warning "/health endpoint not responding"
}

# Test /version
try {
    $version = Invoke-RestMethod -Uri "http://localhost:8000/version" -TimeoutSec 5
    Write-Success "/version endpoint: Model=$($version.model_variant), Version=$($version.version)"
} catch {
    Write-Warning "/version endpoint not responding"
}

# Test /predict
try {
    $predictBody = '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $predictBody -TimeoutSec 10
    Write-Success "/predict endpoint: Score=$($prediction.scores[0]), Model=$($prediction.model_variant)"
} catch {
    Write-Warning "/predict endpoint not responding: $_"
}

# ============================================================================
# STEP 7: Display Service URLs
# ============================================================================
Write-Step "Service URLs"

Write-Host @"

  Service             URL                              Credentials
  ----------------    -----------------------------    -------------------
  FastAPI (Predict)   http://localhost:8000/predict    -
  FastAPI (Docs)      http://localhost:8000/docs       -
  FastAPI (Health)    http://localhost:8000/health     -
  Prometheus          http://localhost:9090            -
  Grafana             http://localhost:3000            admin / admin123
  MLflow              http://localhost:5001            -

"@ -ForegroundColor White

# ============================================================================
# STEP 8: Quick Test Commands
# ============================================================================
Write-Step "Quick Test Commands"

Write-Host @"

# Health check:
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Make a prediction:
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'

# Check version:
Invoke-RestMethod -Uri "http://localhost:8000/version"

# View logs:
cd $ProjectPath\docker; docker compose logs -f api

# Stop services:
cd $ProjectPath\docker; docker compose down

"@ -ForegroundColor Gray

# ============================================================================
# STEP 9: Open Browser
# ============================================================================
Write-Step "Opening Services in Browser"

$openBrowser = Read-Host "Open Grafana dashboard in browser? (Y/n)"
if ($openBrowser -ne 'n' -and $openBrowser -ne 'N') {
    Start-Process "http://localhost:3000/d/crypto-volatility-api"
    Write-Success "Opened Grafana dashboard"
}

# ============================================================================
# COMPLETE
# ============================================================================
Write-Host @"

========================================================================
                    SETUP COMPLETE!
========================================================================

The Real-Time Crypto Volatility Detection Service is now running.

Next steps:
1. Login to Grafana (admin/admin123) to view dashboards
2. Test the /predict endpoint with the commands above
3. View API documentation at http://localhost:8000/docs

To stop all services:
  cd $ProjectPath\docker
  docker compose down

To view logs:
  cd $ProjectPath\docker
  docker compose logs -f

========================================================================

"@ -ForegroundColor Green


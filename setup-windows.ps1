#Requires -Version 5.1
<#
.SYNOPSIS
    End-to-end setup script for Real-Time Crypto Volatility Detection Service on Windows.

.DESCRIPTION
    This script will:
    1. Check prerequisites (Docker, Git, WSL2)
    2. Clone or update the repository
    3. Build and start all services
    4. Verify the API is working
    5. Open Grafana dashboard

.PARAMETER SkipClone
    Skip cloning if you already have the repo locally.

.PARAMETER ProjectPath
    Path where to clone/run the project. Default: $env:USERPROFILE\crypto-volatility

.PARAMETER NonInteractive
    Run without user prompts (auto-accept defaults).

.EXAMPLE
    # First, allow script execution (run as Administrator):
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    
    # Then run the setup:
    .\setup-windows.ps1
    
.EXAMPLE
    .\setup-windows.ps1 -SkipClone -ProjectPath "C:\MyProjects\crypto"
    
.EXAMPLE
    # Non-interactive mode (for automation):
    .\setup-windows.ps1 -NonInteractive

.NOTES
    PREREQUISITES:
    - Windows 10/11 (64-bit)
    - Docker Desktop installed and running
    - Git installed (for cloning)
    - WSL2 enabled (for Docker)
    - At least 4GB RAM available for Docker
    
    COMMON ISSUES:
    1. "Running scripts is disabled" - Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    2. "Docker daemon not running" - Start Docker Desktop application
    3. "WSL2 not installed" - Run: wsl --install (requires restart)
#>

param(
    [switch]$SkipClone,
    [string]$ProjectPath = "$env:USERPROFILE\crypto-volatility",
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Gray }

# Banner
Write-Host @"

  ____                  _        __     __    _       _   _ _ _ _         
 / ___|_ __ _   _ _ __ | |_ ___  \ \   / /__ | | __ _| |_(_) (_) |_ _   _ 
| |   | '__| | | | '_ \| __/ _ \  \ \ / / _ \| |/ _`` | __| | | | __| | | |
| |___| |  | |_| | |_) | || (_) |  \ V / (_) | | (_| | |_| | | | |_| |_| |
 \____|_|   \__, | .__/ \__\___/    \_/ \___/|_|\__,_|\__|_|_|_|\__|\__, |
            |___/|_|                                                |___/ 
            
        Real-Time AI Service - Windows Setup Script
        Repository: https://github.com/mattr2624-star/Crypto-Project-FOAI

"@ -ForegroundColor Magenta

# ============================================================================
# STEP 0: Check Execution Policy
# ============================================================================
Write-Step "Checking PowerShell Configuration"

$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Err "PowerShell execution policy is 'Restricted'"
    Write-Host @"

To fix this, run PowerShell as Administrator and execute:
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then run this script again.
"@ -ForegroundColor Yellow
    exit 1
}
Write-Success "Execution policy: $policy"

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
Write-Step "Checking Prerequisites"

# Check Windows version
$osInfo = Get-CimInstance -ClassName Win32_OperatingSystem
$osVersion = [version]$osInfo.Version
if ($osVersion.Major -lt 10) {
    Write-Err "Windows 10 or later is required. Current: $($osInfo.Caption)"
    exit 1
}
Write-Success "Windows version: $($osInfo.Caption)"

# Check WSL2
Write-Info "Checking WSL2..."
try {
    $wslStatus = wsl --status 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "WSL not installed"
    }
    Write-Success "WSL2 is available"
} catch {
    Write-Warn "WSL2 may not be installed or enabled"
    Write-Host @"

Docker Desktop requires WSL2. To install:
1. Open PowerShell as Administrator
2. Run: wsl --install
3. Restart your computer
4. Run this script again

"@ -ForegroundColor Yellow
    if (-not $NonInteractive) {
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne 'y' -and $continue -ne 'Y') {
            exit 1
        }
    }
}

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker installed: $dockerVersion"
    } else {
        throw "Docker not found"
    }
} catch {
    Write-Err "Docker is not installed or not in PATH"
    Write-Host @"

Please install Docker Desktop from:
    https://www.docker.com/products/docker-desktop/

After installation:
1. Start Docker Desktop
2. Wait for it to fully start (whale icon in system tray)
3. Run this script again

"@ -ForegroundColor Yellow
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
    Write-Err "Docker daemon is not running"
    Write-Host @"

Please start Docker Desktop:
1. Click the Docker Desktop icon in Start Menu
2. Wait for the whale icon in system tray to stop animating
3. Run this script again

If Docker Desktop won't start, try:
- Restarting your computer
- Checking that WSL2 is enabled
- Running Docker Desktop as Administrator

"@ -ForegroundColor Yellow
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
    Write-Err "Docker Compose is not available"
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
        Write-Err "Git is not installed or not in PATH"
        Write-Host @"

Please install Git from:
    https://git-scm.com/download/win

Use default installation options, then:
1. Close and reopen PowerShell
2. Run this script again

"@ -ForegroundColor Yellow
        exit 1
    }
}

# Check available memory
$memory = Get-CimInstance -ClassName Win32_ComputerSystem
$totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 1)
if ($totalMemoryGB -lt 4) {
    Write-Warn "Low memory detected: ${totalMemoryGB}GB. Docker recommends at least 4GB."
} else {
    Write-Success "System memory: ${totalMemoryGB}GB"
}

# Check port availability
Write-Info "Checking port availability..."
$portsToCheck = @(8000, 3000, 9090, 5001, 9092)
$portsInUse = @()
foreach ($port in $portsToCheck) {
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connection) {
        $portsInUse += $port
    }
}
if ($portsInUse.Count -gt 0) {
    Write-Warn "Ports already in use: $($portsInUse -join ', ')"
    Write-Host @"

These ports are required:
- 8000: FastAPI
- 3000: Grafana
- 9090: Prometheus
- 5001: MLflow
- 9092: Kafka

To find what's using a port:
    Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess

"@ -ForegroundColor Yellow
    if (-not $NonInteractive) {
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne 'y' -and $continue -ne 'Y') {
            exit 1
        }
    }
} else {
    Write-Success "All required ports are available"
}

# ============================================================================
# STEP 2: Clone or Update Repository
# ============================================================================
Write-Step "Setting Up Repository"

$repoUrl = "https://github.com/mattr2624-star/Crypto-Project-FOAI.git"

if (-not $SkipClone) {
    if (Test-Path $ProjectPath) {
        Write-Warn "Project directory exists: $ProjectPath"
        if ($NonInteractive) {
            Write-Info "Non-interactive mode: pulling latest changes"
            Push-Location $ProjectPath
            git pull origin master 2>&1 | Out-Null
            Pop-Location
        } else {
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
    }
    
    if (-not (Test-Path $ProjectPath)) {
        Write-Host "Cloning repository..." -ForegroundColor Yellow
        git clone $repoUrl $ProjectPath 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to clone repository"
            Write-Host @"

Possible causes:
- No internet connection
- GitHub is down
- Firewall blocking git

Try cloning manually:
    git clone $repoUrl

"@ -ForegroundColor Yellow
            exit 1
        }
    }
    Write-Success "Repository ready at: $ProjectPath"
} else {
    if (-not (Test-Path $ProjectPath)) {
        Write-Err "Project path does not exist: $ProjectPath"
        exit 1
    }
    Write-Success "Using existing project at: $ProjectPath"
}

# Change to project directory
Set-Location $ProjectPath
Write-Info "Working directory: $(Get-Location)"

# ============================================================================
# STEP 3: Stop Any Existing Containers
# ============================================================================
Write-Step "Cleaning Up Existing Containers"

Push-Location "$ProjectPath\docker"
try {
    docker compose down --remove-orphans 2>&1 | Out-Null
    Write-Success "Cleaned up any existing containers"
} catch {
    Write-Info "No existing containers to clean up"
}
Pop-Location

# ============================================================================
# STEP 4: Build and Start Services
# ============================================================================
Write-Step "Building and Starting Services"

Push-Location "$ProjectPath\docker"

Write-Host "Building Docker images (this may take 2-5 minutes on first run)..." -ForegroundColor Yellow
docker compose build 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to build Docker images"
    Write-Host @"

Possible causes:
- Insufficient disk space
- Network issues downloading base images
- Docker resource limits

Try:
1. Check Docker Desktop settings -> Resources
2. Increase memory/disk limits
3. Run: docker system prune -a (WARNING: removes all unused images)

"@ -ForegroundColor Yellow
    exit 1
}
Write-Success "Docker images built successfully"

Write-Host "Starting services..." -ForegroundColor Yellow
docker compose up -d 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to start services"
    Write-Host "Check logs with: docker compose logs" -ForegroundColor Yellow
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

Write-Host "Waiting for API to be ready (up to 60 seconds)..." -ForegroundColor Yellow

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
    Write-Warn "API health check timed out"
    Write-Host @"

The API may still be starting. Check with:
    docker compose -f $ProjectPath\docker\compose.yaml logs api

Common issues:
- Model file loading (first startup is slower)
- Container still initializing

"@ -ForegroundColor Yellow
}

# ============================================================================
# STEP 6: Verify API Endpoints
# ============================================================================
Write-Step "Verifying API Endpoints"

# Test /health
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
    Write-Success "/health: $($health.status) (model_loaded: $($health.model_loaded))"
} catch {
    Write-Warn "/health endpoint not responding"
}

# Test /version
try {
    $version = Invoke-RestMethod -Uri "http://localhost:8000/version" -TimeoutSec 5
    Write-Success "/version: model=$($version.model_variant), version=$($version.version)"
} catch {
    Write-Warn "/version endpoint not responding"
}

# Test /predict
try {
    $predictBody = '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
    $prediction = Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body $predictBody -TimeoutSec 10
    Write-Success "/predict: score=$($prediction.scores[0]), model=$($prediction.model_variant)"
} catch {
    Write-Warn "/predict endpoint not responding: $_"
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
`$body = '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body `$body

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

if ($NonInteractive) {
    Start-Process "http://localhost:3000/d/crypto-volatility-api"
    Write-Success "Opened Grafana dashboard"
} else {
    $openBrowser = Read-Host "Open Grafana dashboard in browser? (Y/n)"
    if ($openBrowser -ne 'n' -and $openBrowser -ne 'N') {
        Start-Process "http://localhost:3000/d/crypto-volatility-api"
        Write-Success "Opened Grafana dashboard"
    }
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

TROUBLESHOOTING:
- If Grafana shows "No data": Wait 1-2 minutes for metrics to populate
- If API returns errors: Check logs with 'docker compose logs api'
- If containers won't start: Check Docker Desktop resources/settings

========================================================================

"@ -ForegroundColor Green

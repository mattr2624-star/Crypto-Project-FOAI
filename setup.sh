#!/bin/bash
# ============================================================================
# Real-Time Crypto Volatility Detection Service - Linux/Mac Setup
# ============================================================================
# This script sets up and starts all services for the crypto volatility
# detection pipeline.
#
# Usage: ./setup.sh [--skip-clone] [--project-path PATH]
#
# Prerequisites:
#   - Docker Desktop or Docker Engine installed and running
#   - Git installed (for cloning)
#   - At least 4GB RAM available for Docker
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SKIP_CLONE=false
PROJECT_PATH="${HOME}/crypto-volatility"
REPO_URL="https://github.com/mattr2624-star/Crypto-Project-FOAI.git"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-clone)
            SKIP_CLONE=true
            shift
            ;;
        --project-path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-clone] [--project-path PATH]"
            echo ""
            echo "Options:"
            echo "  --skip-clone     Skip cloning if you already have the repo"
            echo "  --project-path   Path where to clone/run the project (default: ~/crypto-volatility)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Functions
print_step() {
    echo -e "\n${CYAN}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "[INFO] $1"
}

# Banner
echo -e "${MAGENTA}"
cat << 'EOF'
  ____                  _        __     __    _       _   _ _ _ _         
 / ___|_ __ _   _ _ __ | |_ ___  \ \   / /__ | | __ _| |_(_) (_) |_ _   _ 
| |   | '__| | | | '_ \| __/ _ \  \ \ / / _ \| |/ _` | __| | | | __| | | |
| |___| |  | |_| | |_) | || (_) |  \ V / (_) | | (_| | |_| | | | |_| |_| |
 \____|_|   \__, | .__/ \__\___/    \_/ \___/|_|\__,_|\__|_|_|_|\__|\__, |
            |___/|_|                                                |___/ 
            
        Real-Time AI Service - Linux/Mac Setup Script
        Repository: https://github.com/mattr2624-star/Crypto-Project-FOAI

EOF
echo -e "${NC}"

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
print_step "Checking Prerequisites"

# Check OS
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
    print_success "Operating System: macOS"
elif [[ "$OS" == "Linux" ]]; then
    print_success "Operating System: Linux"
else
    print_warn "Unknown OS: $OS - may work but not tested"
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker installed: $DOCKER_VERSION"
else
    print_error "Docker is not installed"
    echo ""
    echo "Please install Docker:"
    echo "  macOS: https://www.docker.com/products/docker-desktop/"
    echo "  Linux: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check Docker is running
if docker info &> /dev/null; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running"
    echo ""
    echo "Please start Docker:"
    echo "  macOS: Open Docker Desktop application"
    echo "  Linux: sudo systemctl start docker"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    print_success "Docker Compose installed: $COMPOSE_VERSION"
else
    print_error "Docker Compose is not available"
    echo "Please ensure Docker Desktop is up to date or install docker-compose-plugin"
    exit 1
fi

# Check Git (only if not skipping clone)
if [[ "$SKIP_CLONE" == false ]]; then
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version)
        print_success "Git installed: $GIT_VERSION"
    else
        print_error "Git is not installed"
        echo ""
        echo "Please install Git:"
        echo "  macOS: brew install git"
        echo "  Linux: sudo apt-get install git"
        exit 1
    fi
fi

# Check available memory
if [[ "$OS" == "Darwin" ]]; then
    TOTAL_MEM=$(sysctl -n hw.memsize)
    TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024 / 1024))
else
    TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024))
fi

if [[ $TOTAL_MEM_GB -lt 4 ]]; then
    print_warn "Low memory detected: ${TOTAL_MEM_GB}GB. Docker recommends at least 4GB."
else
    print_success "System memory: ${TOTAL_MEM_GB}GB"
fi

# Check port availability
print_info "Checking port availability..."
PORTS_TO_CHECK=(8000 3000 9090 5001 9092)
PORTS_IN_USE=()

for PORT in "${PORTS_TO_CHECK[@]}"; do
    if lsof -i :$PORT &> /dev/null || netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
        PORTS_IN_USE+=($PORT)
    fi
done

if [[ ${#PORTS_IN_USE[@]} -gt 0 ]]; then
    print_warn "Ports already in use: ${PORTS_IN_USE[*]}"
    echo ""
    echo "These ports are required:"
    echo "  - 8000: FastAPI"
    echo "  - 3000: Grafana"
    echo "  - 9090: Prometheus"
    echo "  - 5001: MLflow"
    echo "  - 9092: Kafka"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "All required ports are available"
fi

# ============================================================================
# STEP 2: Clone or Update Repository
# ============================================================================
print_step "Setting Up Repository"

if [[ "$SKIP_CLONE" == false ]]; then
    if [[ -d "$PROJECT_PATH" ]]; then
        print_warn "Project directory exists: $PROJECT_PATH"
        read -p "Delete and re-clone? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing directory..."
            rm -rf "$PROJECT_PATH"
        else
            echo "Pulling latest changes..."
            cd "$PROJECT_PATH"
            git pull origin master || true
        fi
    fi
    
    if [[ ! -d "$PROJECT_PATH" ]]; then
        echo "Cloning repository..."
        git clone "$REPO_URL" "$PROJECT_PATH"
        if [[ $? -ne 0 ]]; then
            print_error "Failed to clone repository"
            exit 1
        fi
    fi
    print_success "Repository ready at: $PROJECT_PATH"
else
    if [[ ! -d "$PROJECT_PATH" ]]; then
        print_error "Project path does not exist: $PROJECT_PATH"
        exit 1
    fi
    print_success "Using existing project at: $PROJECT_PATH"
fi

cd "$PROJECT_PATH"
print_info "Working directory: $(pwd)"

# ============================================================================
# STEP 3: Stop Any Existing Containers
# ============================================================================
print_step "Cleaning Up Existing Containers"

cd "$PROJECT_PATH/docker"
docker compose down --remove-orphans 2>/dev/null || true
print_success "Cleaned up any existing containers"

# ============================================================================
# STEP 4: Build and Start Services
# ============================================================================
print_step "Building and Starting Services"

echo "Building Docker images (this may take 2-5 minutes on first run)..."
if ! docker compose build; then
    print_error "Failed to build Docker images"
    echo ""
    echo "Possible causes:"
    echo "  - Insufficient disk space"
    echo "  - Network issues downloading base images"
    echo "  - Docker resource limits"
    echo ""
    echo "Try:"
    echo "  1. Check Docker Desktop settings -> Resources"
    echo "  2. Increase memory/disk limits"
    echo "  3. Run: docker system prune -a (WARNING: removes all unused images)"
    exit 1
fi
print_success "Docker images built successfully"

echo "Starting services..."
if ! docker compose up -d; then
    print_error "Failed to start services"
    echo "Check logs with: docker compose logs"
    exit 1
fi
print_success "Services started"

# ============================================================================
# STEP 5: Wait for Services to be Ready
# ============================================================================
print_step "Waiting for Services to Initialize"

MAX_RETRIES=30
RETRY_COUNT=0
API_READY=false

echo "Waiting for API to be ready (up to 60 seconds)..."

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]] && [[ "$API_READY" == false ]]; do
    if curl -s -f "http://localhost:8000/health" > /dev/null 2>&1; then
        API_READY=true
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -n "."
        sleep 2
    fi
done

echo ""

if [[ "$API_READY" == true ]]; then
    print_success "API is healthy and ready!"
else
    print_warn "API health check timed out"
    echo ""
    echo "The API may still be starting. Check with:"
    echo "  docker compose -f $PROJECT_PATH/docker/compose.yaml logs api"
fi

# ============================================================================
# STEP 6: Verify API Endpoints
# ============================================================================
print_step "Verifying API Endpoints"

# Test /health
HEALTH_RESPONSE=$(curl -s "http://localhost:8000/health" 2>/dev/null || echo "")
if [[ -n "$HEALTH_RESPONSE" ]]; then
    print_success "/health: $HEALTH_RESPONSE"
else
    print_warn "/health endpoint not responding"
fi

# Test /version
VERSION_RESPONSE=$(curl -s "http://localhost:8000/version" 2>/dev/null || echo "")
if [[ -n "$VERSION_RESPONSE" ]]; then
    print_success "/version: $VERSION_RESPONSE"
else
    print_warn "/version endpoint not responding"
fi

# Test /predict
PREDICT_RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}' 2>/dev/null || echo "")
if [[ -n "$PREDICT_RESPONSE" ]]; then
    print_success "/predict: $PREDICT_RESPONSE"
else
    print_warn "/predict endpoint not responding"
fi

# ============================================================================
# STEP 7: Display Service URLs
# ============================================================================
print_step "Service URLs"

echo ""
echo "  Service             URL                              Credentials"
echo "  ----------------    -----------------------------    -------------------"
echo "  FastAPI (Predict)   http://localhost:8000/predict    -"
echo "  FastAPI (Docs)      http://localhost:8000/docs       -"
echo "  FastAPI (Health)    http://localhost:8000/health     -"
echo "  Prometheus          http://localhost:9090            -"
echo "  Grafana             http://localhost:3000            (anonymous access enabled)"
echo "  MLflow              http://localhost:5001            -"
echo ""

# ============================================================================
# STEP 8: Quick Test Commands
# ============================================================================
print_step "Quick Test Commands"

echo ""
echo "# Health check:"
echo "curl http://localhost:8000/health"
echo ""
echo "# Make a prediction:"
echo 'curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '\''{"rows":[{"ret_mean":0.05,"ret_std":0.01,"n":50}]}'\'
echo ""
echo "# Check version:"
echo "curl http://localhost:8000/version"
echo ""
echo "# View logs:"
echo "cd $PROJECT_PATH/docker && docker compose logs -f api"
echo ""
echo "# Stop services:"
echo "cd $PROJECT_PATH/docker && docker compose down"
echo ""

# ============================================================================
# STEP 9: Open Browser
# ============================================================================
print_step "Opening Grafana Dashboard"

# Open Grafana in browser
if [[ "$OS" == "Darwin" ]]; then
    open "http://localhost:3000/d/crypto-volatility-api" 2>/dev/null || true
elif command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:3000/d/crypto-volatility-api" 2>/dev/null || true
fi
print_success "Opened Grafana dashboard (anonymous access - no login required)"

# ============================================================================
# COMPLETE
# ============================================================================
echo -e "${GREEN}"
cat << 'EOF'

========================================================================
                    SETUP COMPLETE!
========================================================================

The Real-Time Crypto Volatility Detection Service is now running.

Next steps:
1. View Grafana dashboard at http://localhost:3000 (no login required)
2. Test the /predict endpoint with the commands above
3. View API documentation at http://localhost:8000/docs

To stop all services:
  cd docker && docker compose down

To view logs:
  cd docker && docker compose logs -f

TROUBLESHOOTING:
- If Grafana shows "No data": Wait 1-2 minutes for metrics to populate
- If API returns errors: Check logs with 'docker compose logs api'
- If containers won't start: Check Docker Desktop resources/settings

========================================================================

EOF
echo -e "${NC}"


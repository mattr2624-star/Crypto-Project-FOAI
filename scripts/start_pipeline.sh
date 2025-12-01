#!/bin/bash
# ============================================================================
# Crypto Volatility Pipeline - Quick Start Script (Linux/Mac)
# ============================================================================
# This script starts all services and begins generating predictions
# to populate the Grafana dashboard with live metrics.
#
# Usage: ./scripts/start_pipeline.sh [--demo-duration SECONDS]
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
DEMO_DURATION=${1:-60}  # Default 60 seconds of demo predictions
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${CYAN}"
echo "============================================================"
echo "  Crypto Volatility Pipeline - Quick Start"
echo "============================================================"
echo -e "${NC}"

# Step 1: Check Docker
echo -e "${CYAN}[Step 1/4]${NC} Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Docker not found. Please install Docker Desktop."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Docker daemon not running. Please start Docker Desktop."
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Docker is running"

# Step 2: Start services
echo -e "\n${CYAN}[Step 2/4]${NC} Starting Docker services..."
cd "$PROJECT_DIR/docker"

# Stop any existing containers
docker compose down --remove-orphans 2>/dev/null || true

# Start fresh
docker compose up -d --build

echo -e "${GREEN}[OK]${NC} Services started"

# Step 3: Wait for API
echo -e "\n${CYAN}[Step 3/4]${NC} Waiting for API to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "http://localhost:8000/health" > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} API is healthy"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n${YELLOW}[WARN]${NC} API health check timed out, but continuing..."
fi

# Step 4: Generate predictions
echo -e "\n${CYAN}[Step 4/4]${NC} Generating predictions for dashboard..."
cd "$PROJECT_DIR"

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${YELLOW}[WARN]${NC} Python not found, skipping prediction generation"
    PYTHON_CMD=""
fi

if [ -n "$PYTHON_CMD" ]; then
    # Install requests if needed
    $PYTHON_CMD -c "import requests" 2>/dev/null || pip install requests -q
    
    # Run prediction consumer in background for demo duration
    echo -e "${GREEN}[INFO]${NC} Running predictions for ${DEMO_DURATION} seconds..."
    $PYTHON_CMD scripts/prediction_consumer.py --mode demo --interval 2 --duration $DEMO_DURATION &
    CONSUMER_PID=$!
    
    # Wait a moment then open browser
    sleep 3
fi

# Open Grafana dashboard
echo -e "\n${CYAN}Opening Grafana dashboard...${NC}"
DASHBOARD_URL="http://localhost:3000/d/crypto-volatility-api"

if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$DASHBOARD_URL" 2>/dev/null || true
elif command -v xdg-open &> /dev/null; then
    xdg-open "$DASHBOARD_URL" 2>/dev/null || true
fi

# Summary
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}  Pipeline Started Successfully!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Services running:"
echo "    • Grafana Dashboard:  http://localhost:3000 (no login required)"
echo "    • API Documentation:  http://localhost:8000/docs"
echo "    • Prometheus:         http://localhost:9090"
echo "    • MLflow:             http://localhost:5001"
echo ""
echo "  Commands:"
echo "    • Stop services:      cd docker && docker compose down"
echo "    • View logs:          cd docker && docker compose logs -f"
echo "    • More predictions:   python scripts/prediction_consumer.py --mode demo"
echo ""

# Wait for consumer if running
if [ -n "$CONSUMER_PID" ]; then
    wait $CONSUMER_PID 2>/dev/null || true
    echo -e "\n${GREEN}[DONE]${NC} Demo predictions complete. Dashboard should show metrics."
fi


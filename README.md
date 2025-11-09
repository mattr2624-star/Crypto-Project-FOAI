# Crypto Volatility Detection - Milestone 1

**Author:** Melissa Wong  
**Course:** Operationalize AI  
**Date:** November 8, 2025

Real-time cryptocurrency volatility detection system using Coinbase WebSocket API, Kafka, and MLflow.

---

## 📋 Milestone 1 Objectives

✅ Launch Kafka and MLflow infrastructure using Docker Compose  
✅ Ingest real-time Coinbase WebSocket ticker data  
✅ Implement reconnect/resubscribe and heartbeat monitoring  
✅ Stream data to Kafka topic `ticks.raw`  
✅ Validate data flow with consumer script  
✅ Define problem scope and success criteria  
✅ Containerize ingestion service  

---

## 🏗️ Project Structure

```
.
├── docker/
│   ├── docker-compose.yaml       # Infrastructure setup (Kafka, Zookeeper, MLflow)
│   └── Dockerfile.ingestor       # Containerized data ingestion service
├── scripts/
│   ├── ws_ingest.py              # WebSocket data ingestion with reconnect logic
│   └── kafka_consume_check.py    # Kafka stream validation tool
├── data/
│   └── raw/                      # Local mirror of raw ticker data (NDJSON)
├── docs/
│   └── scoping_brief.pdf         # Problem definition and success metrics
├── config.yaml                   # Configuration (optional)
├── .env                          # Environment variables (not committed)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Python 3.9+ with pip
- Git

### 1. Clone Repository

```bash
git clone <repository-url>
cd operationaliseai
```

### 2. Start Infrastructure

```bash
cd docker
docker compose up -d
```

Verify all services are running:
```bash
docker compose ps
```

Expected output:
- ✅ `kafka` - Running on port 9092
- ✅ `zookeeper` - Running on port 2182
- ✅ `mlflow` - Running on port 5001

Access MLflow UI: http://localhost:5001

### 3. Create Kafka Topics

```bash
# Create raw ticks topic
docker exec -it kafka kafka-topics --create \
  --topic ticks.raw \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Create features topic (for Milestone 2)
docker exec -it kafka kafka-topics --create \
  --topic ticks.features \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Verify topics exist
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

### 4. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create `.env` file in project root (if needed):
```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

**Note:** For running locally, add hostname resolution:
```bash
echo "127.0.0.1 kafka" | sudo tee -a /etc/hosts
```

---

## 📊 Running Data Ingestion

### Option A: Local Execution (Development)

```bash
# From project root with virtual environment activated
python scripts/ws_ingest.py --pair BTC-USD --minutes 15 --save-disk
```

**Arguments:**
- `--pair`: Trading pair (e.g., BTC-USD, ETH-USD)
- `--minutes`: Duration to run (default: 15)
- `--save-disk`: Mirror data to `data/raw/` directory

### Option B: Docker Container (Production-like)

```bash
# Build container
docker build -f docker/Dockerfile.ingestor -t crypto-ingestor .

# Run container
docker run --rm \
  --network docker_crypto-network \
  -v $(pwd)/data:/app/data \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  crypto-ingestor \
  python scripts/ws_ingest.py --pair BTC-USD --minutes 15 --save-disk
```

---

## ✅ Validating Data Flow

### Check Kafka Messages

In a separate terminal:

```bash
# Validate at least 100 messages received
python scripts/kafka_consume_check.py --topic ticks.raw --min 100
```

### Inspect Raw Data Files

```bash
# View most recent data file
ls -lth data/raw/ | head -5

# Preview contents (first 5 lines)
head -5 data/raw/ticks_BTCUSD_*.ndjson
```

### Monitor with Kafka Console Consumer

```bash
docker exec -it kafka kafka-console-consumer \
  --topic ticks.raw \
  --bootstrap-server localhost:9092 \
  --from-beginning \
  --max-messages 10
```

---

## 📄 Data Format

### Raw Ticker Message Schema

```json
{
  "timestamp": "2025-11-08T20:15:42.123456",
  "product_id": "BTC-USD",
  "price": "76543.21",
  "volume_24h": "12345.67890123",
  "low_24h": "75000.00",
  "high_24h": "77000.00",
  "best_bid": "76543.20",
  "best_ask": "76543.22",
  "raw": { ... }
}
```

**Fields:**
- `timestamp`: ISO 8601 capture time (UTC)
- `product_id`: Trading pair identifier
- `price`: Last trade price
- `volume_24h`: 24-hour trading volume
- `low_24h` / `high_24h`: 24-hour price range
- `best_bid` / `best_ask`: Top of order book
- `raw`: Complete Coinbase WebSocket message

---

## 🧪 Testing & Verification

### Test Checklist

- [ ] All Docker services show "Up" status
- [ ] Kafka topics created successfully
- [ ] Local ingestion runs for 15 minutes without errors
- [ ] At least 100 messages received in `ticks.raw`
- [ ] Container builds without errors
- [ ] Container runs and streams data successfully
- [ ] MLflow UI accessible at http://localhost:5001

### Run All Tests

```bash
# 1. Check infrastructure
cd docker && docker compose ps

# 2. Run ingestion (15 minutes)
python scripts/ws_ingest.py --pair BTC-USD --minutes 15 --save-disk

# 3. Validate messages (in separate terminal)
python scripts/kafka_consume_check.py --topic ticks.raw --min 100

# 4. Test container
docker build -f docker/Dockerfile.ingestor -t crypto-ingestor .
docker run --rm --network docker_crypto-network \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  crypto-ingestor \
  python scripts/ws_ingest.py --pair BTC-USD --minutes 2
```

---

## 🔧 Troubleshooting

### Kafka Connection Issues

**Problem:** `DNS lookup failed for kafka:9092`

**Solution:**
```bash
# Add hostname resolution (for local execution)
echo "127.0.0.1 kafka" | sudo tee -a /etc/hosts
```

### MLflow Shows "Unhealthy"

**Check if actually working:**
```bash
curl http://localhost:5001/health
# Should return: OK
```

If returns `OK`, MLflow is functional despite health check status.

### No Messages in Kafka

**Debugging steps:**
```bash
# 1. Check WebSocket connection in logs
# Look for: "WebSocket connected to wss://..."

# 2. Verify topics exist
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# 3. Check Kafka producer logs for errors
```

### Port Conflicts

**Ports in use:**
- 5001: MLflow UI
- 9092: Kafka broker
- 2182: Zookeeper

**Change ports in `docker/docker-compose.yaml` if needed**

---

## 📚 Key Components

### WebSocket Ingestion (`ws_ingest.py`)

**Features:**
- ✅ Auto-reconnect on connection loss
- ✅ Heartbeat monitoring (30-second timeout)
- ✅ Graceful shutdown (Ctrl+C)
- ✅ Dual output: Kafka stream + local NDJSON files
- ✅ Structured logging

**Error Handling:**
- Exponential backoff for reconnection
- Message validation before Kafka publish
- Connection state tracking

### Kafka Consumer Validator (`kafka_consume_check.py`)

**Purpose:** Verify streaming pipeline health

**Usage:**
```bash
python scripts/kafka_consume_check.py --topic ticks.raw --min 100
```

**Output:**
- Message count validation
- Sample message preview
- Success/failure status

---

## 📖 Documentation

### Scoping Brief

See `docs/scoping_brief.pdf` for:
- Use case and business context
- 60-second volatility prediction goal
- Success metrics (PR-AUC ≥ 0.70)
- Risk assumptions and constraints
- Labeling strategy

---

## 🔐 Security Notes

- **No secrets committed:** API keys and credentials in `.env` (gitignored)
- **Public data only:** Using free Coinbase WebSocket API
- **No trading:** Analysis and detection only

---

---

## 🎯 Milestone 2: Feature Engineering & Analysis

**Status:** ✅ Complete

### Objectives Achieved

✅ Built streaming feature engineering pipeline  
✅ Implemented replay script for reproducibility  
✅ Conducted exploratory data analysis (EDA)  
✅ Selected volatility spike threshold (τ)  
✅ Generated Evidently data quality report  
✅ Documented feature specifications  

### New Components

```
features/
├── featurizer.py                 # Kafka consumer for real-time feature computation
└── __init__.py

scripts/
├── replay.py                     # Reproducibility verification script
├── generate_evidently_report.py  # Data drift monitoring
└── check_milestone2.py           # Milestone 2 verification checklist

notebooks/
└── eda.ipynb                     # Exploratory analysis & threshold selection

docs/
└── feature_spec.md               # Feature engineering documentation

data/
├── processed/
│   ├── features.parquet          # Live-computed features
│   └── features_replay.parquet   # Replay-computed features
└── reports/
    └── evidently_report.html     # Data quality & drift report
```

### Running Milestone 2 Pipeline

#### 1. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

New dependencies include:
- `pandas==2.1.4` - DataFrame operations
- `pyarrow==14.0.1` - Parquet file support
- `evidently==0.4.11` - Drift monitoring
- `jupyter==1.0.0` - Notebook environment

#### 2. Run Feature Engineering Pipeline

**Real-time streaming mode:**
```bash
python features/featurizer.py --topic_in ticks.raw --topic_out ticks.features
```

This consumes from `ticks.raw`, computes rolling window features, and publishes to `ticks.features`.

**Features computed:**
- Price returns (1min, 5min rolling windows)
- Volatility (rolling standard deviation)
- Bid-ask spread dynamics
- Volume-weighted metrics
- Trade intensity indicators

Let it run for 10-15 minutes to accumulate sufficient data.

#### 3. Verify Reproducibility with Replay

```bash
python scripts/replay.py \
  --raw "data/raw/*.ndjson" \
  --out data/processed/features_replay.parquet
```

This re-processes raw data through the same feature pipeline and compares outputs to verify deterministic behavior.

**Expected output:**
```
✓ Features match between live and replay
✓ Row counts identical
✓ Reproducibility verified
```

#### 4. Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

**Analysis includes:**
- Feature distribution visualization
- Correlation analysis
- Volatility pattern identification
- Threshold selection (90th percentile)
- Label generation strategy

**Key finding:** Volatility threshold (τ) set at [YOUR VALUE]% based on 90th percentile of rolling standard deviation.

#### 5. Generate Evidently Report

```bash
python scripts/generate_evidently_report.py
```

Generates `data/reports/evidently_report.html` with:
- Data quality metrics
- Feature drift detection
- Distribution comparisons (early vs late data windows)
- Missing value analysis
- Statistical test results

**View report:**
```bash
open data/reports/evidently_report.html
```

#### 6. Verify Milestone Completion

```bash
python scripts/check_milestone2.py
```

Checklist includes:
- ✓ Feature pipeline files present
- ✓ Processed features exist
- ✓ Replay features match live features
- ✓ EDA notebook completed
- ✓ Feature specification documented
- ✓ Evidently report generated

---

## 📊 Feature Engineering Details

### Feature Categories

**1. Price-Based Features**
- `price_return_1min` - 1-minute price return
- `price_return_5min` - 5-minute price return
- `price_volatility_5min` - Rolling standard deviation

**2. Spread Features**
- `bid_ask_spread` - Absolute spread
- `bid_ask_spread_bps` - Spread in basis points

**3. Volume Features**
- `volume_24h_pct_change` - 24-hour volume change
- `trade_intensity` - Trades per minute (estimated)

**4. Target Variable**
- `volatility_spike` - Binary label (1 = spike detected, 0 = normal)

### Labeling Strategy

**Definition of Volatility Spike:**
```python
# Look-ahead window: 60 seconds
future_volatility = rolling_std(returns, window=60s)
threshold = percentile_90(historical_volatility)
label = 1 if future_volatility >= threshold else 0
```

### Data Quality Findings

From Evidently report:
- **Missing data rate:** [X]%
- **Feature drift detected:** [Yes/No]
- **Data distribution shifts:** [Description]
- **Recommended actions:** [Retraining schedule, monitoring thresholds]

---

## 🔄 Reproducibility Verification

### Replay Testing

The replay script ensures deterministic feature computation:

```bash
# Step 1: Collect live data
python features/featurizer.py --duration 15

# Step 2: Replay same data
python scripts/replay.py --raw "data/raw/*.ndjson"

# Step 3: Compare outputs
python scripts/check_milestone2.py
```

**Success criteria:**
- Feature values match to floating-point precision
- Row counts identical between live and replay
- Timestamps align correctly

---

## 📈 Next Steps (Milestone 3)

- [ ] Train baseline model (rule-based)
- [ ] Train ML model (Logistic Regression / XGBoost)
- [ ] Log experiments to MLflow
- [ ] Implement model serving pipeline
- [ ] Set up automated retraining
- [ ] Deploy monitoring dashboard

---

## 🛠️ Technology Stack

- **Language:** Python 3.9
- **Streaming:** Apache Kafka (KRaft mode)
- **Experiment Tracking:** MLflow 2.10.2
- **Container Orchestration:** Docker Compose
- **Data Processing:** Pandas, NumPy
- **Data Formats:** NDJSON, Parquet
- **Quality Monitoring:** Evidently 0.4.11
- **Analysis:** Jupyter Notebooks

---

## 🐛 Troubleshooting Milestone 2

### Featurizer Not Processing Messages

**Check:**
```bash
# Verify Kafka topics exist
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092

# Check raw data is flowing
python scripts/kafka_consume_check.py --topic ticks.raw --min 10
```

### Replay Features Don't Match

**Common causes:**
- Timestamp sorting issues
- Floating-point precision differences
- Missing data handling inconsistencies

**Fix:** Ensure consistent data sorting and NaN handling in both pipelines.

### Evidently Report Empty

**Check:**
```bash
# Verify features.parquet exists and has data
ls -lh data/processed/features.parquet
python -c "import pandas as pd; print(pd.read_parquet('data/processed/features.parquet').shape)"
```

Need at least 100+ rows for meaningful drift analysis.

---

## 📞 Support

For issues or questions:
1. Check troubleshooting sections above
2. Review Docker logs: `docker compose logs <service-name>`
3. Verify environment configuration
4. Run verification script: `python scripts/check_milestone2.py`
5. Consult course materials

---

## 📝 License

Educational project for Operationalize AI course.

---

## 📅 Project Timeline

**Milestone 1:** ✅ Complete (November 8, 2025) - Streaming infrastructure  
**Milestone 2:** ✅ Complete (November [DATE], 2025) - Feature engineering & analysis  
**Milestone 3:** 🔄 In Progress - Model training & deployment
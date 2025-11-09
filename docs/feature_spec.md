# Feature Specification

## Project: Real-Time Crypto Volatility Detection

**Date:** November 9, 2025  
**Author:** Melissa Wong

---

## 1. Problem Definition

**Use Case:** Detect short-term volatility spikes in cryptocurrency markets to enable proactive risk management and trading decisions.

**Prediction Goal:** Predict whether a volatility spike will occur in the next 60 seconds based on real-time market data.

---

## 2. Target Variable

### Target Horizon
**60 seconds** - We predict volatility in the next minute.

### Volatility Proxy
**Rolling standard deviation of midprice returns** over the future 60-second window.

Mathematically:
```
σ_future = std(r_t+1, r_t+2, ..., r_t+n)
where:
  r_i = (price_i - price_{i-1}) / price_{i-1}
  n = number of ticks in 60 seconds
```

### Label Definition
Binary classification:
```
label = 1  if σ_future >= τ  (volatility spike)
label = 0  if σ_future < τ   (normal conditions)
```

### Chosen Threshold (τ)
**Value:** `0.000026` (from EDA analysis, 90th percentile)

**Justification:**
- Selected at the **90th percentile** of observed future volatility
- Based on percentile analysis in EDA (see `notebooks/eda.ipynb`)
- This threshold captures the top 10% of volatile periods
- Results in exactly **10.0%** positive class (spikes) - 5,251 out of 52,524 samples

**Trade-offs:**
- Higher threshold → fewer false positives, but might miss moderate spikes
- Lower threshold → more sensitivity, but higher false alarm rate
- Current threshold balances detection rate with actionable signal quality

---

## 3. Features

### 3.1 Raw Features

| Feature | Description | Type | Source |
|---------|-------------|------|--------|
| `timestamp` | Event timestamp | datetime | Coinbase WebSocket |
| `product_id` | Trading pair (e.g., BTC-USD) | string | Coinbase WebSocket |
| `price` | Midprice: (best_bid + best_ask) / 2 | float | Computed |
| `best_bid` | Best bid price | float | Coinbase WebSocket |
| `best_ask` | Best ask price | float | Coinbase WebSocket |
| `spread` | Bid-ask spread (absolute) | float | Computed |
| `spread_bps` | Bid-ask spread in basis points | float | Computed |

### 3.2 Windowed Features

Features are computed over rolling windows: **30s, 60s (1min), and 300s (5min)**

#### Feature Categories

**Price-based Features:**
- **Simple Returns**: `r_t = (p_t - p_{t-1}) / p_{t-1}`
- **Log Returns**: `log_r_t = log(p_t / p_{t-1}) = log(p_t) - log(p_{t-1})` (more stable for crypto)
- **Rolling Volatility**: Standard deviation of returns over window
- **Price Momentum**: Mean, min, max returns over window

**Market Microstructure Features:**
- **Bid-ask Spread**: Absolute and basis points (bps)
- **Spread Volatility**: Standard deviation of spreads over window

**Volume/Activity Features:**
- **Trade Intensity**: Tick count per window
- **Time Since Last Trade**: Seconds since previous tick

#### Complete Feature List

| Feature Name | Formula | Window | Aggregation | Missing Value Handling |
|--------------|---------|--------|-------------|------------------------|
| `return_mean_{window}s` | `mean((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Mean | 0.0 |
| `return_std_{window}s` | `std((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Std Dev | 0.0 |
| `return_min_{window}s` | `min((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Min | 0.0 |
| `return_max_{window}s` | `max((p_t - p_{t-1}) / p_{t-1})` | 30s, 60s, 300s | Max | 0.0 |
| `log_return_mean_{window}s` | `mean(log(p_t) - log(p_{t-1}))` | 30s, 60s, 300s | Mean | 0.0 |
| `log_return_std_{window}s` | `std(log(p_t) - log(p_{t-1}))` | 30s, 60s, 300s | Std Dev | 0.0 |
| `price_mean_{window}s` | `mean(p_t)` | 30s, 60s, 300s | Mean | 0.0 |
| `price_std_{window}s` | `std(p_t)` | 30s, 60s, 300s | Std Dev | 0.0 |
| `tick_count_{window}s` | Count of ticks | 30s, 60s, 300s | Count | 0 |
| `spread_std_{window}s` | `std(spread_t)` | 30s, 60s, 300s | Std Dev | 0.0 |
| `spread_mean_{window}s` | `mean(spread_t)` | 30s, 60s, 300s | Mean | 0.0 |
| `time_since_last_trade` | `t_current - t_previous` (seconds) | N/A | Difference | 0.0 |
| `gap_seconds` | Time gap between consecutive ticks | N/A | Difference | 0.0 |

#### Features Used in Model (8 features)

The current model uses a subset of features selected based on feature separation analysis:

| Feature Name | Description | Window | Rationale |
|--------------|-------------|--------|-----------|
| `return_std_60s` | 60-second volatility | 60s | Best separation (0.78 std dev) |
| `return_std_30s` | 30-second volatility | 30s | Excellent separation (0.66 std dev) |
| `return_std_300s` | 300-second volatility | 300s | Longer-term context |
| `return_mean_60s` | 1-minute return mean | 60s | Good separation (0.74 std dev) |
| `return_mean_300s` | 5-minute return mean | 300s | Moderate separation (0.51 std dev) |
| `return_min_30s` | Minimum return in 30s | 30s | Good separation (0.64 std dev) |
| `tick_count_60s` | Trading intensity | 60s | Moderate separation (0.21 std dev) |
| `return_range_60s` | Return range (max - min) | 60s | Derived feature |

**Note:** Additional features are computed but not used in the current model. Log returns and spread volatility features are available for future model iterations.

### 3.3 Feature Engineering Rationale

**Why these features?**

1. **Multiple time windows** capture both short-term noise and longer-term trends
2. **Return statistics** directly measure price movement patterns
3. **Spread metrics** indicate market liquidity and stress
4. **Tick intensity** proxies for trading activity and information flow

**What we're NOT using (yet):**
- Order book imbalance (complexity vs benefit trade-off)
- Volume-weighted features (not available in ticker channel)
- Cross-asset correlations (single-pair focus for MVP)

---

## 4. Reproducibility & Determinism

### 4.1 Deterministic Computations
- ✅ No randomness in feature computation logic
- ✅ Fixed window boundaries (time-based, not tick-based)
- ✅ Deterministic aggregation functions (mean, std, min, max)

### 4.2 Timestamp Handling
- **Timezone:** UTC (explicitly enforced)
- **Format:** ISO 8601 or Unix timestamp (auto-detected)
- **Consistency:** All timestamps converted to UTC timezone-aware
- **Validation:** Timestamp ordering checked (warns on backward jumps >1s)

### 4.3 Replay Verification
- **Script:** `scripts/replay.py` verifies reproducibility
- **Method:** Re-process raw data through feature pipeline
- **Verification:** Compare replayed features with original (within 1e-6 tolerance)
- **Usage:** `python scripts/replay.py --raw data/raw/*.ndjson --compare data/processed/features.parquet`

### 4.4 Window Boundaries
- **Type:** Fixed-size sliding windows (time-based)
- **Boundaries:** `[current_time - window_seconds, current_time]` (inclusive)
- **Example:** At t=15:01:00, 60s window includes [15:00:00, 15:01:00]
- **Documentation:** Window logic documented in code comments

## 5. Data Processing Pipeline

### 5.1 Real-Time Pipeline
```
Coinbase WebSocket → Kafka (ticks.raw) → Featurizer → Kafka (ticks.features) → Parquet
```

### 5.2 Replay Pipeline (for reproducibility)
```
NDJSON files → replay.py → FeatureComputer → Parquet
```

**Validation:** Replay and live features must match exactly (verified via `scripts/replay.py`)

---

## 5. Data Quality Considerations

### 5.1 Missing Data Handling

**Strategy:** Set features to `0.0` for consistency (not `None` or `NaN`)

- **Midprice missing:** Skip tick (requires both bid and ask)
- **Insufficient window data:** Features set to `0.0` (occurs for first few ticks)
- **Timestamp issues:** Use current UTC time as fallback
- **NaN/Infinite values:** Detected and replaced with `0.0` (logged as warning)

**Rationale:** 
- Consistent handling downstream (train.py fills NaN with 0)
- Prevents downstream errors from None values
- Missing data is rare (< 0.01% in practice)

### 5.2 Gap Handling

**Gap Detection:**
- Feature `gap_seconds` tracks time between consecutive ticks
- Large gaps (>10s) logged as warnings
- Gaps are natural in crypto markets (24/7 trading but brief pauses)

**Current Strategy:**
- No forward-fill implemented (gaps preserved in features)
- Windowed features naturally handle gaps (fewer ticks = lower tick_count)
- Future enhancement: Forward-fill short gaps (<10s) with last known price

**Gap Tolerance:**
- Windows with >10% missing ticks may have reduced signal quality
- Documented in feature statistics but not filtered
- Model learns to handle variable tick density

### 5.3 Data Quality Checks

**Implemented Checks:**
- ✅ NaN detection and replacement
- ✅ Infinite value detection and replacement
- ✅ Timestamp ordering validation (warns on >1s backward jumps)
- ✅ Gap detection and logging

**Monitoring:**
- Periodic statistics logged (min, max, mean) during processing
- Quality issues logged as warnings
- Feature distributions tracked in Evidently reports

### 5.4 Edge Cases

- **Market reconnections:** Feature windows reset when buffer empty
- **Extreme outliers:** Not filtered in feature computation (model's job)
- **Time gaps:** Preserved in features (no interpolation)
- **Out-of-order timestamps:** Validated and logged (small backward jumps allowed)

### 5.5 Known Limitations

- Features lag reality by ~100-500ms (typical Kafka + compute latency)
- Window sizes fixed (not adaptive to market regime)
- No handling of trading halts or circuit breakers
- Volume-based features not available (ticker channel doesn't provide volume data)
- Forward-fill not implemented (gaps preserved)

---

## 6. Feature Statistics

**From EDA (`notebooks/eda.ipynb`) and processed data:**

| Metric | Value |
|--------|-------|
| Total samples | 52,524 |
| Time range | 2025-11-08 15:12:31 to 2025-11-09 01:25:17 (~10.2 hours) |
| Positive class % | 10.00% |
| Missing data % | 0.01% |
| Avg ticks/second | 1.43 |

**Feature Statistics (mean, std):**
- `return_mean_60s`: mean=-0.000000, std=0.000002
- `return_mean_300s`: mean=-0.000000, std=0.000001
- `return_std_300s`: mean=0.000040, std=0.000013
- `spread`: mean=0.271610, std=0.948744
- `spread_bps`: mean=0.026680, std=0.093189

---

## 7. Next Steps (Milestone 3)

1. **Train models** using these features
2. **Evaluate** using PR-AUC (primary metric)
3. **Monitor drift** between train and test distributions
4. **Iterate** on features based on model performance

---

## Appendix: Feature Correlation

**Correlation with target variable (`volatility_spike`):**

Top 3 features correlated with future volatility:
1. `return_std_300s`: r = 0.1917 (strongest predictor)
2. `return_mean_60s`: r = 0.0416
3. `return_mean_300s`: r = 0.0357

**Interpretation:**
- `return_std_300s` (5-minute volatility) shows the strongest positive correlation with future volatility spikes, confirming that recent volatility is a key indicator
- Short-term return means show weaker but positive correlations
- Spread features (`spread`, `spread_bps`) show minimal correlation with future volatility in this dataset
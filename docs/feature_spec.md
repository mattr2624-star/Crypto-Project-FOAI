# Feature Specification

## Project: Real-Time Crypto Volatility Detection

**Date:** [Insert Date]  
**Author:** [Your Name]

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
**Value:** `[INSERT YOUR THRESHOLD FROM EDA]` (e.g., 0.000423)

**Justification:**
- Selected at the **[INSERT PERCENTILE]th percentile** (e.g., 90th) of observed future volatility
- Based on percentile analysis in EDA (see `notebooks/eda.ipynb`)
- This threshold captures the top [X]% of volatile periods
- Results in approximately [Y]% positive class (spikes)

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

Features are computed over three rolling windows: **30s, 60s, and 300s (5min)**

#### Return Statistics (per window)
- `return_mean_{W}s`: Mean return over window W
- `return_std_{W}s`: Standard deviation of returns (volatility proxy)
- `return_min_{W}s`: Minimum return observed
- `return_max_{W}s`: Maximum return observed

#### Price Statistics (per window)
- `price_mean_{W}s`: Average price over window
- `price_std_{W}s`: Standard deviation of prices

#### Market Activity (per window)
- `tick_count_{W}s`: Number of ticks received in window (intensity proxy)

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

## 4. Data Processing Pipeline

### 4.1 Real-Time Pipeline
```
Coinbase WebSocket → Kafka (ticks.raw) → Featurizer → Kafka (ticks.features) → Parquet
```

### 4.2 Replay Pipeline (for reproducibility)
```
NDJSON files → replay.py → FeatureComputer → Parquet
```

**Validation:** Replay and live features must match exactly (verified via `scripts/replay.py`)

---

## 5. Data Quality Considerations

### 5.1 Missing Data Handling
- **Midprice missing:** Skip tick (requires both bid and ask)
- **Insufficient window data:** Features set to `None` or 0 for counts
- **Timestamp issues:** Use current system time as fallback

### 5.2 Edge Cases
- **Market reconnections:** Feature windows reset when buffer empty
- **Extreme outliers:** Not filtered in feature computation (model's job)
- **Time gaps:** No interpolation; gaps natural in windowed features

### 5.3 Known Limitations
- Features lag reality by ~[X]ms (typical Kafka + compute latency)
- Window sizes fixed (not adaptive to market regime)
- No handling of trading halts or circuit breakers

---

## 6. Feature Statistics

**From EDA (`notebooks/eda.ipynb`):**

| Metric | Value |
|--------|-------|
| Total samples | [INSERT] |
| Time range | [INSERT] |
| Positive class % | [INSERT]% |
| Missing data % | [INSERT]% |
| Avg ticks/second | [INSERT] |

---

## 7. Next Steps (Milestone 3)

1. **Train models** using these features
2. **Evaluate** using PR-AUC (primary metric)
3. **Monitor drift** between train and test distributions
4. **Iterate** on features based on model performance

---

## Appendix: Feature Correlation

[Include key correlation findings from EDA]

Top 3 features correlated with future volatility:
1. [Feature name]: r = [value]
2. [Feature name]: r = [value]
3. [Feature name]: r = [value]
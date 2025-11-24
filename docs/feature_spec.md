# Feature Specification

## Project: Real-Time Crypto Volatility Detection

**Date:** November 13, 2025  
**Author:** Melissa Wong  
**Version:** 1.2

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

**Implementation:** Chunk-aware forward-looking calculation that respects data collection gaps.

Mathematically:
```
σ_future = std(r_t+1, r_t+2, ..., r_t+n)
where:
  r_i = (price_i - price_{i-1}) / price_{i-1}
  n = number of ticks in [t, t+60 seconds]
  
Constraint: Only ticks within the same data collection chunk are considered
(no calculation across gaps > gap_threshold_seconds)
```

**Key Changes (v1.1):**
- **Chunk-aware calculation:** Volatility computed only within continuous data segments
- **Forward-looking:** For each timestamp, finds all ticks in the next 60 seconds and computes std of returns
- **Gap handling:** Data collection gaps (>300s default) define chunk boundaries
- **Iterative method:** Correctly handles variable tick density and ensures no look-ahead bias

**Key Changes (v1.2):**
- **Refocused feature set:** Streamlined to focus on Momentum & Volatility, Liquidity & Microstructure, and Activity features
- **New features:** Added Realized Volatility (1-second returns), Price Velocity, Order Book Imbalance, and Volume Velocity
- **Improved windowing:** All features use proper rolling window strategies (REQUIRED for most features)
- **1-second granularity:** Realized volatility and price velocity computed from 1-second returns/changes for precision

### Label Definition
Binary classification:
```
label = 1  if σ_future >= τ  (volatility spike)
label = 0  if σ_future < τ   (normal conditions)
```

### Chosen Threshold (τ)
**Value:** 90th percentile (configurable, default: 90)

**Justification:**
- Selected at the **90th percentile** of observed future volatility within each data chunk
- Based on percentile analysis in EDA (see `notebooks/eda.ipynb`)
- This threshold captures the top 10% of volatile periods
- **Chunk-aware:** Threshold calculated separately for each data collection chunk to account for temporal variations
- Results in approximately **10.0%** positive class (spikes) when data is balanced

**Trade-offs:**
- Higher threshold → fewer false positives, but might miss moderate spikes
- Lower threshold → more sensitivity, but higher false alarm rate
- Current threshold balances detection rate with actionable signal quality

**Configuration:**
- Default: `label_threshold_percentile=90` (configurable in `featurizer.py`)
- Gap threshold: `label_gap_threshold_seconds=300` (5 minutes) defines chunk boundaries

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

**1. Momentum & Volatility (Price Trends)**
These features measure how fast and how erratically the price is changing:
- **Log Returns**: Logarithmic difference between current mid-price and mid-price at window start
- **Realized Volatility**: Rolling standard deviation of 1-second returns (target proxy, volatility clusters)
- **Price Velocity**: Rolling mean of absolute 1-second price changes

**2. Liquidity & Microstructure (Market Nerves)**
These features measure the "friction" in the market. When friction increases, volatility often follows:
- **Bid-Ask Spread**: Rolling mean of spread (smoothed to reduce noise from raw tick data)
- **Order Book Imbalance (OBI)**: Rolling mean of buy volume vs. sell volume ratio at top of book

**3. Activity (Market Energy)**
These features measure the sheer volume of action. High energy often precedes a breakout:
- **Trade Intensity**: Rolling sum of tick count (number of trades/messages in window)
- **Volume Velocity**: Rolling sum of trade sizes over the window

#### Complete Feature List

| Feature Name | Formula | Window | Aggregation | Missing Value Handling | Description |
|--------------|---------|--------|-------------|------------------------|-------------|
| `log_return_{window}s` | `log(p_current / p_window_start)` | 30s, 60s, 300s | Log ratio | 0.0 | Log return over fixed lookback period |
| `realized_volatility_{window}s` | `std(r_1s)` where `r_1s` are 1-second returns | 30s, 60s, 300s | Std Dev | 0.0 | Rolling std dev of 1-second returns (target proxy) |
| `price_velocity_{window}s` | `mean(\|Δp_1s\|)` where `Δp_1s` are 1-second changes | 30s, 60s, 300s | Mean | 0.0 | Rolling mean of absolute 1-second price changes |
| `spread_mean_{window}s` | `mean(spread_t)` | 30s, 60s, 300s | Mean | 0.0 | Rolling mean of bid-ask spread (smoothed) |
| `order_book_imbalance_{window}s` | `mean(bid_qty / (bid_qty + ask_qty))` | 30s, 60s, 300s | Mean | 0.5 | Rolling mean of order book imbalance (0.5 = neutral) |
| `trade_intensity_{window}s` | `sum(1)` for each tick | 30s, 60s, 300s | Sum | 0 | Rolling sum of tick count (trade intensity) |
| `volume_velocity_{window}s` | `sum(trade_size_t)` | 30s, 60s, 300s | Sum | 0.0 | Rolling sum of trade sizes (0.0 if not available) |
| `time_since_last_trade` | `t_current - t_previous` (seconds) | N/A | Difference | 0.0 | Seconds since previous tick |
| `gap_seconds` | Time gap between consecutive ticks | N/A | Difference | 0.0 | Time gap between consecutive ticks |

#### Features Used in Model

**Note:** With the new feature set (v1.2), model retraining is required. The previous model used 10 features from the old feature set. The new focused feature set includes:

**New Feature Set (7 features per window × 3 windows = 21 windowed features):**
- `log_return_{window}s` - Log returns over fixed lookback periods
- `realized_volatility_{window}s` - **Most critical predictor** (target proxy, volatility clusters)
- `price_velocity_{window}s` - Momentum indicator
- `spread_mean_{window}s` - Liquidity indicator
- `order_book_imbalance_{window}s` - Microstructure signal
- `trade_intensity_{window}s` - Activity measure
- `volume_velocity_{window}s` - Volume activity (may be 0.0 if unavailable)

**Model Retraining Required:**
- Previous model performance metrics (PR-AUC 0.7815 with XGBoost) are based on old feature set
- New model should evaluate feature importance and selection from the new feature set
- Expected that `realized_volatility_{window}s` will be highly predictive (it's the target proxy)
- Feature selection should focus on minimizing multicollinearity while maximizing predictive power

#### Baseline Model Features

**Note:** The baseline model will need to be updated to use the new feature set. Previous baseline used 8 features from the old feature set.

**Recommended Baseline Features (from new set):**
- `realized_volatility_{window}s` - Most critical (target proxy)
- `log_return_{window}s` - Price trend indicator
- `price_velocity_{window}s` - Momentum indicator
- `spread_mean_{window}s` - Liquidity indicator
- `order_book_imbalance_{window}s` - Microstructure signal
- `trade_intensity_{window}s` - Activity measure

**Baseline Method (to be updated):**
1. Standardize each feature using training mean/std
2. Compute per-feature z-scores
3. Calculate composite score as mean of z-scores (weighted by feature importance)
4. Apply threshold (default: 2.0) to composite z-score
5. Predict spike if composite z-score >= threshold

### 3.3 Feature Engineering Rationale

**Why these features?**

1. **Momentum & Volatility Features:**
   - **Log Returns**: Measure price trends over fixed lookback periods (more stable than simple returns for crypto)
   - **Realized Volatility**: Critical predictor - volatility clusters (high volatility follows high volatility). Computed as rolling std dev of 1-second returns.
   - **Price Velocity**: Measures absolute rate of price change per second, capturing momentum

2. **Liquidity & Microstructure Features:**
   - **Bid-Ask Spread**: Rolling mean smooths noise from raw tick data, providing clearer signal of sustained liquidity drying up
   - **Order Book Imbalance**: Sustained imbalance over 5-10 seconds indicates strong directional pressure (not just "flickering quotes")

3. **Activity Features:**
   - **Trade Intensity**: Rolling sum captures total number of trades in window (cannot measure intensity at single point)
   - **Volume Velocity**: Rolling sum of trade sizes measures total quantity traded (may be 0.0 if not available in ticker channel)

4. **Multiple time windows** (30s, 60s, 300s) capture different aspects of market dynamics

**Implementation Details:**
- **1-second returns**: Computed on-the-fly by finding prices approximately 1 second apart (0.5-1.5s tolerance for sparse data)
- **Order Book Imbalance**: Extracted from `best_bid_quantity` and `best_ask_quantity` in tick data (from `raw` field)
- **Volume Velocity**: Handles missing data gracefully (ticker channel may not provide per-tick volume)

**What we're NOT using (yet):**
- Cross-asset correlations (single-pair focus for MVP)
- Volume-weighted price features (not available in ticker channel)

**Performance Impact (Current Model - v1.2):**
- **Random Forest (Production):** PR-AUC 0.9859 with 10-feature set (new feature set, consolidated dataset)
- **Test Set Metrics:** Accuracy 0.9888, Precision 0.9572, Recall 0.9372, F1-Score 0.9471, ROC-AUC 0.9983
- **Validation Set Metrics:** PR-AUC 0.9806, Accuracy 0.9903, Precision 0.9557, Recall 0.9535, F1-Score 0.9546
- **Top Features:** Order Book Imbalance (18.8%), Trade Intensity (17.1%), Spread Mean (14.9%)
- **Improvement:** Random Forest outperforms baseline by 132.5% (Baseline: 0.4240)
- **Model Selection:** Random Forest selected over XGBoost (0.5573) based on best test performance
- **Dataset:** Consolidated dataset (26,881 samples) with stratified splits (70/15/15 train/val/test)

**Previous Performance (v1.1 - for reference):**
- **XGBoost (Stratified):** PR-AUC 0.7815 with 10-feature set (old feature set)
- **Feature reduction:** Removing perfectly correlated features improved Logistic Regression PR-AUC by +6.6%
- **Stratified splitting:** Balancing spike rates across splits improved XGBoost PR-AUC from 0.7359 to 0.7815

**Feature Set Evolution:**
- **v1.2 (Current):** Focus on Momentum & Volatility, Liquidity & Microstructure, Activity features
- **v1.1 (Previous):** Log return volatility, return statistics, spread volatility, trade intensity
- **Realized Volatility** is highly predictive (target proxy) and included in top features
- **Order Book Imbalance** provides strong microstructure signal (top feature at 18.8% importance)

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
NDJSON files → replay.py → FeatureComputer → FeaturePipeline._add_labels_to_dataframe → Parquet
```

**Validation:** Replay and live features must match exactly (verified via `scripts/replay.py`)

**Label Creation:**
- Labels (`volatility_spike`) are created using `FeaturePipeline._add_labels_to_dataframe`
- Chunk-aware calculation ensures labels respect data collection gaps
- Can be added during feature generation (`--add-labels` flag) or separately via `scripts/add_labels.py`

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

**Chunk Detection (v1.1):**
- Data collection gaps (>300s default) define chunk boundaries
- Chunks are detected automatically during label creation
- Volatility calculation respects chunk boundaries (no calculation across gaps)
- Prevents artificial volatility spikes from connecting unrelated data segments

**Current Strategy:**
- No forward-fill implemented (gaps preserved in features)
- Windowed features naturally handle gaps (fewer ticks = lower tick_count)
- Chunk-aware label creation ensures forward-looking volatility only uses ticks within same chunk
- Future enhancement: Forward-fill short gaps (<10s) with last known price

**Gap Tolerance:**
- Windows with >10% missing ticks may have reduced signal quality
- Documented in feature statistics but not filtered
- Model learns to handle variable tick density
- Chunk boundaries prevent cross-gap calculations that could introduce bias

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
- **Volume Velocity** may be 0.0 if trade sizes not available in ticker channel (ticker channel doesn't always provide per-tick volume)
- Forward-fill not implemented (gaps preserved)
- **1-second returns**: Uses 0.5-1.5s tolerance window to handle sparse tick data (may miss exact 1-second intervals)

---

## 6. Feature Statistics

**From EDA (`notebooks/eda.ipynb`) and processed data:**

| Metric | Value |
|--------|-------|
| Total samples | ~9,629 (after filtering) |
| Time range | 2025-11-08 15:12:31 to 2025-11-09 01:25:17 (~10.2 hours) |
| Positive class % | ~10.00% (varies by split method) |
| Missing data % | 0.01% |
| Avg ticks/second | 1.43 |
| Data chunks | Multiple (gaps >300s define boundaries) |

**Data Split Statistics:**

**Time-Based Split:**
- Training: ~6,740 samples (6.60% spike rate)
- Validation: ~1,444 samples (2.42% spike rate)
- Test: ~1,445 samples (33.43% spike rate)
- *Note: Temporal clustering causes test set to have much higher spike rate*

**Stratified Split (Recommended):**
- Training: ~6,740 samples (10.0% spike rate)
- Validation: ~1,444 samples (10.0% spike rate)
- Test: ~1,445 samples (10.0% spike rate)
- *Note: Balanced spike rates improve model performance*

**Feature Statistics (mean, std):**
- `return_mean_60s`: mean≈0.0, std≈0.000002
- `return_mean_300s`: mean≈0.0, std≈0.000001
- `log_return_std_60s`: Best separation (0.569 std dev between classes)
- `spread`: mean≈0.27, std≈0.95
- `spread_bps`: mean≈0.027, std≈0.093

---

## 7. Model Performance with These Features

**Previous Model Performance (v1.1 - Old Feature Set):**
- **Best Model: XGBoost (Stratified Split)**
- PR-AUC: 0.7815 (Test)
- Recall: 97.31%
- Precision: 52.87%
- F1-Score: 0.6851

**Key Findings (v1.1):**
- **Stratified splitting** significantly improves performance (XGBoost PR-AUC: 0.7359 → 0.7815)
- **Chunk-aware label creation** ensures correct forward-looking volatility calculation
- **10-feature set** provided good balance between information and model complexity
- **Log returns** preferred over simple returns for crypto volatility modeling

**Feature Importance (v1.1 - XGBoost):**
- Top features: `log_return_std_60s`, `log_return_std_300s`, `return_range_60s`
- Spread features (`spread_std_300s`, `spread_mean_60s`) contribute to model performance
- Trade intensity (`tick_count_60s`) provides additional signal

**New Model Performance (v1.2 - New Feature Set):**
- **Model retraining required** - Performance metrics to be determined after retraining
- **Expected:** `realized_volatility_{window}s` will be highly predictive (target proxy)
- **Expected:** Focused feature set should maintain or improve performance while improving interpretability

## 8. Next Steps & Future Enhancements

1. ✅ **Train models** using old feature set - Complete (v1.1)
2. ✅ **Evaluate** using PR-AUC (primary metric) - Complete (v1.1)
3. ✅ **Monitor drift** between train and test distributions - Complete (v1.1)
4. ✅ **Iterate** on features based on model performance - Complete (v1.2)
5. 🔄 **Retrain models** using new focused feature set - **In Progress (v1.2)**
6. 🔄 **Evaluate** new feature set performance - **Pending (v1.2)**
7. 🔄 **Update API** to use new feature set - **Pending (v1.2)**
8. **Future:** Explore additional features (order book depth beyond top level, volume-weighted metrics)
9. **Future:** Adaptive window sizes based on market regime
10. **Future:** Multi-asset features for cross-market signals

---

## Appendix: Feature Correlation & Importance

**Correlation with target variable (`volatility_spike`) - Previous Model (v1.1):**

Top features correlated with future volatility:
1. `log_return_std_60s`: Best separation (0.569 std dev between classes)
2. `return_mean_60s`: Good separation (0.74 std dev)
3. `return_min_30s`: Good separation (0.64 std dev), downside risk indicator
4. `return_mean_300s`: Moderate separation (0.51 std dev)
5. `tick_count_60s`: Moderate separation (0.21 std dev)

**Expected Correlation (New Feature Set - v1.2):**
- **`realized_volatility_{window}s`**: Expected to be highly correlated (it's the target proxy)
- **`log_return_{window}s`**: Should correlate well with volatility (measures price trends)
- **`price_velocity_{window}s`**: Expected moderate correlation (momentum indicator)
- **`order_book_imbalance_{window}s`**: New feature, correlation to be determined
- **`spread_mean_{window}s`**: Expected moderate correlation (liquidity indicator)
- **`trade_intensity_{window}s`**: Expected moderate correlation (activity measure)

**Interpretation (v1.2):**
- **Realized Volatility** is expected to be the strongest predictor (it's the target proxy)
- **Log Returns** measure price trends over fixed lookback periods
- **Price Velocity** captures momentum and rate of price change
- **Order Book Imbalance** provides microstructure signal about directional pressure
- **Spread and Activity features** contribute to model performance, indicating market conditions
- **Multiple time windows** (30s, 60s, 300s) capture different aspects of market dynamics

**Model Performance by Feature Set:**
- **Previous 10-feature set (v1.1):** XGBoost PR-AUC 0.7815 (best performance)
- **New focused feature set (v1.2):** Performance metrics to be determined after retraining
- **Baseline (v1.1):** Composite z-score approach, PR-AUC 0.2295 (stratified) to 0.2881 (time-based)
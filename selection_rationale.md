# Model Selection Rationale

## Problem Framing
We aim to detect **high-volatility regimes** in BTC-USD tick data in (near) real time based on engineered microstructure features. The model outputs a **probability of high volatility in the next horizon**.

Target label:
- `y = 1` if `volatility_30s` exceeds the 90th percentile threshold τ (estimated from the training set).
- `y = 0` otherwise.

This aligns with the assignment goal of flagging volatility spikes rather than predicting exact returns.

## Candidate Models Considered
1. **Logistic Regression**
   - Pros: simple, interpretable, fast.
   - Cons: underfits complex non-linear relationships in microstructure features.

2. **Random Forest**
   - Pros: robust to noise, non-linear, works well out-of-the-box.
   - Cons: larger ensembles can be slow; less smooth decision boundaries; feature importance can be harder to interpret.

3. **Gradient Boosted Trees (Chosen)**
   - Examples: `GradientBoostingClassifier`, XGBoost-like behavior.
   - Pros:
     - Captures non-linear interactions between features (e.g., volatility + intensity).
     - Strong performance on tabular data.
     - Stable probability estimates.
   - Cons:
     - More sensitive to hyperparameters.
     - Training cost higher than logistic regression, but acceptable for offline training.

## Final Choice
We selected a **Gradient Boosting classifier** trained on:

- **Features:**
  - `midprice`
  - `spread`
  - `trade_intensity`
  - `volatility_30s`

- **Label:**
  - High-vol regime based on τ (90th percentile of `volatility_30s`).

## Why This Model Is Appropriate
- **Tabular, low-dimensional features** → tree-based boosting is a strong default.
- **Non-linearity:** volatility spikes emerge from joint behavior of price movements and trade intensity.
- **Operational Fit:**
  - Inference is fast enough for real-time serving in FastAPI.
  - Model artifact is small and easy to version with MLflow.
  - Works well with Prometheus / Grafana monitoring and rollback strategies.

## Comparison Against Alternatives
- Logistic Regression underperformed on PR-AUC and could not capture interactions between features.
- Random Forest achieved decent accuracy but with:
  - Less stable probability calibration.
  - Slightly worse PR-AUC on rare-event regimes.
- Gradient Boosting reached the best trade-off between:
  - Precision/recall on volatility spikes,
  - Inference latency,
  - Complexity manageable for a course project.

## Next Steps
- Use this GBM as the **"ml" variant** for Week 6 rollback (`MODEL_VARIANT=ml|baseline`).
- Compare against a simple baseline (e.g., logistic regression or moving-average rule).
- Log model runs and metrics in MLflow for reproducibility and comparison.

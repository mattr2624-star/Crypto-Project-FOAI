# Model Evaluation Report – BTC-USD Volatility Spike Detection

**Author:** YOUR NAME  
**Date:** 2025-11-14

---

## 1. Experimental Setup

- **Data:** Coinbase BTC-USD ticker data.
- **Features:** As described in `docs/feature_spec.md`.
- **Target:** Volatility spike within the next 60 seconds.
- **Train/Val/Test split:** Time-based (60/20/20).

---

## 2. Baseline Model

- **Type:** Z-score rule on past volatility (`vol_past_60s`).
- **Metric:** PR-AUC on test set.
- **Result:** (insert value from MLflow)

Interpretation:
- Baseline performance shows how far we can go with a single handcrafted rule.

---

## 3. ML Model (Logistic Regression)

- **Algorithm:** Logistic Regression with standardized numeric features.
- **Features:** All numeric features except label.
- **Metric:** PR-AUC on test set.
- **Result:** (insert value from MLflow)

Interpretation:
- Compare ML PR-AUC vs baseline PR-AUC.
- Discuss whether the gain is meaningful given class imbalance.

---

## 4. Additional Notes

- Check calibration and threshold choice if using F1 or operating-points.
- Consider retraining periodically due to possible drift.

---

## 5. Conclusion

Summarize:
- Whether the ML model **meaningfully outperforms** the baseline.
- Any next steps (e.g., try tree-based models, tune features, or add regularization).


# Team Handoff Package

**Project:** Crypto Volatility Detection  
**Author:** Melissa Wong  
**Date:** November 9, 2025

---

## Package Contents

### Docker & Infrastructure
- `docker/compose.yaml` - Docker Compose configuration (Kafka, Zookeeper, MLflow)
- `docker/Dockerfile.ingestor` - Containerized data ingestion service

### Documentation
- `docs/feature_spec.md` - Feature specification and labeling strategy
- `docs/model_card_v1.md` - Model documentation with performance metrics
- `docs/genai_appendix.md` - GenAI usage disclosure

### Models & Artifacts
- `models/artifacts/` - Trained models (baseline + logistic regression)
  - Model files (`.pkl`)
  - Evaluation plots (PR curves, ROC curves, feature importance)

### Data
- `data/raw/` - 10-minute raw data slice (NDJSON format)
- `data/processed/features_sample.parquet` - Corresponding features for raw data slice

### Reports
- `reports/model_eval.pdf` - Model evaluation report with metrics and comparisons
- `reports/evidently/train_test_drift_report.html` - Data drift analysis (train vs test)

### Predictions
- `predictions.parquet` - Sample predictions on test set

### Dependencies
- `requirements.txt` - Python package dependencies

---

## Model Selection: **Selected-Base**

**Decision:** Use this model as the base for team integration.

**Rationale:**
- Logistic Regression model shows good recall (89.77%) for spike detection
- Model is interpretable and production-ready
- Feature engineering pipeline is well-documented and reproducible
- All infrastructure (Docker, Kafka, MLflow) is containerized and ready

**Model Performance:**
- PR-AUC (Test): 0.0699
- Recall: 89.77% (detects most spikes)
- Precision: 6.66% (produces alerts, requires filtering)

---

## Integration Steps

### 1. Setup Infrastructure
```bash
cd docker
docker compose up -d
```

### 2. Verify Data Pipeline
```bash
# Test feature generation
python features/featurizer.py --topic_in ticks.raw --topic_out ticks.features

# Verify replay consistency
python scripts/replay.py --raw data/raw/*.ndjson --out data/processed/features.parquet
```

### 3. Load Model
```bash
python models/infer.py --model models/artifacts/logistic_regression/model.pkl --features data/processed/features.parquet
```

### 4. View Results
- MLflow UI: http://localhost:5001
- Evidently Report: `reports/evidently/train_test_drift_report.html`

---

## Key Files for Team Integration

1. **Feature Pipeline:** `features/featurizer.py`
2. **Training Script:** `models/train.py`
3. **Inference Script:** `models/infer.py`
4. **Model:** `models/artifacts/logistic_regression/model.pkl`
5. **Feature Spec:** `docs/feature_spec.md`

---

## Notes

- Model uses 5 features (volume_24h_pct_change not available)
- Threshold: 0.000066 (90th percentile)
- Spike rate: ~10% in labeled data
- All metrics logged to MLflow at http://localhost:5001

---

## Contact

For questions about this handoff package, refer to:
- Model Card: `docs/model_card_v1.md`
- Feature Spec: `docs/feature_spec.md`
- GenAI Usage: `docs/genai_appendix.md`


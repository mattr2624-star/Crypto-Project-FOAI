# 🔎 Drift Summary Report

📅 **Report Date:** $(date at runtime)

This document summarizes the latest data drift check using **EvidentlyAI** on the real-time crypto volatility system.

---

## 📊 Overview

We compare:
- **Reference Dataset**: Historical distribution used to train model
- **Current Dataset**: Live (or replayed live) data recently ingested

| Dataset       | Rows | Features |
|--------------|------|----------|
| Reference     | ~6.4K | 3 (`ret_mean`, `ret_std`, `n`) |
| Current       | ~200 | Same features |

---

## 🚨 Drift Status

The full drift visualization is available at:

👉 **`reports/drift_report.html`**

It includes:
- Feature-level drift
- Distribution comparisons
- Statistical test results

---

## 🧠 Model Action Guidance

| Condition | Action |
|-----------|--------|
| Low drift | Continue monitoring |
| Medium drift | Flag for review in next training cycle |
| High drift | 🔁 Retrain candidate model + evaluate rollback risk |

---

## ⚙️ Integration Notes

- This summary is updated via:  
  `docker compose exec model-server python /app/scripts/drift_summary.py`
- Future automation should run on a schedule (e.g., GitHub Actions or Cron)

---

✉️ Contact your team's **Monitoring Owner** if drift persists for more than 24 hours.

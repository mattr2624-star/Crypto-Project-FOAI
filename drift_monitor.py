print("🚀 Drift Monitor container started")

"""
drift_monitor.py

Periodically:
  - Loads reference.csv and current.csv
  - Computes data drift using Evidently
  - Exposes drift metrics via Prometheus on port 9001
  - Saves an HTML drift report into /app/reports
"""

import os
import time
from datetime import datetime

import pandas as pd
from prometheus_client import start_http_server, Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("📦 Imports loaded successfully")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
REPORT_DIR = os.environ.get("REPORT_DIR", "/app/reports")
REFERENCE_PATH = os.path.join(DATA_DIR, "reference.csv")
CURRENT_PATH = os.path.join(DATA_DIR, "current.csv")

CHECK_INTERVAL_SECONDS = int(os.environ.get("DRIFT_CHECK_INTERVAL", "300"))  # Default: 5 minutes
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# PROMETHEUS METRICS
# ---------------------------------------------------------------------
DRIFT_SCORE = Gauge(
    "crypto_data_drift_share",
    "Share of features detected as drifted (0.0–1.0)",
)

DRIFT_DETECTED = Gauge(
    "crypto_data_drift_detected",
    "1 if dataset-level drift is detected, else 0",
)

LAST_DRIFT_TS = Gauge(
    "crypto_data_drift_last_run_ts",
    "Unix timestamp (seconds) of last drift computation",
)

# ---------------------------------------------------------------------
# DRIFT COMPUTATION
# ---------------------------------------------------------------------
def compute_and_log_drift():
    """Load reference/current data, compute Evidently drift, update metrics, write HTML."""
    
    if not os.path.exists(REFERENCE_PATH):
        print(f"⚠ reference.csv not found at {REFERENCE_PATH}")
        return

    if not os.path.exists(CURRENT_PATH):
        print(f"⚠ current.csv not found at {CURRENT_PATH}")
        return

    print(f"📥 Loading reference data from: {REFERENCE_PATH}")
    reference = pd.read_csv(REFERENCE_PATH)

    print(f"📥 Loading current data from:   {CURRENT_PATH}")
    current = pd.read_csv(CURRENT_PATH)

    print("📊 Running Evidently DataDriftPreset...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # Extract summary drift info
    result = report.as_dict()
    drift_result = result["metrics"][0]["result"]

    share_drifted = drift_result.get("share_of_drifted_features", 0.0)
    dataset_drift = drift_result.get("dataset_drift", False)

    # Update Prometheus metrics
    DRIFT_SCORE.set(share_drifted)
    DRIFT_DETECTED.set(1 if dataset_drift else 0)
    LAST_DRIFT_TS.set(time.time())

    print(
        f"✅ Drift check complete | share_drifted={share_drifted:.3f}, "
        f"dataset_drift={dataset_drift}"
    )

    # Save HTML report
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(REPORT_DIR, f"drift_report_{ts}.html")
    report.save_html(html_path)

    print(f"📝 Saved Evidently drift report to: {html_path}")

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
def main():
    port = int(os.environ.get("DRIFT_MONITOR_PORT", "9001"))
    print(f"📡 Starting Prometheus metrics server on port {port} ...")

    start_http_server(port)

    print(f"🔁 Drift monitor loop started (interval={CHECK_INTERVAL_SECONDS}s).")

    while True:
        try:
            compute_and_log_drift()
        except Exception as e:
            print(f"❌ Drift computation error: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()

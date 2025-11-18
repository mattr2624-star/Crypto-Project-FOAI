import os
import time
import pandas as pd
from prometheus_client import start_http_server, Gauge
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Paths
DATA_PATH = "/app/data"
REPORT_PATH = "/app/reports/drift_report.html"

# Prometheus metric
DRIFT_RATIO = Gauge("drift_share_ratio", "Share of features detected as drifted")

# Create drift report
def generate_drift_report():
    try:
        reference_path = os.path.join(DATA_PATH, "reference.csv")
        current_path = os.path.join(DATA_PATH, "current.csv")

        if not (os.path.exists(reference_path) and os.path.exists(current_path)):
            print("⚠️ Missing reference or current data CSV.")
            return 0.0

        ref = pd.read_csv(reference_path)
        cur = pd.read_csv(current_path)

        mapping = ColumnMapping()
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur, column_mapping=mapping)

        report.save_html(REPORT_PATH)
        drift_dict = report.as_dict()
        ratio = drift_dict["metrics"][0]["result"]["share_drifted"]
        DRIFT_RATIO.set(ratio)
        print(f"✅ Drift report generated: {REPORT_PATH}")
        print(f"📊 Drift share ratio = {ratio:.2f}")
        return ratio

    except Exception as e:
        print(f"❌ Error generating drift report: {e}")
        return 0.0

if __name__ == "__main__":
    print("📡 Prometheus metrics server started on port 9001")
    start_http_server(9001)

    INTERVAL = int(os.getenv("DRIFT_INTERVAL", 300))  # 5 minutes
    while True:
        generate_drift_report()
        print(f"✅ Drift monitor run complete, sleeping {INTERVAL//60} minutes...")
        time.sleep(INTERVAL)

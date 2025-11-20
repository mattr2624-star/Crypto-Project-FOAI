from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

DATA_DIR = Path("/app/data")
REPORT_DIR = Path("/app/reports")
DOCS_DIR = Path("/app/docs")

REFERENCE_PATH = DATA_DIR / "reference.csv"
CURRENT_PATH = DATA_DIR / "current.csv"
HTML_REPORT_PATH = REPORT_DIR / "drift_report.html"
MD_SUMMARY_PATH = DOCS_DIR / "drift_summary.md"


if __name__ == "__main__":
    print("📊 Loading data...")
    ref = pd.read_csv(REFERENCE_PATH)
    cur = pd.read_csv(CURRENT_PATH)

    print(f"📌 Loaded reference: {ref.shape}, current: {cur.shape}")

    # We treat all columns as numerical features for this assignment
    num_features = list(ref.columns)

    print("📈 Computing Evidently drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving drift report → {HTML_REPORT_PATH}")
    report.save_html(str(HTML_REPORT_PATH))

    # Simple text summary for docs/drift_summary.md
    summary = report.as_dict()
    drift_share = summary["metrics"][0]["result"]["drift_share"]
    n_drifted = summary["metrics"][0]["result"]["number_of_drifted_columns"]
    n_features = summary["metrics"][0]["result"]["number_of_columns"]

    with MD_SUMMARY_PATH.open("w", encoding="utf-8") as f:
        f.write("# Drift Summary\n\n")
        f.write(f"- Reference rows: {ref.shape[0]}\n")
        f.write(f"- Current rows: {cur.shape[0]}\n")
        f.write(f"- Total features: {n_features}\n")
        f.write(f"- Drifted features: {n_drifted}\n")
        f.write(f"- Drift share: {drift_share:.3f}\n")
        f.write(f"- HTML report: `reports/drift_report.html`\n")

    print("🎉 Drift summary complete!")

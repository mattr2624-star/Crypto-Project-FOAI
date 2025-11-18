import argparse
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading reference: {args.reference}")
    reference = pd.read_parquet(args.reference)

    print(f"[INFO] Loading current:   {args.current}")
    current = pd.read_parquet(args.current)

    print("[INFO] Building Evidently Report (Data Drift)...")
    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(reference_data=reference, current_data=current)

    print(f"[INFO] Saving report: {args.output}")
    report.save_html(args.output)
    print("[DONE] Drift report generated successfully.")

if __name__ == "__main__":
    main()

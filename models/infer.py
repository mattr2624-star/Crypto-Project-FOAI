import argparse
import os

import numpy as np
import pandas as pd
from joblib import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to features parquet file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/artifacts/logreg_model.joblib",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/predictions.parquet",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model file not found: {args.model_path}")

    df = pd.read_parquet(args.features)
    model = load(args.model_path)

    # Use same numeric feature logic as in train.py (except label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" in numeric_cols:
        numeric_cols.remove("label")

    X = df[numeric_cols].values
    proba = model.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out["score"] = proba

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out.to_parquet(args.out, index=False)
    print(f"✅ Saved predictions to {args.out}")


if __name__ == "__main__":
    main()

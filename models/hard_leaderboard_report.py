#!/usr/bin/env python3
"""
Generate a detailed PDF report for the HARD volatility spike models.

Inputs
------
- data/processed/features_hard_train_*.parquet
- data/processed/features_hard_test_*.parquet
- models/artifacts/leaderboard_hard_results.csv

Output
------
- models/artifacts/hard_leaderboard_report_<timestamp>.pdf
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# -------- Paths --------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
RESULTS_CSV = MODEL_DIR / "leaderboard_hard_results.csv"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest(pattern: str) -> Path:
    """Return the latest file matching a glob pattern in DATA_DIR."""
    candidates = sorted(DATA_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files match pattern {pattern} in {DATA_DIR}")
    return candidates[-1]


def load_hard_datasets():
    """Load the latest hard train/test datasets and compute basic stats."""
    train_path = latest("features_hard_train_*.parquet")
    test_path = latest("features_hard_test_*.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    label_col = "volatility_spike_future"
    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(
            f"Label column '{label_col}' missing from hard train/test sets."
        )

    # Feature columns (ignore timestamp + label)
    feature_cols = [
        c
        for c in train_df.columns
        if c not in [label_col, "timestamp"]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    stats = {
        "train_path": train_path.name,
        "test_path": test_path.name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
        "train_spike_rate": float(train_df[label_col].mean()),
        "test_spike_rate": float(test_df[label_col].mean()),
        "train_time_min": str(
            train_df["timestamp"].min() if "timestamp" in train_df.columns else "N/A"
        ),
        "train_time_max": str(
            train_df["timestamp"].max() if "timestamp" in train_df.columns else "N/A"
        ),
        "test_time_min": str(
            test_df["timestamp"].min() if "timestamp" in test_df.columns else "N/A"
        ),
        "test_time_max": str(
            test_df["timestamp"].max() if "timestamp" in test_df.columns else "N/A"
        ),
    }

    return train_df, test_df, feature_cols, stats


def load_leaderboard():
    """Load model leaderboard CSV and select the winner."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Leaderboard file not found: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)

    # Ensure expected columns exist
    required = {"Model", "Accuracy", "AUC", "Precision", "Recall", "F1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Leaderboard CSV is missing required columns: {sorted(missing)}"
        )

    # Sort by F1 (desc), then AUC (desc)
    df_sorted = df.sort_values(by=["F1", "AUC"], ascending=[False, False])
    winner = df_sorted.iloc[0].to_dict()

    return df_sorted, winner


def draw_heading(c, text, y, font_size=16, underline=True):
    c.setFont("Helvetica-Bold", font_size)
    c.drawString(1 * inch, y, text)
    if underline:
        c.setLineWidth(0.5)
        c.line(1 * inch, y - 2, 7.5 * inch, y - 2)
    return y - (font_size + 8)


def draw_paragraph(c, text, y, font_size=10, leading=12):
    """Simple left-aligned paragraph wrapper."""
    c.setFont("Helvetica", font_size)
    max_width = 7.5 * inch
    lines = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ")
        current = ""
        for w in words:
            test = (current + " " + w).strip()
            if c.stringWidth(test, "Helvetica", font_size) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
    for line in lines:
        c.drawString(1 * inch, y, line)
        y -= leading
    return y


def generate_pdf():
    # -------- Load data --------
    train_df, test_df, feature_cols, ds_stats = load_hard_datasets()
    leaderboard_df, winner = load_leaderboard()

    out_pdf = MODEL_DIR / f"hard_leaderboard_report_{timestamp()}.pdf"
    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter

    # -------- TITLE PAGE / HEADER --------
    y = height - 1 * inch
    c.setFont("Helvetica-Bold", 20)
    c.drawString(1 * inch, y, "Crypto Volatility Spike Detection")
    y -= 24
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, y, "Hard Dataset – Model Comparison Report")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 14
    c.drawString(1 * inch, y, f"HARD Train file: {ds_stats['train_path']}")
    y -= 12
    c.drawString(1 * inch, y, f"HARD Test file : {ds_stats['test_path']}")
    y -= 20

    # -------- DATASET SUMMARY --------
    y = draw_heading(c, "1. Dataset Summary (Hard, Time-Based Split)", y)

    summary_lines = [
        f"Train rows          : {ds_stats['n_train']}",
        f"Test rows           : {ds_stats['n_test']}",
        f"Number of features  : {ds_stats['n_features']}",
        f"Train spike rate    : {ds_stats['train_spike_rate']*100:.2f}%",
        f"Test spike rate     : {ds_stats['test_spike_rate']*100:.2f}%",
        f"Train time range    : {ds_stats['train_time_min']} → {ds_stats['train_time_max']}",
        f"Test time range     : {ds_stats['test_time_min']} → {ds_stats['test_time_max']}",
        "",
        "Construction notes:",
        "- Label is a FUTURE volatility spike (proxy computed from future_volatility window).",
        "- Train/test split is strictly time-based (no shuffling) to simulate live deployment.",
        "- All volatility-like and future-looking columns are removed from features to avoid leakage.",
        "- Simple median imputation is used for missing numeric values.",
    ]
    y = draw_paragraph(c, "\n".join(summary_lines), y)

    # New page if needed
    if y < 2 * inch:
        c.showPage()
        y = height - 1 * inch

    # -------- MODELING & EVALUATION SETUP --------
    y = draw_heading(c, "2. Modeling & Evaluation Setup", y)

    eval_text = """
Four models are trained on the HARD TRAIN split and evaluated on the HARD TEST split:

  • Logistic Regression
  • Random Forest
  • Gradient Boosting
  • XGBoost

All models:
  • Use the same engineered numeric feature set (after leakage removal).
  • Are trained only on past information (no access to future volatility).
  • Are evaluated on the hard test set with ~15% spike rate.

Metrics reported:
  • Accuracy – overall correctness (dominated by the majority class).
  • AUC – ranking quality for spike vs non-spike.
  • Precision – fraction of predicted spikes that are real spikes.
  • Recall – fraction of true spikes that the model catches.
  • F1 – harmonic mean of precision and recall (primary selection metric).
"""
    y = draw_paragraph(c, eval_text.strip(), y)

    if y < 2 * inch:
        c.showPage()
        y = height - 1 * inch

    # -------- MODEL COMPARISON TABLE --------
    y = draw_heading(c, "3. Model Comparison (Hard Test Set)", y)

    # Optional threshold column
    has_threshold = "Threshold" in leaderboard_df.columns

    # Table header
    c.setFont("Helvetica-Bold", 9)
    x0 = 1 * inch
    col_model = x0
    col_thr = x0 + 2.5 * inch
    col_acc = x0 + 3.2 * inch
    col_auc = x0 + 3.9 * inch
    col_prec = x0 + 4.6 * inch
    col_rec = x0 + 5.3 * inch
    col_f1 = x0 + 6.0 * inch

    c.drawString(col_model, y, "Model")
    if has_threshold:
        c.drawString(col_thr, y, "Thr")
    c.drawString(col_acc, y, "Acc")
    c.drawString(col_auc, y, "AUC")
    c.drawString(col_prec, y, "Prec")
    c.drawString(col_rec, y, "Rec")
    c.drawString(col_f1, y, "F1")
    y -= 12
    c.setLineWidth(0.3)
    c.line(x0, y, 7.5 * inch, y)
    y -= 10

    # Table rows
    c.setFont("Helvetica", 9)
    for _, row in leaderboard_df.iterrows():
        model_name = row["Model"]
        is_winner = model_name == winner["Model"]

        if is_winner:
            c.setFillColor(colors.darkblue)
            prefix = "★ "
        else:
            c.setFillColor(colors.black)
            prefix = "  "

        c.drawString(col_model, y, prefix + str(model_name)[:28])

        if has_threshold:
            thr_val = row["Threshold"]
            c.drawRightString(col_thr + 20, y, f"{thr_val:.3f}")

        c.drawRightString(col_acc + 25, y, f"{row['Accuracy']:.3f}")
        c.drawRightString(col_auc + 25, y, f"{row['AUC']:.3f}")
        c.drawRightString(col_prec + 25, y, f"{row['Precision']:.3f}")
        c.drawRightString(col_rec + 25, y, f"{row['Recall']:.3f}")
        c.drawRightString(col_f1 + 25, y, f"{row['F1']:.3f}")

        y -= 12
        if y < 1.5 * inch:
            c.setFillColor(colors.black)
            c.showPage()
            y = height - 1 * inch

    c.setFillColor(colors.black)

    if y < 2 * inch:
        c.showPage()
        y = height - 1 * inch

    # -------- WINNER SUMMARY --------
    y = draw_heading(c, "4. Best Performing Model (Winner)", y)

    winner_text = f"""
Winner model (by F1, then AUC):

  • Model name      : {winner['Model']}
  • Accuracy        : {winner['Accuracy']:.3f}
  • AUC             : {winner['AUC']:.3f}
  • Precision       : {winner['Precision']:.3f}
  • Recall          : {winner['Recall']:.3f}
  • F1 score        : {winner['F1']:.3f}
"""
    if "Threshold" in winner:
        winner_text += f"  • Decision threshold: {winner['Threshold']:.3f}\n"

    winner_text += """
Interpretation:

  • The winner is selected to maximize F1, which balances precision and recall.
  • A higher AUC indicates better ranking quality, even if classification
    at a fixed threshold is conservative.
  • In this hard, time-based setup with rare spikes, even modest recall
    is challenging — this reflects a more realistic production scenario
    compared to the earlier, easier combined dataset.
"""

    y = draw_paragraph(c, winner_text.strip(), y)

    # Finalize
    c.showPage()
    c.save()
    print(f"\n📄 PDF report saved → {out_pdf}\n")


if __name__ == "__main__":
    generate_pdf()

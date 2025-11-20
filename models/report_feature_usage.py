#!/usr/bin/env python3
"""
Generate a detailed PDF report for HARD model comparison.

Includes:
- Dataset stats (train + test)
- Leaderboard metrics (from leaderboard_hard_results.csv)
- Winner selection by AUC
- The feature set used to train all HARD models

Inputs:
    data/processed/features_hard_train_*.parquet
    data/processed/features_hard_test_*.parquet
    models/artifacts/leaderboard_hard_results.csv
    models/artifacts/*_hard_*.joblib

Output:
    models/hard_model_report_<timestamp>.pdf
"""

import glob
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# -------------------------------------------------------------------
# Paths / config
# -------------------------------------------------------------------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
REPORT_DIR = Path("models")
REPORT_DIR.mkdir(exist_ok=True)

LEADERBOARD_CSV = MODEL_DIR / "leaderboard_hard_results.csv"

WINNER_METRIC = "AUC"  # per your choice: AUC is the main selection metric


def latest(pattern: str, base_dir: Path) -> Path:
    """Return the latest file matching pattern in base_dir."""
    candidates = sorted(base_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files match pattern {pattern} in {base_dir}")
    return candidates[-1]


def load_hard_datasets():
    """Load latest HARD TRAIN and TEST parquet files."""
    train_path = latest("features_hard_train_*.parquet", DATA_DIR)
    test_path = latest("features_hard_test_*.parquet", DATA_DIR)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df, train_path, test_path


def load_leaderboard():
    """Load leaderboard CSV and sort by AUC (descending)."""
    if not LEADERBOARD_CSV.exists():
        raise FileNotFoundError(f"Leaderboard CSV not found: {LEADERBOARD_CSV}")

    df = pd.read_csv(LEADERBOARD_CSV)
    # Ensure expected columns exist
    required_cols = {"Model", "Accuracy", "AUC", "Precision", "Recall", "F1"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Leaderboard is missing columns: {missing}")

    df_sorted = df.sort_values(by=[WINNER_METRIC, "F1"], ascending=False).reset_index(drop=True)
    return df_sorted


def load_feature_names_for_model(model_name: str, model_path: str):
    """
    Load one of your *_hard_* model bundles and extract feature names.

    We assume your training code saved:
        {"model": estimator, "imputer": imputer, "feature_names": list}
    """
    bundle = joblib.load(model_path)

    feature_names = None

    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        feature_names = bundle.get("feature_names", None)
    else:
        model = bundle

    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    return model, feature_names


def build_pdf_story():
    """Assemble the PDF story (list of Flowables)."""
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    heading3_style = styles["Heading3"]
    body_style = styles["BodyText"]

    # Slightly tighter, neutral body style
    body_style.leading = 13

    story = []

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ----------------------------------------------------------------
    # 1. Title
    # ----------------------------------------------------------------
    story.append(Paragraph("Hard Model Comparison Report", title_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"Generated: {now_str}", body_style))
    story.append(Spacer(1, 0.25 * inch))

    # ----------------------------------------------------------------
    # 2. Load data + leaderboard
    # ----------------------------------------------------------------
    train_df, test_df, train_path, test_path = load_hard_datasets()
    leaderboard = load_leaderboard()

    # Dataset stats
    label_col = "volatility_spike_future"
    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' missing from HARD datasets.")

    train_y = train_df[label_col].astype(int)
    test_y = test_df[label_col].astype(int)

    train_spike_rate = train_y.mean()
    test_spike_rate = test_y.mean()

    n_features_train = train_df.drop(columns=[label_col, "timestamp"], errors="ignore").shape[1]
    n_features_test = test_df.drop(columns=[label_col, "timestamp"], errors="ignore").shape[1]

    # ----------------------------------------------------------------
    # 3. Dataset summary section
    # ----------------------------------------------------------------
    story.append(Paragraph("1. Hard Dataset Overview", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    ds_text = f"""
    The hard dataset is constructed using a realistic time-based split with a
    future volatility proxy label (<b>{label_col}</b>). Training and test
    windows are separated chronologically to mimic production deployment.
    """
    story.append(Paragraph(ds_text, body_style))
    story.append(Spacer(1, 0.1 * inch))

    ds_table_data = [
        ["Split", "Rows", "Features (numeric)", "Spike rate"],
        [
            "TRAIN",
            f"{len(train_df):,}",
            f"{n_features_train}",
            f"{train_spike_rate*100:.2f}%",
        ],
        [
            "TEST",
            f"{len(test_df):,}",
            f"{n_features_test}",
            f"{test_spike_rate*100:.2f}%",
        ],
    ]

    ds_table = Table(ds_table_data, hAlign="LEFT")
    ds_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(ds_table)
    story.append(Spacer(1, 0.25 * inch))

    # ----------------------------------------------------------------
    # 4. Leaderboard section
    # ----------------------------------------------------------------
    story.append(Paragraph("2. Hard Model Leaderboard (AUC as Tie-Breaker)", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    lb_intro = """
    All models below were evaluated on the same HARD test set. The official
    winner is selected using AUC (Area Under the ROC Curve) as the primary
    metric, with F1 used as a secondary tiebreaker.
    """
    story.append(Paragraph(lb_intro, body_style))
    story.append(Spacer(1, 0.1 * inch))

    # Build leaderboard table
    lb_cols = ["Model", "Threshold", "Accuracy", "AUC", "Precision", "Recall", "F1"]
    for c in lb_cols:
        if c not in leaderboard.columns:
            # gracefully handle older leaderboard versions
            leaderboard[c] = float("nan")

    table_data = [lb_cols]
    for _, row in leaderboard.iterrows():
        table_data.append(
            [
                str(row["Model"]),
                f"{row.get('Threshold', float('nan')):.3f}" if pd.notna(row.get("Threshold", float("nan"))) else "–",
                f"{row['Accuracy']:.4f}" if pd.notna(row["Accuracy"]) else "–",
                f"{row['AUC']:.4f}" if pd.notna(row["AUC"]) else "–",
                f"{row['Precision']:.4f}" if pd.notna(row["Precision"]) else "–",
                f"{row['Recall']:.4f}" if pd.notna(row["Recall"]) else "–",
                f"{row['F1']:.4f}" if pd.notna(row["F1"]) else "–",
            ]
        )

    lb_table = Table(table_data, hAlign="LEFT")
    lb_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(lb_table)
    story.append(Spacer(1, 0.25 * inch))

    # ----------------------------------------------------------------
    # 5. Winner description (by AUC)
    # ----------------------------------------------------------------
    best_row = leaderboard.iloc[0]
    best_model_name = str(best_row["Model"])
    best_auc = best_row["AUC"]
    best_f1 = best_row["F1"]
    best_acc = best_row["Accuracy"]

    story.append(Paragraph("3. Selected Winner (AUC-Based)", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    winner_text = f"""
    Based on the HARD test set, the top-performing model by <b>AUC</b> is
    <b>{best_model_name}</b>. It achieves:
    <br/>
    • AUC = <b>{best_auc:.4f}</b><br/>
    • F1  = <b>{best_f1:.4f}</b><br/>
    • Accuracy = <b>{best_acc:.4f}</b><br/>
    <br/>
    This model is selected as the final winner because AUC captures the
    model's ability to rank volatility spike risk across all thresholds, which
    is especially important for imbalanced, event-driven problems like crypto
    spike prediction.
    """
    story.append(Paragraph(winner_text, body_style))
    story.append(Spacer(1, 0.25 * inch))

    # ----------------------------------------------------------------
    # 6. Features used by the HARD models
    # ----------------------------------------------------------------
    story.append(Paragraph("4. Features Used by HARD Models", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    # We’ll use the best model’s artifact path from the leaderboard
    # The leaderboard stores a "Path" column with the joblib path
    if "Path" in leaderboard.columns and isinstance(best_row["Path"], str):
        best_model_path = best_row["Path"]
        # If path is relative, resolve from MODEL_DIR
        if not os.path.isabs(best_model_path):
            best_model_path = str((MODEL_DIR / best_model_path).resolve())
    else:
        # Fallback: search for *_hard_*.joblib with matching stem
        candidate = list(MODEL_DIR.glob(f"{best_model_name}.joblib"))
        if not candidate:
            raise FileNotFoundError(
                f"Could not resolve model path for winner '{best_model_name}'."
            )
        best_model_path = str(candidate[0])

    _, feature_names = load_feature_names_for_model(best_model_name, best_model_path)

    if feature_names is None:
        feature_intro = """
        The hard models were trained on a numeric feature matrix derived from
        combined exchange ticks, but this environment does not expose a saved
        feature list. In your local environment, the joblib bundles include
        the exact feature_name array used during training.
        """
        story.append(Paragraph(feature_intro, body_style))
    else:
        feature_intro = f"""
        All four HARD models (Logistic Regression, Random Forest, Gradient
        Boosting, XGBoost) were trained on the same engineered feature set
        derived from order book levels, short–/medium–horizon returns, and
        activity statistics. The winner <b>{best_model_name}</b> was trained
        on <b>{len(feature_names)}</b> numeric features. The list below shows
        the exact feature names passed into the model.
        """
        story.append(Paragraph(feature_intro, body_style))
        story.append(Spacer(1, 0.1 * inch))

        # Print features in a multi-column table (for readability)
        # We’ll use 3 columns if possible
        cols = 3
        rows = []
        current_row = []
        for i, feat in enumerate(feature_names, start=1):
            current_row.append(feat)
            if i % cols == 0:
                rows.append(current_row)
                current_row = []
        if current_row:
            # pad the last row
            while len(current_row) < cols:
                current_row.append("")
            rows.append(current_row)

        # Add header row
        table_data_feats = [["Feature 1", "Feature 2", "Feature 3"]] + rows

        feat_table = Table(table_data_feats, hAlign="LEFT", colWidths=[2.0 * inch] * cols)
        feat_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ]
            )
        )
        story.append(feat_table)

    story.append(Spacer(1, 0.25 * inch))

    # Closing
    story.append(Paragraph("End of report.", body_style))

    return story


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORT_DIR / f"hard_model_report_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Hard Model Comparison Report",
    )

    story = build_pdf_story()
    doc.build(story)

    print(f"\n✅ PDF report generated → {out_path}\n")


if __name__ == "__main__":
    main()

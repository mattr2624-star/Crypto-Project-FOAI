#!/usr/bin/env python3
"""
Leaderboard Evaluation for All Saved Models

✔ Loads ALL versioned model files (.joblib)
✔ Evaluates each on the combined dataset
✔ Computes:
    - Accuracy
    - AUC
    - Precision, Recall, F1
    - Confusion Matrix
✔ Ranks models by AUC (best first)
✔ Saves:
    - leaderboard.csv
    - leaderboard.pdf
    - ROC comparison plot

Requires: pandas, sklearn, matplotlib, seaborn, joblib, reportlab
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ---------- CONFIG ----------
DATA_PATH = "data/processed/features_combined_labeled.parquet"
MODEL_DIR = "models/artifacts"
OUT_CSV = "models/leaderboard_results.csv"
OUT_PDF = "models/leaderboard_report.pdf"
ROC_IMG = "models/roc_comparison.png"

# ---------- Load Test Data ----------
print("📌 Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df = df.select_dtypes(include=["float", "int"])
X = df.drop(columns=["volatility_spike"])
y = df["volatility_spike"]

print(f"   ✔ Dataset loaded: {len(X)} rows")

# ---------- Evaluate Model Function ----------
def evaluate_model(model, name):
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y, preds),
        "AUC": roc_auc_score(y, probas),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1": f1_score(y, preds),
        "ConfusionMatrix": confusion_matrix(y, preds),
        "ROC": roc_curve(y, probas)
    }

# ---------- Load & Evaluate All Models ----------
print("\n🔎 Evaluating models...\n")

results = []
roc_curves = []

for file in os.listdir(MODEL_DIR):
    if file.endswith(".joblib"):
        path = os.path.join(MODEL_DIR, file)
        model = joblib.load(path)
        name = file.replace(".joblib", "")

        try:
            r = evaluate_model(model, name)
            results.append(r)
            roc_curves.append((name, r["ROC"]))
            print(f"   ✔ {name} evaluated.")
        except Exception as e:
            print(f"   ⚠ Failed to evaluate {name}: {e}")

# ---------- Create Leaderboard ----------
df_leaderboard = pd.DataFrame([{
    "Model": r["Model"],
    "Accuracy": r["Accuracy"],
    "AUC": r["AUC"],
    "Precision": r["Precision"],
    "Recall": r["Recall"],
    "F1": r["F1"]
} for r in results])

df_leaderboard.sort_values("AUC", ascending=False, inplace=True)
df_leaderboard.to_csv(OUT_CSV, index=False)

print(f"\n🏆 Leaderboard saved ➜ {OUT_CSV}")
print(df_leaderboard)

# ---------- Plot ROC Comparison ----------
plt.figure(figsize=(10, 7))
for name, (fpr, tpr, _) in roc_curves:
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
plt.title("ROC Curve Comparison", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(ROC_IMG)
plt.close()
print(f"📈 ROC curve saved ➜ {ROC_IMG}")

# ---------- Generate PDF Report ----------
def generate_pdf():
    c = canvas.Canvas(OUT_PDF, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, height - 50, "📊 Model Leaderboard Report")

    c.setFont("Helvetica", 12)
    y_pos = height - 80

    for _, row in df_leaderboard.iterrows():
        line = f"{row['Model']}  |  AUC={row['AUC']:.4f}  |  F1={row['F1']:.4f}"
        c.drawString(30, y_pos, line)
        y_pos -= 16

    c.drawImage(ROC_IMG, 30, 50, width-60, 300)
    c.save()

generate_pdf()
print(f"\n📄 PDF Report saved ➜ {OUT_PDF}")

print("\n🎉 DONE! Leaderboard complete.\n")

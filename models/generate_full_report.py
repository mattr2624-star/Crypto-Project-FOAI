#!/usr/bin/env python3
"""
Generate a PDF report with:
- Dataset statistics
- Feature summary
- Training/testing distribution
- Leaderboard performance

Uses:
    models/leaderboard_results.csv
    data/processed/features_combined_labeled.parquet

Creates:
    models/final_report.pdf
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

# ================================
#  CONFIG
# ================================
DATA_PATH       = "data/processed/features_combined_labeled.parquet"
LEADERBOARD_CSV = "models/leaderboard_results.csv"
REPORT_PDF      = "models/final_report.pdf"

# ================================
#  LOAD DATA
# ================================
df = pd.read_parquet(DATA_PATH)
leader = pd.read_csv(LEADERBOARD_CSV)

# ================================
#  HELPERS
# ================================
def add_title(c, text, y):
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, text)
    c.setFont("Helvetica", 12)

def add_subtitle(c, text, y):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, text)
    c.setFont("Helvetica", 11)

def draw_table(c, data, x, y):
    table = Table(data, repeatRows=1)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
        ('TEXTCOLOR',   (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.lightgrey),
        ('GRID',         (0,0), (-1,-1), 0.25, colors.black),
    ])
    table.setStyle(style)
    w, h = table.wrapOn(c, 40, 40)
    table.drawOn(c, x, y - h)
    return h

# ================================
#  PLOT & SAVE GRAPHICS
# ================================

# Training/testing distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df["volatility_spike"])
plt.title("Label Distribution (Spike=1)")
plt.savefig("models/tmp_label_dist.png", dpi=150)
plt.close()

# Feature importance heatmap placeholder (for CNN if added)
plt.figure(figsize=(6,4))
sns.heatmap(df.select_dtypes(include=["float","int"]).corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig("models/tmp_corr_heatmap.png", dpi=150)
plt.close()

# ================================
#  CREATE PDF
# ================================
c = canvas.Canvas(REPORT_PDF, pagesize=A4)
width, height = A4

# #################################
#  PAGE 1 — Overview
# #################################
add_title(c, "Crypto Volatility Detection — Final Report", height - 50)
add_subtitle(c, "📊 Dataset Summary", height - 90)

c.drawString(60, height - 120, f"Total Rows: {len(df):,}")
c.drawString(60, height - 140, f"Total Features: {df.shape[1]}")
c.drawString(60, height - 160, f"Spikes (1): {(df['volatility_spike']==1).sum():,}")
c.drawString(60, height - 180, f"Normal (0): {(df['volatility_spike']==0).sum():,}")
c.drawString(60, height - 200, f"Spike Rate: {(df['volatility_spike'].mean()*100):.2f}%")

c.drawImage("models/tmp_label_dist.png", 300, height - 260, width=200, height=140)

c.showPage()

# #################################
#  PAGE 2 — Leaderboard Table
# #################################
add_title(c, "🏆 Model Leaderboard", height - 50)

# Prepare table
leader_rounded = leader.copy()
leader_rounded["Accuracy"] = leader_rounded["Accuracy"].round(4)
leader_rounded["AUC"]      = leader_rounded["AUC"].round(4)
leader_rounded["F1"]       = leader_rounded["F1"].round(4)

data = [leader_rounded.columns.tolist()] + leader_rounded.values.tolist()
_ = draw_table(c, data, 40, height - 120)

c.showPage()

# #################################
#  PAGE 3 — Correlation Heatmap
# #################################
add_title(c, "📌 Feature Correlation Heatmap", height - 50)
c.drawImage("models/tmp_corr_heatmap.png", 60, height - 500, width=480, height=420)

c.showPage()

# #################################
#  PAGE 4 — Best Model Summary
# #################################
best = leader.iloc[0]
add_title(c, "🥇 Best Model Summary", height - 50)

c.drawString(60, height - 90, f"Best Model: {best['Model']}")
c.drawString(60, height - 110, f"Accuracy:   {best['Accuracy']:.4f}")
c.drawString(60, height - 130, f"AUC:        {best['AUC']:.4f}")
c.drawString(60, height - 150, f"Precision:  {best['Precision']:.4f}")
c.drawString(60, height - 170, f"Recall:     {best['Recall']:.4f}")
c.drawString(60, height - 190, f"F1 Score:   {best['F1']:.4f}")

c.showPage()

# #################################
#  SAVE PDF
# #################################
c.save()

print(f"\n📄 FINAL REPORT SAVED ➜ {REPORT_PDF}\n")

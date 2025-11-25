#!/usr/bin/env python3
"""
Train and Compare Models for Real-Time Prediction

This script trains multiple models on the same dataset and evaluates them
to determine which will perform best on new, unseen real-time data.

Usage:
    python scripts/train_and_compare.py --features data/processed/features_sample.parquet
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def prepare_data(df: pd.DataFrame, target_col: str = "volatility_spike"):
    """Prepare features and target from dataframe."""
    # Identify numeric feature columns
    exclude_cols = [target_col, 'timestamp', 'ts', 'time', 'label', 'future_volatility',
                    'price', 'best_bid', 'best_ask', 'midprice']
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].values
    
    return X, y, feature_cols


def train_model(model, X_train, y_train, X_test, y_test, model_name: str) -> Dict:
    """Train a model and evaluate it."""
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Find optimal threshold using F1
    best_f1 = 0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_temp = (y_prob >= thresh).astype(int)
        f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = thresh
    
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "threshold": best_threshold,
        "n_samples": len(y_test),
        "n_positive": int(y_test.sum()),
        "n_predicted_positive": int(y_pred.sum()),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
        "pr_auc": average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
        "train_time_s": train_time,
        "inference_time_ms": inference_time,
        "inference_per_sample_ms": inference_time / len(y_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return metrics, model


def print_results(results: List[Dict]):
    """Print comparison results."""
    print("\n" + "=" * 100)
    print(" MODEL COMPARISON RESULTS (Trained on Same Data)")
    print("=" * 100)
    
    # Header
    print(f"\n{'Model':<25} {'PR-AUC':>10} {'ROC-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Latency':>12}")
    print("-" * 100)
    
    # Sort by PR-AUC
    sorted_results = sorted(results, key=lambda x: x['pr_auc'], reverse=True)
    
    for r in sorted_results:
        latency_str = f"{r['inference_per_sample_ms']:.4f} ms"
        print(f"{r['model_name']:<25} {r['pr_auc']:>10.4f} {r['roc_auc']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {latency_str:>12}")
    
    print("-" * 100)
    
    winner = sorted_results[0]
    print(f"\n*** BEST MODEL FOR REAL-TIME: {winner['model_name']} ***")
    print(f"    PR-AUC: {winner['pr_auc']:.4f} (best for imbalanced data)")
    print(f"    F1: {winner['f1']:.4f} | Precision: {winner['precision']:.4f} | Recall: {winner['recall']:.4f}")
    print(f"    Inference: {winner['inference_per_sample_ms']:.4f} ms/sample")
    print(f"    Optimal Threshold: {winner['threshold']:.2f}")
    
    return winner


def main():
    parser = argparse.ArgumentParser(description="Train and compare models")
    parser.add_argument("--features", type=str, default="data/processed/features_sample.parquet",
                        help="Path to features file")
    parser.add_argument("--target", type=str, default="volatility_spike", help="Target column")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test set fraction")
    parser.add_argument("--save-best", action="store_true", help="Save the best model")
    parser.add_argument("--output", type=str, default="docs/model_comparison_results.json")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 100)
    print(" TRAIN AND COMPARE MODELS FOR REAL-TIME PREDICTION")
    print("=" * 100)
    
    # Load data
    print(f"\n[*] Loading data from: {args.features}")
    
    if not os.path.exists(args.features):
        print(f"[ERROR] File not found: {args.features}")
        sys.exit(1)
    
    df = pd.read_parquet(args.features)
    print(f"    Total samples: {len(df)}")
    
    # Prepare data
    X, y, feature_names = prepare_data(df, args.target)
    print(f"    Features: {len(feature_names)}")
    print(f"    Positive class rate: {y.mean()*100:.2f}%")
    
    # Split data (time-based: use last portion as test)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Test positive rate: {y_test.mean()*100:.2f}%")
    
    # Define models to compare
    models = [
        ("Logistic Regression", Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
        )),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )),
        ("Random Forest (Deep)", RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1
        )),
    ]
    
    # Add simple baseline
    class SimpleBaseline:
        """Z-score based baseline detector."""
        def __init__(self, threshold_percentile=90):
            self.threshold_percentile = threshold_percentile
            self.threshold = None
            
        def fit(self, X, y):
            # Use first numeric column as signal
            signal = X.iloc[:, 0].values
            self.threshold = np.percentile(np.abs(signal), self.threshold_percentile)
            return self
            
        def predict(self, X):
            signal = X.iloc[:, 0].values
            return (np.abs(signal) > self.threshold).astype(int)
            
        def predict_proba(self, X):
            signal = X.iloc[:, 0].values
            scores = np.abs(signal) / (self.threshold + 1e-6)
            scores = np.clip(scores, 0, 1)
            return np.column_stack([1 - scores, scores])
    
    models.insert(0, ("Z-Score Baseline", SimpleBaseline()))
    
    # Train and evaluate each model
    print("\n[*] Training and evaluating models...")
    results = []
    trained_models = {}
    
    for name, model in models:
        print(f"\n    Training: {name}")
        try:
            metrics, trained_model = train_model(model, X_train, y_train, X_test, y_test, name)
            results.append(metrics)
            trained_models[name] = trained_model
            print(f"    [OK] PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"    [ERROR] {e}")
    
    # Print results
    winner = print_results(results)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "features_file": args.features,
            "n_samples": len(df),
            "n_features": len(feature_names),
            "test_size": args.test_size,
            "results": results,
            "recommendation": winner['model_name']
        }, f, indent=2)
    print(f"\n[*] Results saved to: {args.output}")
    
    # Save best model if requested
    if args.save_best:
        best_model = trained_models[winner['model_name']]
        output_dir = Path("models/artifacts") / winner['model_name'].lower().replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": best_model,
            "feature_names": feature_names,
            "threshold": winner['threshold'],
            "metrics": winner,
            "trained_at": datetime.now().isoformat()
        }
        
        with open(output_dir / "model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[*] Best model saved to: {output_dir / 'model.pkl'}")
    
    # Recommendation
    print("\n" + "=" * 100)
    print(" RECOMMENDATION FOR REAL-TIME DEPLOYMENT")
    print("=" * 100)
    print(f"""
  Best Model: {winner['model_name']}
  
  Performance on Holdout Data:
  - PR-AUC: {winner['pr_auc']:.4f} (primary metric for imbalanced data)
  - Recall: {winner['recall']:.4f} (catches {winner['recall']*100:.1f}% of volatility spikes)
  - Precision: {winner['precision']:.4f} ({winner['precision']*100:.1f}% of alerts are true spikes)
  - Inference: {winner['inference_per_sample_ms']:.4f} ms/sample (real-time capable)
  
  To deploy, update MODEL_VERSION in docker-compose.yaml or run:
    $env:MODEL_VERSION="{winner['model_name'].lower().replace(' ', '_')}"
    docker compose up -d api
""")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()


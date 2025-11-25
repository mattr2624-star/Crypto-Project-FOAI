#!/usr/bin/env python3
"""
Model Comparison Script for Real-Time Crypto Volatility Prediction

This script evaluates multiple models on the same holdout dataset to determine
which model will perform best on new, unseen real-time data.

Usage:
    python scripts/compare_models.py --features data/processed/features_consolidated_test.parquet
    python scripts/compare_models.py --features data/processed/features_consolidated.parquet --holdout-pct 0.2
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_path: str) -> Tuple[Any, dict]:
    """Load a model and its metadata."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        model = model_data.get("model", model_data.get("classifier"))
        metadata = {
            "feature_names": model_data.get("feature_names", model_data.get("features", [])),
            "threshold": model_data.get("threshold", 0.5),
            "model_type": model_data.get("model_type", "unknown"),
        }
    else:
        model = model_data
        metadata = {"feature_names": [], "threshold": 0.5, "model_type": "unknown"}
    
    return model, metadata


def prepare_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Prepare features for prediction, handling missing columns."""
    # Get available features
    available = [f for f in feature_names if f in df.columns]
    missing = [f for f in feature_names if f not in df.columns]
    
    if missing:
        print(f"  Warning: Missing {len(missing)} features")
    
    # Use available features or fall back to numeric columns
    if available:
        X = df[available].copy()
    else:
        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['volatility_spike', 'timestamp', 'ts']
        X = df[[c for c in numeric_cols if c not in exclude]].copy()
    
    # Fill NaN values
    X = X.fillna(0)
    
    return X


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "model"
) -> Dict:
    """Evaluate a model and return metrics."""
    
    # Time predictions
    start_time = time.time()
    
    # Get predictions
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        y_prob = y_pred.astype(float)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "n_predicted_positive": int(y_pred.sum()),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0,
        "pr_auc": average_precision_score(y, y_prob) if len(np.unique(y)) > 1 else 0,
        "inference_time_ms": inference_time,
        "inference_per_sample_ms": inference_time / len(y),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    
    return metrics


def find_models(models_dir: str = "models/artifacts") -> List[Tuple[str, str]]:
    """Find all available models in the artifacts directory."""
    models = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return models
    
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "model.pkl"
            if model_file.exists():
                models.append((model_dir.name, str(model_file)))
    
    return models


def print_comparison_table(results: List[Dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print(" MODEL COMPARISON RESULTS")
    print("=" * 100)
    
    # Header
    print(f"\n{'Model':<20} {'PR-AUC':>10} {'ROC-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Latency':>12}")
    print("-" * 100)
    
    # Sort by PR-AUC (best metric for imbalanced data)
    sorted_results = sorted(results, key=lambda x: x['pr_auc'], reverse=True)
    
    for r in sorted_results:
        latency_str = f"{r['inference_per_sample_ms']:.3f} ms"
        print(f"{r['model_name']:<20} {r['pr_auc']:>10.4f} {r['roc_auc']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {latency_str:>12}")
    
    print("-" * 100)
    
    # Winner
    winner = sorted_results[0]
    print(f"\n*** RECOMMENDED MODEL: {winner['model_name']} ***")
    print(f"    PR-AUC: {winner['pr_auc']:.4f} | F1: {winner['f1']:.4f} | Recall: {winner['recall']:.4f}")
    print(f"    Latency: {winner['inference_per_sample_ms']:.4f} ms per sample")
    
    return winner


def print_detailed_results(results: List[Dict]):
    """Print detailed results for each model."""
    print("\n" + "=" * 100)
    print(" DETAILED MODEL ANALYSIS")
    print("=" * 100)
    
    for r in results:
        print(f"\n{'-' * 50}")
        print(f"[MODEL] {r['model_name'].upper()}")
        print(f"{'-' * 50}")
        
        print(f"\n  Performance Metrics:")
        print(f"    PR-AUC (primary):     {r['pr_auc']:.4f}")
        print(f"    ROC-AUC:              {r['roc_auc']:.4f}")
        print(f"    F1 Score:             {r['f1']:.4f}")
        print(f"    Precision:            {r['precision']:.4f}")
        print(f"    Recall:               {r['recall']:.4f}")
        print(f"    Accuracy:             {r['accuracy']:.4f}")
        
        print(f"\n  Prediction Stats:")
        print(f"    Total samples:        {r['n_samples']}")
        print(f"    Actual positives:     {r['n_positive']} ({r['n_positive']/r['n_samples']*100:.2f}%)")
        print(f"    Predicted positives:  {r['n_predicted_positive']} ({r['n_predicted_positive']/r['n_samples']*100:.2f}%)")
        print(f"    Threshold:            {r['threshold']:.4f}")
        
        print(f"\n  Latency:")
        print(f"    Total inference:      {r['inference_time_ms']:.2f} ms")
        print(f"    Per sample:           {r['inference_per_sample_ms']:.4f} ms")
        
        cm = np.array(r['confusion_matrix'])
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Neg    Pos")
        print(f"    Actual Neg   {cm[0,0]:>5}  {cm[0,1]:>5}")
        print(f"    Actual Pos   {cm[1,0]:>5}  {cm[1,1]:>5}")


def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "models": results,
        "recommendation": max(results, key=lambda x: x['pr_auc'])['model_name']
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[*] Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple models on the same dataset for real-time prediction"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features_consolidated_test.parquet",
        help="Path to features file (use test set for unbiased evaluation)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/artifacts",
        help="Directory containing model artifacts"
    )
    parser.add_argument(
        "--holdout-pct",
        type=float,
        default=0.0,
        help="If > 0, create holdout set from features (use last N%% of data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/model_comparison_results.json",
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="volatility_spike",
        help="Target column name"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 100)
    print(" CRYPTO VOLATILITY MODEL COMPARISON")
    print(" Evaluating models for real-time prediction accuracy")
    print("=" * 100)
    
    # Load features
    print(f"\n[*] Loading features from: {args.features}")
    
    if not os.path.exists(args.features):
        print(f"\n[ERROR] Features file not found: {args.features}")
        print("Available files in data/processed/:")
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            for f in processed_dir.glob("*.parquet"):
                print(f"  - {f.name}")
        sys.exit(1)
    
    df = pd.read_parquet(args.features)
    print(f"    Total samples: {len(df)}")
    
    # Create holdout if requested
    if args.holdout_pct > 0:
        holdout_size = int(len(df) * args.holdout_pct)
        df = df.iloc[-holdout_size:]  # Use last N% as holdout
        print(f"    Using last {args.holdout_pct*100:.0f}% as holdout: {len(df)} samples")
    
    # Check target column
    if args.target not in df.columns:
        print(f"\n[ERROR] Target column '{args.target}' not found in features")
        print(f"    Available columns: {df.columns.tolist()[:10]}...")
        sys.exit(1)
    
    y = df[args.target].values
    print(f"    Positive class rate: {y.mean()*100:.2f}%")
    
    # Find models
    print(f"\n[*] Searching for models in: {args.models_dir}")
    models = find_models(args.models_dir)
    
    if not models:
        print("[ERROR] No models found!")
        sys.exit(1)
    
    print(f"    Found {len(models)} models: {[m[0] for m in models]}")
    
    # Evaluate each model
    results = []
    
    for model_name, model_path in models:
        print(f"\n[*] Evaluating: {model_name}")
        
        try:
            # Load model
            model, metadata = load_model(model_path)
            
            # Prepare features
            X = prepare_features(df, metadata.get("feature_names", []))
            print(f"    Features used: {X.shape[1]}")
            
            # Get threshold
            threshold = metadata.get("threshold", 0.5)
            
            # Try to load threshold from metadata file
            threshold_file = Path(model_path).parent / "threshold_metadata.json"
            if threshold_file.exists():
                with open(threshold_file) as f:
                    threshold_data = json.load(f)
                    threshold = threshold_data.get("optimal_f1_threshold", threshold)
                    print(f"    Loaded threshold: {threshold:.4f}")
            
            # Evaluate
            metrics = evaluate_model(model, X, y, threshold, model_name)
            results.append(metrics)
            
            print(f"    [OK] PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
            
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue
    
    if not results:
        print("\n[ERROR] No models could be evaluated!")
        sys.exit(1)
    
    # Print comparison
    winner = print_comparison_table(results)
    print_detailed_results(results)
    
    # Save results
    save_results(results, args.output)
    
    # Final recommendation
    print("\n" + "=" * 100)
    print(" RECOMMENDATION FOR REAL-TIME DEPLOYMENT")
    print("=" * 100)
    print(f"""
  Based on evaluation on unseen data:
  
  *** Deploy: {winner['model_name']} ***
  
  Why this model?
  - Highest PR-AUC ({winner['pr_auc']:.4f}) - best for imbalanced data
  - Good balance of precision ({winner['precision']:.4f}) and recall ({winner['recall']:.4f})
  - Fast inference ({winner['inference_per_sample_ms']:.4f} ms/sample) - suitable for real-time
  
  To deploy this model:
  
    # Update docker-compose environment:
    MODEL_VARIANT=ml
    MODEL_VERSION={winner['model_name']}
    MODEL_PATH=/app/models/artifacts/{winner['model_name']}/model.pkl
    
    # Or restart with specific model:
    $env:MODEL_VERSION="{winner['model_name']}"
    docker compose up -d api
""")
    
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

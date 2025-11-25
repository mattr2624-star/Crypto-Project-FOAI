#!/usr/bin/env python3
"""
Train and Compare Models for Real-Time Prediction with GridSearchCV

This script trains multiple models using GridSearchCV for hyperparameter optimization,
utilizing all available CPU cores for parallel training.

Usage:
    python scripts/train_and_compare.py --features data/processed/features_sample.parquet
    python scripts/train_and_compare.py --features data/processed/features_sample.parquet --grid-search
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    make_scorer,
)

# Try to import psutil for hardware info
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed. Install with: pip install psutil")


def get_hardware_info() -> Dict:
    """Get system hardware information."""
    info = {
        "cpu_count": os.cpu_count(),
        "cpu_count_physical": os.cpu_count(),
    }

    if HAS_PSUTIL:
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024**3), 2)
        info["memory_available_gb"] = round(mem.available / (1024**3), 2)
        info["memory_percent"] = mem.percent

    return info


def print_hardware_info():
    """Print hardware info for training."""
    info = get_hardware_info()
    print("\n" + "-" * 60)
    print(" HARDWARE CONFIGURATION")
    print("-" * 60)
    print(f"  CPU Cores (logical):  {info['cpu_count']}")
    print(f"  CPU Cores (physical): {info['cpu_count_physical']}")
    if HAS_PSUTIL:
        print(f"  CPU Usage:            {info['cpu_percent']}%")
        print(f"  Memory Total:         {info['memory_total_gb']} GB")
        print(f"  Memory Available:     {info['memory_available_gb']} GB")
        print(f"  Memory Usage:         {info['memory_percent']}%")
    print(f"  Parallel Jobs:        {info['cpu_count']} (using all cores)")
    print("-" * 60 + "\n")
    return info


def prepare_data(df: pd.DataFrame, target_col: str = "volatility_spike"):
    """Prepare features and target from dataframe."""
    exclude_cols = [
        target_col,
        "timestamp",
        "ts",
        "time",
        "label",
        "future_volatility",
        "price",
        "best_bid",
        "best_ask",
        "midprice",
    ]

    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in [
            np.float64,
            np.float32,
            np.int64,
            np.int32,
        ]:
            feature_cols.append(col)

    X = df[feature_cols].fillna(0)
    y = df[target_col].values

    return X, y, feature_cols


def get_param_grids() -> Dict:
    """Define hyperparameter grids for GridSearchCV."""
    return {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear"],
            "clf__class_weight": ["balanced", None],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "min_samples_split": [2, 5],
            "subsample": [0.8, 1.0],
        },
    }


def train_with_gridsearch(
    model,
    param_grid: Dict,
    X_train,
    y_train,
    model_name: str,
    n_jobs: int = -1,
    cv: int = 5,
) -> Tuple:
    """Train model with GridSearchCV for hyperparameter optimization."""
    print(f"      Searching {len(param_grid)} hyperparameters...")
    print(f"      Using {n_jobs} CPU cores for parallel search")

    # Use PR-AUC as scoring metric (best for imbalanced data)
    scorer = make_scorer(average_precision_score, needs_proba=True)

    # Stratified K-Fold to maintain class balance
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv_splitter,
        scoring=scorer,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time

    print(f"      Best CV Score (PR-AUC): {grid_search.best_score_:.4f}")
    print(f"      Best Params: {grid_search.best_params_}")
    print(f"      Search Time: {search_time:.1f}s")

    return grid_search.best_estimator_, grid_search.best_params_, search_time


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    use_gridsearch: bool = False,
    param_grid: Dict = None,
    n_jobs: int = -1,
) -> Dict:
    """Train a model and evaluate it."""
    best_params = None
    search_time = 0

    if use_gridsearch and param_grid:
        model, best_params, search_time = train_with_gridsearch(
            model, param_grid, X_train, y_train, model_name, n_jobs
        )
        train_time = search_time
    else:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

    # Predict
    start_time = time.time()
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)
    inference_time = (time.time() - start_time) * 1000

    # Find optimal threshold
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
        "pr_auc": (
            average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
        ),
        "train_time_s": train_time,
        "inference_time_ms": inference_time,
        "inference_per_sample_ms": inference_time / len(y_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "best_params": best_params,
        "used_gridsearch": use_gridsearch and param_grid is not None,
    }

    # Get feature importances if available
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        metrics["feature_importances"] = np.abs(model.coef_[0]).tolist()
    elif hasattr(model, "named_steps") and hasattr(
        model.named_steps.get("clf", {}), "coef_"
    ):
        metrics["feature_importances"] = np.abs(
            model.named_steps["clf"].coef_[0]
        ).tolist()

    return metrics, model


def print_results(results: List[Dict], feature_names: List[str] = None):
    """Print comparison results with feature importances."""
    print("\n" + "=" * 100)
    print(" MODEL COMPARISON RESULTS")
    print("=" * 100)

    # Header
    print(
        f"\n{'Model':<25} {'PR-AUC':>10} {'ROC-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Latency':>12} {'GridSearch':>12}"
    )
    print("-" * 110)

    sorted_results = sorted(results, key=lambda x: x["pr_auc"], reverse=True)

    for r in sorted_results:
        latency_str = f"{r['inference_per_sample_ms']:.4f} ms"
        gs_str = "Yes" if r.get("used_gridsearch", False) else "No"
        print(
            f"{r['model_name']:<25} {r['pr_auc']:>10.4f} {r['roc_auc']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {latency_str:>12} {gs_str:>12}"
        )

    print("-" * 110)

    winner = sorted_results[0]
    print(f"\n*** BEST MODEL: {winner['model_name']} ***")
    print(f"    PR-AUC: {winner['pr_auc']:.4f}")
    print(
        f"    F1: {winner['f1']:.4f} | Precision: {winner['precision']:.4f} | Recall: {winner['recall']:.4f}"
    )
    print(f"    Inference: {winner['inference_per_sample_ms']:.4f} ms/sample")

    if winner.get("best_params"):
        print(f"    Best Hyperparameters: {winner['best_params']}")

    # Print feature importances for winner
    if feature_names and winner.get("feature_importances"):
        print(f"\n    Top Feature Importances for {winner['model_name']}:")
        importances = winner["feature_importances"]
        feat_imp = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        for i, (feat, imp) in enumerate(feat_imp[:7]):
            print(f"      {i+1}. {feat}: {imp:.4f}")

    return winner


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare models with GridSearchCV"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features_sample.parquet",
        help="Path to features file",
    )
    parser.add_argument(
        "--target", type=str, default="volatility_spike", help="Target column"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Test set fraction"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Use GridSearchCV for hyperparameter tuning",
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Number of CV folds for GridSearchCV"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all cores)",
    )
    parser.add_argument("--save-best", action="store_true", help="Save the best model")
    parser.add_argument(
        "--output", type=str, default="docs/model_comparison_results.json"
    )

    args = parser.parse_args()

    # Set UTF-8 encoding for Windows console
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 100)
    print(" TRAIN AND COMPARE MODELS WITH HYPERPARAMETER OPTIMIZATION")
    print("=" * 100)

    # Print hardware info
    hw_info = print_hardware_info()
    n_jobs = args.n_jobs if args.n_jobs != -1 else hw_info["cpu_count"]

    # Load data
    print(f"[*] Loading data from: {args.features}")

    if not os.path.exists(args.features):
        print(f"[ERROR] File not found: {args.features}")
        sys.exit(1)

    df = pd.read_parquet(args.features)
    print(f"    Total samples: {len(df)}")

    # Prepare data
    X, y, feature_names = prepare_data(df, args.target)
    print(f"    Features: {len(feature_names)}")
    print(f"    Positive class rate: {y.mean()*100:.2f}%")

    # Split data
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Test positive rate: {y_test.mean()*100:.2f}%")

    if args.grid_search:
        print(f"\n[*] GridSearchCV enabled with {args.cv}-fold cross-validation")
        print(f"    Using {n_jobs} parallel jobs")

    # Get parameter grids
    param_grids = get_param_grids()

    # Define models
    models = [
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            ),
            param_grids.get("Logistic Regression"),
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=42, n_jobs=n_jobs),
            param_grids.get("Random Forest"),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42),
            param_grids.get("Gradient Boosting"),
        ),
    ]

    # Add baseline (no grid search)
    class SimpleBaseline:
        """Z-score based baseline detector."""

        def __init__(self, threshold_percentile=90):
            self.threshold_percentile = threshold_percentile
            self.threshold = None

        def fit(self, X, y):
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

    models.insert(0, ("Z-Score Baseline", SimpleBaseline(), None))

    # Train and evaluate
    print("\n[*] Training and evaluating models...")
    results = []
    trained_models = {}
    total_start = time.time()

    for name, model, param_grid in models:
        print(f"\n    Training: {name}")
        try:
            use_gs = args.grid_search and param_grid is not None
            metrics, trained_model = train_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                name,
                use_gridsearch=use_gs,
                param_grid=param_grid,
                n_jobs=n_jobs,
            )
            results.append(metrics)
            trained_models[name] = trained_model
            print(f"    [OK] PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1']:.4f}")
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback

            traceback.print_exc()

    total_time = time.time() - total_start
    print(f"\n[*] Total training time: {total_time:.1f}s")

    # Print results
    winner = print_results(results, feature_names)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "features_file": args.features,
                "n_samples": len(df),
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "test_size": args.test_size,
                "used_gridsearch": args.grid_search,
                "cv_folds": args.cv if args.grid_search else None,
                "n_jobs": n_jobs,
                "hardware": hw_info,
                "total_training_time_s": total_time,
                "results": results,
                "recommendation": winner["model_name"],
            },
            f,
            indent=2,
        )
    print(f"\n[*] Results saved to: {args.output}")

    # Save best model
    if args.save_best:
        best_model = trained_models[winner["model_name"]]
        output_dir = Path("models/artifacts") / winner["model_name"].lower().replace(
            " ", "_"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": best_model,
            "feature_names": feature_names,
            "threshold": winner["threshold"],
            "metrics": winner,
            "trained_at": datetime.now().isoformat(),
            "hyperparameters": winner.get("best_params"),
        }

        with open(output_dir / "model.pkl", "wb") as f:
            pickle.dump(model_data, f)

        print(f"[*] Best model saved to: {output_dir / 'model.pkl'}")

    # Recommendation
    print("\n" + "=" * 100)
    print(" RECOMMENDATION FOR REAL-TIME DEPLOYMENT")
    print("=" * 100)
    print(
        f"""
  Best Model: {winner['model_name']}
  
  Performance:
  - PR-AUC: {winner['pr_auc']:.4f}
  - Recall: {winner['recall']:.4f} ({winner['recall']*100:.1f}% of spikes detected)
  - Precision: {winner['precision']:.4f} ({winner['precision']*100:.1f}% alerts are true)
  - Inference: {winner['inference_per_sample_ms']:.4f} ms/sample
  
  Training Info:
  - GridSearchCV: {'Yes' if winner.get('used_gridsearch') else 'No'}
  - Best Params: {winner.get('best_params', 'N/A')}
  - Training Time: {winner['train_time_s']:.1f}s
  
  To deploy:
    $env:MODEL_VERSION="{winner['model_name'].lower().replace(' ', '_')}"
    docker compose up -d api
"""
    )
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()

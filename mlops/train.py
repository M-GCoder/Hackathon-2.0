"""
ML Model Training — Train multiple models and select the best one.
Uses MLflow for experiment tracking.
"""

import os
import sys
import pickle
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "flights_processed.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# Feature columns for training (exclude metadata and target)
EXCLUDE_COLS = [
    'Year', 'FlightDate', 'Reporting_Airline', 'Origin', 'Dest',
    'ArrDelay', 'Flight_Number_Reporting_Airline', 'is_delayed'
]


def load_data(filepath: str = None):
    """Load processed data and split into train/test."""
    filepath = filepath or PROCESSED_DATA
    df = pd.read_csv(filepath)

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    X = df[feature_cols].values
    y = df['is_delayed'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"📊 Data loaded: {len(df)} rows, {len(feature_cols)} features")
    logger.info(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"   Delay rate: {y.mean():.1%}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def get_models() -> dict:
    """Return dict of model name -> (model, params)."""
    return {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            {'max_iter': 1000, 'class_weight': 'balanced', 'solver': 'lbfgs'}
        ),
        'RandomForest': (
            RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, class_weight='balanced', n_jobs=-1
            ),
            {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'class_weight': 'balanced'}
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                min_samples_split=5, random_state=42
            ),
            {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, 'min_samples_split': 5}
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics


def train_all_models():
    """Train all models, log to MLflow, and save the best one."""
    try:
        from mlops.mlflow_tracking import setup_mlflow, log_model_run
        mlflow_available = True
        setup_mlflow()
    except Exception as e:
        logger.warning(f"⚠️ MLflow not available: {e}. Training without tracking.")
        mlflow_available = False

    # Load data
    X_train, X_test, y_train, y_test, scaler, feature_cols = load_data()

    # Save scaler and feature columns
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(FEATURE_COLS_PATH, 'wb') as f:
        pickle.dump(feature_cols, f)

    # Train models
    models = get_models()
    results = {}
    best_model = None
    best_f1 = 0
    best_name = ""

    for name, (model, params) in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"🏋️ Training: {name}")
        logger.info(f"{'='*50}")

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics['train_time_seconds'] = round(train_time, 2)

        logger.info(f"   ⏱ Training time: {train_time:.1f}s")
        logger.info(f"   📊 Metrics: {metrics}")

        # Log to MLflow
        if mlflow_available:
            try:
                log_model_run(model, name, params, metrics,
                              tags={'framework': 'sklearn', 'task': 'binary_classification'})
            except Exception as e:
                logger.warning(f"   ⚠️ MLflow logging failed: {e}")

        results[name] = {'model': model, 'metrics': metrics, 'params': params}

        # Track best
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = model
            best_name = name

    # Save best model
    logger.info(f"\n{'='*50}")
    logger.info(f"🏆 Best Model: {best_name} (F1: {best_f1:.4f})")
    logger.info(f"{'='*50}")

    with open(BEST_MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"💾 Best model saved to: {BEST_MODEL_PATH}")

    # Print comparison table
    logger.info("\n📊 MODEL COMPARISON:")
    logger.info(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    logger.info("-" * 75)
    for name, res in results.items():
        m = res['metrics']
        marker = " ⭐" if name == best_name else ""
        logger.info(
            f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['roc_auc']:>10.4f}{marker}"
        )

    return results, best_model, best_name, feature_cols, scaler


# Try to import and train XGBoost separately
def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model if available."""
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        logger.warning("⚠️ XGBoost not available, skipping")
        return None


if __name__ == "__main__":
    results, best_model, best_name, feature_cols, scaler = train_all_models()
    print(f"\n✅ Training complete! Best model: {best_name}")

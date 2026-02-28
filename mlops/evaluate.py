"""
ML Model Evaluation — Load and evaluate trained models with detailed metrics.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "flights_processed.csv")

EXCLUDE_COLS = [
    'Year', 'FlightDate', 'Reporting_Airline', 'Origin', 'Dest',
    'ArrDelay', 'Flight_Number_Reporting_Airline', 'is_delayed'
]


def load_model():
    """Load the best model, scaler, and feature columns."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLS_PATH, 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols


def evaluate(model=None, scaler=None, feature_cols=None):
    """Run full evaluation on test data."""

    if model is None:
        model, scaler, feature_cols = load_model()

    # Load data
    df = pd.read_csv(PROCESSED_DATA)
    X = df[feature_cols].values
    y = df['is_delayed'].values
    X_scaled = scaler.transform(X)

    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['On-Time', 'Delayed'])

    logger.info("\n" + "=" * 60)
    logger.info("📊 MODEL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Test samples: {len(y_test)}")
    logger.info(f"\n{report}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    logger.info(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Log to MLflow
    try:
        from mlops.mlflow_tracking import setup_mlflow
        import mlflow
        setup_mlflow()
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("stage", "evaluation")
    except Exception as e:
        logger.warning(f"⚠️ MLflow evaluation logging failed: {e}")

    return metrics, cm, report


if __name__ == "__main__":
    metrics, cm, report = evaluate()
    print(f"\n✅ Evaluation complete!")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

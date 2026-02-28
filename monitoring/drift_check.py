"""
Monitoring — Prediction logging and data drift detection.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PREDICTION_LOG = os.path.join(PROJECT_ROOT, "monitoring", "prediction_log.csv")
TRAINING_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "flights_processed.csv")
DRIFT_REPORT = os.path.join(PROJECT_ROOT, "monitoring", "drift_report.json")

# Features to monitor for drift
MONITORED_FEATURES = ['distance', 'dep_delay', 'dep_hour', 'month']


def log_prediction(prediction_data: dict):
    """Append a prediction to the log file."""
    os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)

    prediction_data['timestamp'] = datetime.now().isoformat()

    df = pd.DataFrame([prediction_data])
    header = not os.path.exists(PREDICTION_LOG)
    df.to_csv(PREDICTION_LOG, mode='a', header=header, index=False)
    logger.info(f"📝 Logged prediction: {prediction_data.get('prediction', 'N/A')}")


def get_prediction_stats() -> dict:
    """Get summary statistics from prediction log."""
    if not os.path.exists(PREDICTION_LOG):
        return {"total": 0, "message": "No predictions logged yet"}

    df = pd.read_csv(PREDICTION_LOG)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    stats_dict = {
        "total_predictions": len(df),
        "delayed_rate": float(df['prediction'].mean()) if 'prediction' in df.columns else None,
        "avg_probability": float(df['probability'].mean()) if 'probability' in df.columns else None,
        "first_prediction": df['timestamp'].min().isoformat(),
        "last_prediction": df['timestamp'].max().isoformat(),
    }

    # Per-carrier stats
    if 'carrier' in df.columns:
        stats_dict['predictions_by_carrier'] = df['carrier'].value_counts().to_dict()

    return stats_dict


def check_drift(reference_data: pd.DataFrame = None, production_data: pd.DataFrame = None) -> dict:
    """
    Check for data drift between training data and production predictions.
    Uses Kolmogorov-Smirnov test for numerical features.

    Returns:
        Dict of feature -> {drift: bool, p_value: float, statistic: float}
    """
    if reference_data is None:
        if not os.path.exists(TRAINING_DATA):
            logger.warning("Training data not found for drift comparison")
            return {}
        reference_data = pd.read_csv(TRAINING_DATA)

    if production_data is None:
        if not os.path.exists(PREDICTION_LOG):
            logger.warning("No prediction log found for drift comparison")
            return {}
        production_data = pd.read_csv(PREDICTION_LOG)

    if len(production_data) < 30:
        logger.info("Not enough production data for drift analysis (need 30+)")
        return {}

    drift_results = {}
    SIGNIFICANCE_LEVEL = 0.05

    # Map prediction log columns to training data columns
    column_mapping = {
        'distance': 'Distance',
        'dep_delay': 'DepDelay',
        'dep_hour': 'dep_hour',
        'month': 'Month'
    }

    for prod_col in MONITORED_FEATURES:
        train_col = column_mapping.get(prod_col, prod_col)

        if prod_col not in production_data.columns or train_col not in reference_data.columns:
            continue

        ref_values = reference_data[train_col].dropna().values
        prod_values = production_data[prod_col].dropna().values

        if len(ref_values) == 0 or len(prod_values) == 0:
            continue

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(ref_values, prod_values)

        drift_results[prod_col] = {
            'drift': p_value < SIGNIFICANCE_LEVEL,
            'p_value': round(float(p_value), 6),
            'ks_statistic': round(float(ks_stat), 6),
            'ref_mean': round(float(np.mean(ref_values)), 4),
            'prod_mean': round(float(np.mean(prod_values)), 4),
            'ref_std': round(float(np.std(ref_values)), 4),
            'prod_std': round(float(np.std(prod_values)), 4),
        }

        status = "🔴 DRIFT" if p_value < SIGNIFICANCE_LEVEL else "🟢 OK"
        logger.info(f"   {prod_col}: {status} (p={p_value:.4f}, KS={ks_stat:.4f})")

    # Save drift report
    if drift_results:
        import json
        report = {
            "timestamp": datetime.now().isoformat(),
            "n_reference": len(reference_data),
            "n_production": len(production_data),
            "features": drift_results,
            "any_drift": any(r['drift'] for r in drift_results.values())
        }
        os.makedirs(os.path.dirname(DRIFT_REPORT), exist_ok=True)
        with open(DRIFT_REPORT, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📊 Drift report saved to: {DRIFT_REPORT}")

    return drift_results


if __name__ == "__main__":
    logger.info("🔍 Running drift check...")
    results = check_drift()
    if results:
        print("\n📊 Drift Analysis Results:")
        for feature, result in results.items():
            status = "🔴 DRIFT" if result['drift'] else "🟢 OK"
            print(f"  {feature}: {status} (p={result['p_value']:.4f})")
    else:
        print("Not enough data for drift analysis.")

    stats = get_prediction_stats()
    print(f"\n📈 Prediction Stats: {stats}")

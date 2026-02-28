"""
MLflow Tracking — Experiment setup and logging helpers.
"""

import mlflow
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"file:///{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlruns').replace(os.sep, '/')}"
)
EXPERIMENT_NAME = "flight-delay-prediction"


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"📊 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"📊 Experiment: {EXPERIMENT_NAME} (ID: {experiment.experiment_id})")
    return experiment


def log_model_run(model, model_name: str, params: dict, metrics: dict,
                  artifacts: dict = None, tags: dict = None):
    """
    Log a complete model training run to MLflow.

    Args:
        model: Trained model object
        model_name: Name for the model
        params: Hyperparameters dict
        metrics: Evaluation metrics dict
        artifacts: Optional dict of artifact_name -> artifact_path
        tags: Optional dict of tags
    """
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("model_type", model_name)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        # Log artifacts
        if artifacts:
            for name, path in artifacts.items():
                if os.path.exists(path):
                    mlflow.log_artifact(path, name)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ Logged run: {model_name} (ID: {run_id})")
        logger.info(f"   Metrics: {metrics}")

        return run_id


def get_best_run(metric: str = "f1_score", ascending: bool = False):
    """Get the best run from the experiment based on a metric."""
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )

    if len(runs) == 0:
        return None

    return runs.iloc[0]


if __name__ == "__main__":
    setup_mlflow()
    print("✅ MLflow tracking initialized")

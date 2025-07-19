"""
MLflow utilities for consistent experiment tracking across all training scripts.
Provides standardized logging and model registration functions.
"""

import os
import json
import logging
import torch
import mlflow
import mlflow.pytorch
from datetime import datetime


def setup_mlflow_experiment(experiment_name_suffix=""):
    """
    Setup MLflow experiment with consistent naming and configuration.
    
    Args:
        experiment_name_suffix: Optional suffix for experiment name
    
    Returns:
        str: The experiment name that was set
    """
    # Ensure MLflow tracking URI is picked up from the environment
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    # Set experiment name with optional suffix
    base_name = "seldon-system"
    if experiment_name_suffix:
        experiment_name = f"{base_name}-{experiment_name_suffix}"
    else:
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", base_name)
    
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow experiment set to: {experiment_name}")
    
    return experiment_name


def log_training_params(config_dict, model_info=None):
    """
    Log training parameters to MLflow with standardized naming.
    
    Args:
        config_dict: Dictionary of training configuration parameters
        model_info: Optional dictionary of model-specific information
    """
    for key, value in config_dict.items():
        mlflow.log_param(key, value)
    
    if model_info:
        for key, value in model_info.items():
            mlflow.log_param(f"model_{key}", value)


def log_training_metrics(metrics_dict, step=None):
    """
    Log training metrics to MLflow with optional step.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        step: Optional step number for time-series metrics
    """
    for metric_name, value in metrics_dict.items():
        if step is not None:
            mlflow.log_metric(metric_name, value, step=step)
        else:
            mlflow.log_metric(metric_name, value)


def log_model_with_artifacts(model, sample_input, model_variant, results_dict=None):
    """
    Log PyTorch model to MLflow with artifacts and metadata.
    
    Args:
        model: Trained PyTorch model
        sample_input: Sample input tensor for model signature
        model_variant: Model variant name (baseline, enhanced, etc.)
        results_dict: Optional results dictionary to save as artifact
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Move to CPU for MLflow compatibility
        model_cpu = model.cpu()
        sample_input_cpu = sample_input.cpu()
        
        # Create model info artifact
        if results_dict:
            model_info_path = f"model_results_{model_variant}.json"
            with open(model_info_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            mlflow.log_artifact(model_info_path)
        
        # Get absolute path to models.py for MLflow
        models_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.py")
        code_paths = [models_py_path] if os.path.exists(models_py_path) else None
        
        # Register model
        registered_model_name = f"FinancialDirectionPredictor_{model_variant.title()}"
        mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            artifact_path=f"model_{model_variant}",
            registered_model_name=registered_model_name,
            code_paths=code_paths,
            input_example=sample_input_cpu.numpy(),
            signature=mlflow.models.infer_signature(
                sample_input_cpu.numpy(), 
                model_cpu(sample_input_cpu).detach().numpy()
            )
        )
        
        logging.info(f"✅ Model ({model_variant}) successfully logged to MLflow")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to log model to MLflow: {e}")
        return False


def create_mlflow_run(run_name_prefix, model_variant=""):
    """
    Create MLflow run with standardized naming.
    
    Args:
        run_name_prefix: Prefix for the run name
        model_variant: Optional model variant to include in name
    
    Returns:
        MLflow run context manager
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if model_variant:
        run_name = f"{run_name_prefix}_{model_variant}_{timestamp}"
    else:
        run_name = f"{run_name_prefix}_{timestamp}"
    
    return mlflow.start_run(run_name=run_name)


def log_training_artifacts(log_file_path="training.log"):
    """
    Log common training artifacts to MLflow.
    
    Args:
        log_file_path: Path to training log file
    """
    if os.path.exists(log_file_path):
        mlflow.log_artifact(log_file_path)
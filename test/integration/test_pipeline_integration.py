import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import subprocess
import sys
from unittest.mock import patch, MagicMock

from torch.utils.data import DataLoader

from src.train_pytorch_model import train_model, load_processed_data, StockPredictor, PROCESSED_DATA_DIR
from src.feature_engineering_pytorch import FinancialTimeSeriesDataset  # Need this for Dataset creation


# Fixtures for temporary directories and data
@pytest.fixture
def temp_processed_data_dir(tmp_path):
    temp_dir = tmp_path / "test_processed_data"
    os.makedirs(temp_dir, exist_ok=True)

    # Create dummy data
    train_f = np.random.rand(100, 10)  # 100 samples, 10 features
    train_t = np.random.randint(0, 2, 100)  # 100 binary targets
    val_f = np.random.rand(20, 10)
    val_t = np.random.randint(0, 2, 20)
    test_f = np.random.rand(20, 10)
    test_t = np.random.randint(0, 2, 20)

    np.save(os.path.join(temp_dir, 'train_features.npy'), train_f)
    np.save(os.path.join(temp_dir, 'train_targets.npy'), train_t)
    np.save(os.path.join(temp_dir, 'val_features.npy'), val_f)
    np.save(os.path.join(temp_dir, 'val_targets.npy'), val_t)
    np.save(os.path.join(temp_dir, 'test_features.npy'), test_f)
    np.save(os.path.join(temp_dir, 'test_targets.npy'), test_t)

    yield str(temp_dir)

# Define PROCESSED_DATA_DIR at the module level
PROCESSED_DATA_DIR = None

def test_train_model_main_execution(tmp_path):
    # Set up temporary processed data directory
    temp_processed_dir = tmp_path / "processed"
    temp_processed_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy processed data
    train_f = np.random.rand(10, 5)
    train_t = np.random.randint(0, 2, 10)
    val_f = np.random.rand(5, 5)
    val_t = np.random.randint(0, 2, 5)
    test_f = np.random.rand(5, 5)
    test_t = np.random.randint(0, 2, 5)
    np.save(temp_processed_dir / 'train_features.npy', train_f)
    np.save(temp_processed_dir / 'train_targets.npy', train_t)
    np.save(temp_processed_dir / 'val_features.npy', val_f)
    np.save(temp_processed_dir / 'val_targets.npy', val_t)
    np.save(temp_processed_dir / 'test_features.npy', test_f)
    np.save(temp_processed_dir / 'test_targets.npy', test_t)

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["PROCESSED_DATA_DIR"] = str(temp_processed_dir)
    env["BATCH_SIZE"] = "2"
    env["EPOCHS"] = "2"
    env["LEARNING_RATE"] = "0.01"
    env["SEQUENCE_LENGTH"] = "2"
    # Optionally set MLFLOW_TRACKING_URI to a temp dir or mock server
    temp_mlflow_dir = tmp_path / "mlruns"
    env["MLFLOW_TRACKING_URI"] = str(temp_mlflow_dir)

    # Run the training script as a subprocess
    result = subprocess.run(
        [sys.executable, "src/train_pytorch_model.py"],
        env=env,
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    assert result.returncode == 0

    # Check that model artifacts and MLflow outputs exist
    # Check for MLflow run directory and model artifact
    assert temp_mlflow_dir.exists()
    run_dirs = list(temp_mlflow_dir.glob("*/"))
    assert run_dirs, "No MLflow run directories found."

    # Check processed data integrity
    loaded_train_f = np.load(temp_processed_dir / 'train_features.npy')
    assert loaded_train_f.shape == (10, 5)
    assert not np.isnan(loaded_train_f).any()

    # Optionally, check for model output in MLflow artifact directory
    # (This depends on MLflow's directory structure and may require more detailed checks)
# src/train_pytorch_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
import mlflow
import mlflow.pytorch  # Needed for mlflow.pytorch.log_model
import json  # For loading model config if needed
import math  # For checking NaN values

# Import the model definition from models.py
from models import StockPredictor
from feature_engineering_pytorch import FinancialTimeSeriesDataset, SEQUENCE_LENGTH  # Re-use the Dataset class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROCESSED_DATA_INPUT_DIR = os.getenv("PROCESSED_DATA_INPUT_DIR", "/mnt/shared-data/processed")
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_DIR", "models")  # Directory to save trained models locally before MLflow logging
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Training Hyperparameters (configurable via environment variables)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
EPOCHS = int(os.environ.get("EPOCHS", 20))  # Keep this relatively low for quick iteration on CPU
RANDOM_SEED = 42
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 10))

# Model Hyperparameters (must match what was used/expected by feature engineering)
# These should be determined based on your feature_engineering_pytorch.py output
# You might pass these as command-line arguments or load from a config file.
# For now, let's hardcode based on common setup, but be aware these need to align
INPUT_SIZE = 12  # This should be `train_features.shape[1]` from FE script. Example value.
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT_PROB = 0.2
NUM_CLASSES = 1  # Binary classification (up/down)

# Ensure MLflow tracking URI is picked up from the environment
if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


# --- Functions ---
def load_processed_data(data_dir: str):
    """Loads processed features and targets from .npy files."""
    try:
        train_features = np.load(os.path.join(data_dir, 'train_features.npy'))
        train_targets = np.load(os.path.join(data_dir, 'train_targets.npy'))
        val_features = np.load(os.path.join(data_dir, 'val_features.npy'))
        val_targets = np.load(os.path.join(data_dir, 'val_targets.npy'))
        test_features = np.load(os.path.join(data_dir, 'test_features.npy'))
        test_targets = np.load(os.path.join(data_dir, 'test_targets.npy'))
        logging.info("Successfully loaded processed data.")
        return {
            'train_features': train_features, 'train_targets': train_targets,
            'val_features': val_features, 'val_targets': val_targets,
            'test_features': test_features, 'test_targets': test_targets
        }
    except FileNotFoundError as e:
        logging.error(
            f"Missing processed data files in {data_dir}. Run feature_engineering_pytorch.py first. Error: {e}")
        exit(1)


def safe_log_metric(metric_name: str, value: float, step: int):
    """
    Safely log a metric to MLflow, handling NaN values and potential duplicates.
    """
    if math.isnan(value) or math.isinf(value):
        logging.warning(f"Skipping logging of {metric_name} at step {step} due to NaN/Inf value: {value}")
        return
    
    try:
        mlflow.log_metric(metric_name, value, step=step)
    except Exception as e:
        logging.warning(f"Failed to log metric {metric_name}={value} at step {step}: {e}")


def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int
):
    """
    Trains and validates the PyTorch model.
    """
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_dataloader):
            # FIXED: Ensure proper tensor shapes and types
            features = features.to('cpu')
            targets = targets.to('cpu').float().unsqueeze(1)  # Shape: [batch_size, 1]

            # Check for NaN in input data
            if torch.isnan(features).any() or torch.isnan(targets).any():
                logging.warning(f"NaN detected in batch {batch_idx} of epoch {epoch}. Skipping batch.")
                continue

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(features)  # Forward pass - Shape: [batch_size, 1]
            
            # Check for NaN in model outputs
            if torch.isnan(outputs).any():
                logging.warning(f"NaN detected in model outputs at batch {batch_idx} of epoch {epoch}. Skipping batch.")
                continue
                
            loss = criterion(outputs, targets)  # Calculate loss
            
            # Check for NaN in loss
            if torch.isnan(loss):
                logging.warning(f"NaN loss detected at batch {batch_idx} of epoch {epoch}. Skipping batch.")
                continue
                
            loss.backward()  # Backward pass
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  # Update weights

            train_loss += loss.item() * features.size(0)
            train_batches += features.size(0)

        # Calculate average training loss, handling case where no valid batches
        if train_batches > 0:
            train_loss /= train_batches
        else:
            logging.error(f"No valid training batches in epoch {epoch}")
            train_loss = float('nan')

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        val_batches = 0

        with torch.no_grad():  # No gradient calculations during validation
            for features, targets in val_dataloader:
                # FIXED: Ensure proper tensor shapes and types
                features = features.to('cpu')
                targets = targets.to('cpu').float().unsqueeze(1)  # Shape: [batch_size, 1]
                
                # Check for NaN in validation data
                if torch.isnan(features).any() or torch.isnan(targets).any():
                    logging.warning(f"NaN detected in validation batch. Skipping batch.")
                    continue
                
                outputs = model(features)  # Shape: [batch_size, 1]
                
                # Check for NaN in validation outputs
                if torch.isnan(outputs).any():
                    logging.warning(f"NaN detected in validation model outputs. Skipping batch.")
                    continue
                    
                loss = criterion(outputs, targets)
                
                # Check for NaN in validation loss
                if torch.isnan(loss):
                    logging.warning(f"NaN validation loss detected. Skipping batch.")
                    continue
                    
                val_loss += loss.item() * features.size(0)
                val_batches += features.size(0)

                # FIXED: Apply sigmoid for predictions since we're using BCEWithLogitsLoss
                predicted_classes = (torch.sigmoid(outputs) > 0.5).float()
                total_predictions += targets.size(0)
                correct_predictions += (predicted_classes == targets).sum().item()

        # Calculate average validation loss and accuracy
        if val_batches > 0:
            val_loss /= val_batches
            val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        else:
            logging.error(f"No valid validation batches in epoch {epoch}")
            val_loss = float('nan')
            val_accuracy = 0.0

        logging.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        # Safely log metrics to MLflow
        safe_log_metric("train_loss", train_loss, step=epoch)
        safe_log_metric("val_loss", val_loss, step=epoch)
        safe_log_metric("val_accuracy", val_accuracy, step=epoch)

        # Simple early stopping (only if we have valid loss)
        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

    logging.info(f"Training finished. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
    return model


def run_training_pipeline():
    global INPUT_SIZE  # Declare global before any use
    logging.info("Starting PyTorch model training process.")

    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Start an MLflow run
    with mlflow.start_run(run_name="pytorch_stock_predictor_training"):
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("num_layers", NUM_LAYERS)
        mlflow.log_param("dropout_prob", DROPOUT_PROB)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("device", "cpu")  # Explicitly log CPU usage

        # 1. Load processed data
        data = load_processed_data(PROCESSED_DATA_INPUT_DIR)
        train_features = data['train_features']
        train_targets = data['train_targets']
        val_features = data['val_features']
        val_targets = data['val_targets']

        # Check for NaN values in loaded data
        if np.isnan(train_features).any():
            logging.error("NaN values found in training features!")
            return
        if np.isnan(train_targets).any():
            logging.error("NaN values found in training targets!")
            return

        actual_input_size = train_features.shape[1]
        if actual_input_size != INPUT_SIZE:
            logging.warning(
                f"Configured INPUT_SIZE ({INPUT_SIZE}) does not match actual data features ({actual_input_size}). Adjusting to {actual_input_size}.")
            INPUT_SIZE = actual_input_size
            mlflow.log_param("input_size", INPUT_SIZE)  # Re-log if adjusted

        mlflow.log_param("input_size", INPUT_SIZE)
        
        train_dataset = FinancialTimeSeriesDataset(train_features, train_targets, sequence_length=SEQUENCE_LENGTH)
        val_dataset = FinancialTimeSeriesDataset(val_features, val_targets, sequence_length=SEQUENCE_LENGTH)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = StockPredictor(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout_prob=DROPOUT_PROB
        ).to('cpu')

        # Initialize model weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        torch.nn.init.constant_(param.data, 0)

        model.apply(init_weights)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

        logging.info(f"Model architecture:\n{model}")
        logging.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        trained_model = train_model(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            criterion,
            EPOCHS
        )

        # Only log the model if training completed without major issues
        try:
            mlflow.pytorch.log_model(
                pytorch_model=trained_model,
                artifact_path="stock_predictor_model",
                registered_model_name="FinancialDirectionPredictor",
                code_paths=["src/models.py"]
            )
            logging.info("PyTorch model logged and registered with MLflow.")
        except Exception as e:
            logging.error(f"Failed to log model to MLflow: {e}")
            
        logging.info("PyTorch model training completed successfully.")


# --- Main execution ---
if __name__ == "__main__":
    run_training_pipeline()
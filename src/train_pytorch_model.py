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

# Import the model definition from models.py
from models import StockPredictor
from feature_engineering_pytorch import FinancialTimeSeriesDataset, SEQUENCE_LENGTH  # Re-use the Dataset class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
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
        for batch_idx, (features, targets) in enumerate(train_dataloader):
            # Ensure data is on CPU
            features, targets = features.to('cpu'), targets.to('cpu').unsqueeze(1)  # Add a dimension for BCELoss

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, targets)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_dataloader.dataset)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():  # No gradient calculations during validation
            for features, targets in val_dataloader:
                features, targets = features.to('cpu'), targets.to('cpu').unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)

                predicted_classes = (outputs > 0.5).float()  # Binary prediction
                total_predictions += targets.size(0)
                correct_predictions += (predicted_classes == targets).sum().item()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct_predictions / total_predictions

        logging.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        # Simple early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # You might save the best model state dict here if not using MLflow's auto-logging
            # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            # logging.info(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

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
        mlflow.log_param("input_size", INPUT_SIZE)
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("num_layers", NUM_LAYERS)
        mlflow.log_param("dropout_prob", DROPOUT_PROB)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("device", "cpu")  # Explicitly log CPU usage

        # 1. Load processed data
        data = load_processed_data(PROCESSED_DATA_DIR)
        train_features = data['train_features']
        train_targets = data['train_targets']
        val_features = data['val_features']
        val_targets = data['val_targets']

        actual_input_size = train_features.shape[1]
        if actual_input_size != INPUT_SIZE:
            logging.warning(
                f"Configured INPUT_SIZE ({INPUT_SIZE}) does not match actual data features ({actual_input_size}). Adjusting to {actual_input_size}.")
            INPUT_SIZE = actual_input_size
            mlflow.log_param("input_size", INPUT_SIZE)  # Re-log if adjusted

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

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        mlflow.pytorch.log_model(
            pytorch_model=trained_model,
            artifact_path="stock_predictor_model",
            registered_model_name="FinancialDirectionPredictor",
            code_paths=["src/models.py"]
        )
        logging.info("PyTorch model logged and registered with MLflow.")
        logging.info("PyTorch model training completed successfully.")


# --- Main execution ---
if __name__ == "__main__":
    run_training_pipeline()
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
from datetime import datetime
import time
from sklearn.utils.class_weight import compute_class_weight

# Import the model definition from models.py
from models import StockPredictor
from feature_engineering_pytorch import FinancialTimeSeriesDataset, SEQUENCE_LENGTH  # Re-use the Dataset class

# Configure logging with more detailed format
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('training.log')  # File output
    ]
)

# --- Configuration ---
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/mnt/shared-data/processed")
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_DIR", "models")  # Directory to save trained models locally before MLflow logging
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model variant configuration - can be set via environment variables for A/B testing
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "baseline")  # baseline, enhanced, lightweight
logging.info(f"DEBUG: MODEL_VARIANT env: {os.environ.get('MODEL_VARIANT')}, variable: {MODEL_VARIANT}")

# Training Hyperparameters (configurable via environment variables)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
EPOCHS = int(os.environ.get("EPOCHS", 50))  # Increased for better convergence
RANDOM_SEED = 42
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 10))

# Model Hyperparameters - Tuned configurations for A/B testing variants
if MODEL_VARIANT == "enhanced":
    # Enhanced model for better performance - optimized for financial time series
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.0008))  # Optimized LR
    INPUT_SIZE = 35  # Fixed to match actual feature dimensions
    HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 96))  # Balanced capacity
    NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 2))     # Optimal depth
    DROPOUT_PROB = float(os.environ.get("DROPOUT_PROB", 0.25))  # Proper regularization
    NUM_CLASSES = 1    # Binary classification (up/down)
elif MODEL_VARIANT == "lightweight":
    # Lightweight model for fast inference
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
    INPUT_SIZE = 35
    HIDDEN_SIZE = 32   # Reduced capacity
    NUM_LAYERS = 1     # Shallow network
    DROPOUT_PROB = 0.1 # Lower regularization
    NUM_CLASSES = 1
else:  # baseline
    # Baseline model configuration - deliberately suboptimal for comparison
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.002))  # Higher LR = less stable
    INPUT_SIZE = 35    # Fixed to match actual feature dimensions (was 12)
    HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 32))   # Smaller capacity
    NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 1))     # Shallow network
    DROPOUT_PROB = float(os.environ.get("DROPOUT_PROB", 0.1))  # Minimal regularization
    NUM_CLASSES = 1    # Binary classification (up/down)

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
        logging.info(f"Training data shape: Features {train_features.shape}, Targets {train_targets.shape}")
        logging.info(f"Validation data shape: Features {val_features.shape}, Targets {val_targets.shape}")
        logging.info(f"Test data shape: Features {test_features.shape}, Targets {test_targets.shape}")
        
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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Print a progress bar for training epochs
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)


def evaluate_test_set(model, test_features, test_targets, sequence_length, batch_size):
    """
    Evaluate the trained model on the test set and return comprehensive metrics.
    """
    logging.info("Starting test set evaluation...")
    
    test_dataset = FinancialTimeSeriesDataset(test_features, test_targets, sequence_length=sequence_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    test_batches = 0
    
    all_predictions = []
    all_targets = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(test_dataloader):
            features = features.to('cpu')
            targets = targets.to('cpu').float().unsqueeze(1)
            
            # Check for NaN in test data
            if torch.isnan(features).any() or torch.isnan(targets).any():
                logging.warning(f"NaN detected in test batch {batch_idx}. Skipping batch.")
                continue
            
            outputs = model(features)
            
            # Check for NaN in test outputs
            if torch.isnan(outputs).any():
                logging.warning(f"NaN detected in test model outputs for batch {batch_idx}. Skipping batch.")
                continue
                
            loss = criterion(outputs, targets)
            
            # Check for NaN in test loss
            if torch.isnan(loss):
                logging.warning(f"NaN test loss detected for batch {batch_idx}. Skipping batch.")
                continue
                
            test_loss += loss.item() * features.size(0)
            test_batches += features.size(0)

            # Apply sigmoid for predictions since we're using BCEWithLogitsLoss
            probabilities = torch.sigmoid(outputs)
            predicted_classes = (probabilities > 0.5).float()
            
            total_predictions += targets.size(0)
            correct_predictions += (predicted_classes == targets).sum().item()
            
            # Store for additional metrics
            all_predictions.extend(predicted_classes.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    # Calculate comprehensive metrics
    if test_batches > 0:
        test_loss /= test_batches
        test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Convert to numpy arrays for sklearn metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate additional metrics
        true_positives = np.sum((all_predictions == 1) & (all_targets == 1))
        true_negatives = np.sum((all_predictions == 0) & (all_targets == 0))
        false_positives = np.sum((all_predictions == 1) & (all_targets == 0))
        false_negatives = np.sum((all_predictions == 0) & (all_targets == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Log comprehensive test metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1_score)
        
        logging.info("=" * 60)
        logging.info("TEST SET EVALUATION RESULTS")
        logging.info("=" * 60)
        logging.info(f"Test Loss: {test_loss:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"Test Precision: {precision:.4f}")
        logging.info(f"Test Recall: {recall:.4f}")
        logging.info(f"Test F1-Score: {f1_score:.4f}")
        logging.info(f"True Positives: {true_positives}")
        logging.info(f"True Negatives: {true_negatives}")
        logging.info(f"False Positives: {false_positives}")
        logging.info(f"False Negatives: {false_negatives}")
        logging.info("=" * 60)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1_score
        }
    else:
        logging.error("No valid test batches found!")
        return None


def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        train_targets: np.ndarray
):
    """
    Trains and validates the PyTorch model with detailed epoch-by-epoch progress.
    Includes class weights for handling imbalanced datasets.
    """
    # Compute class weights for imbalanced dataset handling
    unique_classes = np.unique(train_targets)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_targets)
    class_weight_tensor = torch.FloatTensor([class_weights[1]]).to('cpu')  # Weight for positive class
    
    # Update criterion with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor)
    
    logging.info(f"Class distribution: {np.bincount(train_targets.astype(int))}")
    logging.info(f"Applied class weight for positive class: {class_weight_tensor.item():.4f}")
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    logging.info("=" * 80)
    logging.info("STARTING TRAINING")
    logging.info("=" * 80)
    logging.info(f"Total epochs: {epochs}")
    logging.info(f"Training batches per epoch: {len(train_dataloader)}")
    logging.info(f"Validation batches per epoch: {len(val_dataloader)}")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # ===== TRAINING PHASE =====
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        logging.info(f"\nEpoch {epoch + 1}/{epochs}")
        logging.info("-" * 40)
        
        for batch_idx, (features, targets) in enumerate(train_dataloader):
            # Ensure proper tensor shapes and types
            features = features.to('cpu')
            targets = targets.to('cpu').float().unsqueeze(1)

            # Check for NaN in input data
            if torch.isnan(features).any() or torch.isnan(targets).any():
                logging.warning(f"NaN detected in batch {batch_idx} of epoch {epoch}. Skipping batch.")
                continue

            optimizer.zero_grad()
            outputs = model(features)
            
            # Check for NaN in model outputs
            if torch.isnan(outputs).any():
                logging.warning(f"NaN detected in model outputs at batch {batch_idx} of epoch {epoch}. Skipping batch.")
                continue
                
            loss = criterion(outputs, targets)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                logging.warning(f"NaN loss detected at batch {batch_idx} of epoch {epoch}. Skipping batch.")
                continue
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_batches += features.size(0)
            
            # Show progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_loss = train_loss / train_batches if train_batches > 0 else 0.0
                print_progress_bar(
                    batch_idx + 1, 
                    len(train_dataloader), 
                    prefix=f'Training', 
                    suffix=f'Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {current_loss:.4f}'
                )

        # Calculate average training loss
        if train_batches > 0:
            train_loss /= train_batches
        else:
            logging.error(f"No valid training batches in epoch {epoch}")
            train_loss = float('nan')

        print()  # New line after progress bar
        
        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(val_dataloader):
                features = features.to('cpu')
                targets = targets.to('cpu').float().unsqueeze(1)
                
                # Check for NaN in validation data
                if torch.isnan(features).any() or torch.isnan(targets).any():
                    logging.warning(f"NaN detected in validation batch {batch_idx}. Skipping batch.")
                    continue
                
                outputs = model(features)
                
                # Check for NaN in validation outputs
                if torch.isnan(outputs).any():
                    logging.warning(f"NaN detected in validation model outputs for batch {batch_idx}. Skipping batch.")
                    continue
                    
                loss = criterion(outputs, targets)
                
                # Check for NaN in validation loss
                if torch.isnan(loss):
                    logging.warning(f"NaN validation loss detected for batch {batch_idx}. Skipping batch.")
                    continue
                    
                val_loss += loss.item() * features.size(0)
                val_batches += features.size(0)

                # Apply sigmoid for predictions since we're using BCEWithLogitsLoss
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

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log detailed epoch results
        logging.info(f"Epoch {epoch + 1}/{epochs} Results:")
        logging.info(f"  Training Loss:    {train_loss:.6f}")
        logging.info(f"  Validation Loss:  {val_loss:.6f}")
        logging.info(f"  Validation Acc:   {val_accuracy:.6f}")
        logging.info(f"  Epoch Time:       {epoch_time:.2f}s")
        
        # Check for improvement
        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            logging.info(f"  🎯 New best validation loss!")
        
        # Safely log metrics to MLflow
        safe_log_metric("train_loss", train_loss, step=epoch)
        safe_log_metric("val_loss", val_loss, step=epoch)
        safe_log_metric("val_accuracy", val_accuracy, step=epoch)
        safe_log_metric("epoch_time", epoch_time, step=epoch)

    logging.info("=" * 80)
    logging.info(f"TRAINING COMPLETED!")
    logging.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
    logging.info("=" * 80)
    
    return model, best_val_loss, best_epoch


def export_onnx_model(model, sample_input, model_save_dir, model_variant):
    """
    Export trained PyTorch model to ONNX format for Seldon Core serving.
    """
    try:
        onnx_path = os.path.join(model_save_dir, f"stock_predictor_{model_variant}.onnx")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['predictions'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        if os.path.exists(onnx_path):
            logging.info(f"✅ ONNX model exported successfully: {onnx_path}")
            # Log ONNX model to MLflow
            mlflow.log_artifact(onnx_path, artifact_path="onnx_model")
            return onnx_path
        else:
            logging.error("❌ ONNX export failed - file not created")
            return None
            
    except Exception as e:
        logging.error(f"❌ ONNX export failed: {e}")
        return None


def run_training_pipeline():
    global INPUT_SIZE  # Declare global before any use
    
    start_time = datetime.now()
    logging.info("=" * 100)
    logging.info(f"PYTORCH FINANCIAL MODEL TRAINING PIPELINE - {MODEL_VARIANT.upper()} VARIANT")
    logging.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 100)

    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Set experiment name for better organization with variant tracking
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", f"financial-mlops-pytorch-{MODEL_VARIANT}")
    mlflow.set_experiment(experiment_name)
    
    # Start an MLflow run
    run_name = f"pytorch_stock_predictor_{MODEL_VARIANT}_{start_time.strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Log all parameters including model variant
        mlflow.log_param("model_variant", MODEL_VARIANT)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("num_layers", NUM_LAYERS)
        mlflow.log_param("dropout_prob", DROPOUT_PROB)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("device", "cpu")
        mlflow.log_param("start_time", start_time.isoformat())

        # 1. Load processed data
        logging.info("Step 1: Loading processed data...")
        data = load_processed_data(PROCESSED_DATA_DIR)
        train_features = data['train_features']
        train_targets = data['train_targets']
        val_features = data['val_features']
        val_targets = data['val_targets']
        test_features = data['test_features']
        test_targets = data['test_targets']

        # Check for NaN values in loaded data
        if np.isnan(train_features).any():
            logging.error("NaN values found in training features!")
            return
        if np.isnan(train_targets).any():
            logging.error("NaN values found in training targets!")
            return

        # Adjust input size if needed (dynamic adjustment logic retained)
        actual_input_size = train_features.shape[1]
        if actual_input_size != INPUT_SIZE:
            logging.warning(
                f"Configured INPUT_SIZE ({INPUT_SIZE}) does not match actual data features ({actual_input_size}). Adjusting to {actual_input_size}.")
            INPUT_SIZE = actual_input_size

        mlflow.log_param("input_size", INPUT_SIZE)
        
        # 2. Create datasets and dataloaders
        logging.info("Step 2: Creating datasets and dataloaders...")
        train_dataset = FinancialTimeSeriesDataset(train_features, train_targets, sequence_length=SEQUENCE_LENGTH)
        val_dataset = FinancialTimeSeriesDataset(val_features, val_targets, sequence_length=SEQUENCE_LENGTH)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logging.info(f"Training dataset size: {len(train_dataset)} sequences")
        logging.info(f"Validation dataset size: {len(val_dataset)} sequences")

        # 3. Initialize model
        logging.info("Step 3: Initializing model...")
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

        # Log model information
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("total_trainable_parameters", total_params)
        
        logging.info("Model Architecture:")
        logging.info(f"{model}")
        logging.info(f"Total trainable parameters: {total_params:,}")

        # 4. Initialize training components
        logging.info("Step 4: Initializing training components...")
        criterion = nn.BCEWithLogitsLoss()  # Will be updated with class weights in train_model
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

        # 5. Train the model
        logging.info("Step 5: Starting model training...")
        training_start_time = time.time()
        
        trained_model, best_val_loss, best_epoch = train_model(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            criterion,
            EPOCHS,
            train_targets
        )
        
        training_duration = time.time() - training_start_time
        mlflow.log_metric("total_training_time_seconds", training_duration)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        logging.info(f"Total training time: {training_duration:.2f} seconds")

        # 6. Evaluate on test set
        logging.info("Step 6: Evaluating on test set...")
        test_results = evaluate_test_set(
            trained_model, 
            test_features, 
            test_targets, 
            SEQUENCE_LENGTH, 
            BATCH_SIZE
        )

        # 7. Export ONNX model for Seldon Core serving
        logging.info("Step 7: Exporting ONNX model for serving...")
        sample_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_SIZE)
        onnx_path = export_onnx_model(trained_model, sample_input, MODEL_SAVE_DIR, MODEL_VARIANT)

        # 8. Log model to MLflow with proper documentation
        logging.info("Step 8: Logging model to MLflow...")
        try:
            # Create model info dictionary with variant-specific information
            model_info = {
                "model_variant": MODEL_VARIANT,
                "model_type": "LSTM Binary Classifier",
                "task": "Financial Direction Prediction",
                "input_features": INPUT_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout_prob": DROPOUT_PROB,
                "learning_rate": LEARNING_RATE,
                "total_parameters": total_params,
                "training_time_seconds": training_duration,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "test_accuracy": test_results['test_accuracy'] if test_results else None,
                "test_f1_score": test_results['test_f1_score'] if test_results else None,
                "pytorch_version": torch.__version__,
                "trained_on": start_time.isoformat(),
                "onnx_exported": onnx_path is not None,
                "onnx_path": onnx_path
            }
            
            # Log model info as artifact
            model_info_path = f"model_info_{MODEL_VARIANT}.json"
            with open(model_info_path, "w") as f:
                json.dump(model_info, f, indent=2)
            mlflow.log_artifact(model_info_path)
            
            # Log the model with signature and example
            registered_model_name = f"FinancialDirectionPredictor_{MODEL_VARIANT.title()}"
            mlflow.pytorch.log_model(
                pytorch_model=trained_model,
                name=f"stock_predictor_{MODEL_VARIANT}_model",
                registered_model_name=registered_model_name,
                code_paths=["src/models.py"],
                input_example=sample_input.numpy(),
                signature=mlflow.models.infer_signature(sample_input.numpy(), trained_model(sample_input).detach().numpy())
            )
            
            logging.info(f"✅ PyTorch model ({MODEL_VARIANT}) successfully logged and registered with MLflow.")
            
        except Exception as e:
            logging.error(f"❌ Failed to log model to MLflow: {e}")
        
        # Final summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logging.info("=" * 100)
        logging.info(f"TRAINING PIPELINE COMPLETED SUCCESSFULLY! - {MODEL_VARIANT.upper()} VARIANT")
        logging.info("=" * 100)
        logging.info(f"Model Variant: {MODEL_VARIANT}")
        logging.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total duration: {total_duration:.2f} seconds")
        logging.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        if test_results:
            logging.info(f"Final test accuracy: {test_results['test_accuracy']:.4f}")
            logging.info(f"Final test F1-score: {test_results['test_f1_score']:.4f}")
        if onnx_path:
            logging.info(f"ONNX model exported: {onnx_path}")
        logging.info("=" * 100)

    if os.path.exists("training.log"):
        mlflow.log_artifact("training.log")


# --- Main execution ---
if __name__ == "__main__":
    run_training_pipeline()
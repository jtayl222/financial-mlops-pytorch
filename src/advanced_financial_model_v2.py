"""
Advanced Financial Model V2 - Compatible with new ticker-specific feature structure
Integrates with the new financial ML splits and processed datasets
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json
import math
import mlflow
import mlflow.pytorch
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class AdvancedFinancialLSTM(torch.nn.Module):
    """Advanced LSTM for financial time series with enhanced architecture"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout_prob=0.3):
        super(AdvancedFinancialLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer LSTM with dropout
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Feature normalization
        self.feature_norm = torch.nn.LayerNorm(hidden_size)
        
        # Enhanced classifier with residual connections
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size//2, hidden_size//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size//4, 1)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.data.dim() > 1:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                torch.nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out  # Residual connection
        
        # Use the last output for classification
        combined = combined[:, -1, :]
        
        # Normalize features
        combined = self.feature_norm(combined)
        
        # Classify
        output = self.classifier(combined)
        
        return output

class FinancialTimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset compatible with processed PyTorch datasets"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_processed_datasets(processed_data_dir):
    """Load processed sequence data following shape contract (same as baseline)"""
    
    logging.info(f"Loading processed sequence data from {processed_data_dir}")
    
    try:
        # Load shape contract metadata (same format as baseline)
        import json
        metadata_path = os.path.join(processed_data_dir, 'shape_contract_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Loaded shape contract: {metadata}")
        else:
            # Fallback metadata
            metadata = {'sequence_length': 10, 'n_features': 205, 'input_shape': [10, 205]}
            logging.warning("Shape contract metadata not found, using defaults")
        
        # Load sequence arrays (same as baseline)
        train_sequences = np.load(os.path.join(processed_data_dir, 'train_sequences.npy'))
        train_targets = np.load(os.path.join(processed_data_dir, 'train_sequence_targets.npy'))
        val_sequences = np.load(os.path.join(processed_data_dir, 'val_sequences.npy'))
        val_targets = np.load(os.path.join(processed_data_dir, 'val_sequence_targets.npy'))
        test_sequences = np.load(os.path.join(processed_data_dir, 'test_sequences.npy'))
        test_targets = np.load(os.path.join(processed_data_dir, 'test_sequence_targets.npy'))
        
        # Create datasets (same as baseline)
        train_dataset = FinancialTimeSeriesDataset(train_sequences, train_targets)
        val_dataset = FinancialTimeSeriesDataset(val_sequences, val_targets)
        test_dataset = FinancialTimeSeriesDataset(test_sequences, test_targets)
        
        logging.info("Successfully loaded sequence datasets following shape contract.")
        logging.info(f"Training dataset: {len(train_dataset)} sequences")
        logging.info(f"Validation dataset: {len(val_dataset)} sequences")
        logging.info(f"Test dataset: {len(test_dataset)} sequences")
        
        # Verify shape contract compliance (same as baseline)
        sample_features, sample_target = train_dataset[0]
        expected_shape = (metadata['sequence_length'], metadata['n_features'])
        actual_shape = sample_features.shape
        
        if actual_shape == expected_shape:
            logging.info(f"✅ Shape contract verified: {actual_shape}")
        else:
            logging.error(f"❌ Shape contract violation: expected {expected_shape}, got {actual_shape}")
        
        # Return metadata in advanced model format
        advanced_metadata = {
            'n_features': metadata['n_features'],
            'sequence_length': metadata['sequence_length'],
            'input_size': metadata['n_features'],
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'ticker_names': ['multi-ticker-shape-contract'],
            'feature_engineering': 'shape_contract_compliant'
        }
        
        logging.info(f"Advanced model metadata: {advanced_metadata}")
        return train_dataset, val_dataset, test_dataset, advanced_metadata
        
    except FileNotFoundError as e:
        logging.error(f"Missing processed data files in {processed_data_dir}. Error: {e}")
        raise

def train_advanced_model():
    """Train the advanced model with enhanced features"""
    
    logging.info("Starting advanced model training...")
    
    # Configuration
    PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "/mnt/financial-data/processed")
    SCALER_DIR = os.environ.get("SCALER_DIR", "/mnt/financial-features/scalers")
    MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_DIR", "/mnt/shared-models")
    
    # Create directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Load processed datasets
    train_dataset, val_dataset, test_dataset, metadata = load_processed_datasets(PROCESSED_DATA_DIR)
    
    # Extract model parameters from metadata
    input_size = metadata['n_features'] 
    sequence_length = metadata['sequence_length']
    
    logging.info(f"Using input_size: {input_size}, sequence_length: {sequence_length}")
    logging.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedFinancialLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout_prob=0.3
    ).to(device)
    
    # Advanced training configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Class imbalance handling
    # Calculate positive class weight from training data
    total_samples = len(train_dataset)
    positive_samples = sum(1 for _, target in train_dataset if target == 1)
    pos_weight = torch.tensor([total_samples / (2 * positive_samples)]).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Positive weight: {pos_weight.item():.3f}")
    
    # Set experiment name for better organization with variant tracking
    MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "advanced")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", f"seldon-system-{MODEL_VARIANT}")
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name="advanced_financial_model_v2"):
        # Log parameters
        mlflow.log_param("model_variant", MODEL_VARIANT)
        mlflow.log_param("model_type", "AdvancedFinancialLSTM")
        mlflow.log_param("hidden_size", 128)
        mlflow.log_param("num_layers", 3)
        mlflow.log_param("dropout_prob", 0.3)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("pos_weight", pos_weight.item())
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        # Log ticker information if available
        ticker_names = metadata.get('ticker_names', ['unknown'])
        if isinstance(ticker_names, list):
            ticker_names_str = ",".join(str(name) for name in ticker_names)
        else:
            ticker_names_str = str(ticker_names)
        
        mlflow.log_param("n_tickers", len(ticker_names) if isinstance(ticker_names, list) else 1)
        mlflow.log_param("ticker_names", ticker_names_str)
        
        # Training loop
        best_val_acc = 0.0
        patience = 20
        patience_counter = 0
        epochs = 100
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                
                # Check for NaN in input data
                if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                    logging.warning(f"NaN detected in batch {batch_idx} of epoch {epoch}. Skipping batch.")
                    continue
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                
                # Check for NaN in model outputs
                if torch.isnan(outputs).any():
                    logging.warning(f"NaN detected in model outputs at batch {batch_idx} of epoch {epoch}. Skipping batch.")
                    continue
                
                loss = criterion(outputs, batch_y)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    logging.warning(f"NaN loss detected at batch {batch_idx} of epoch {epoch}. Skipping batch.")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                    
                    # Check for NaN in validation data
                    if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                        logging.warning(f"NaN detected in validation batch {batch_idx}. Skipping batch.")
                        continue
                    
                    outputs = model(batch_x).squeeze()
                    
                    # Check for NaN in validation outputs
                    if torch.isnan(outputs).any():
                        logging.warning(f"NaN detected in validation model outputs for batch {batch_idx}. Skipping batch.")
                        continue
                    
                    loss = criterion(outputs, batch_y)
                    
                    # Check for NaN in validation loss
                    if torch.isnan(loss):
                        logging.warning(f"NaN validation loss detected for batch {batch_idx}. Skipping batch.")
                        continue
                    
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_advanced_model.pth'))
                logging.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 5 == 0:
                logging.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Log metrics to MLflow safely
            safe_log_metric("train_loss", train_loss / len(train_loader), step=epoch)
            safe_log_metric("train_accuracy", train_acc, step=epoch)
            safe_log_metric("val_loss", val_loss / len(val_loader), step=epoch)
            safe_log_metric("val_accuracy", val_acc, step=epoch)
            safe_log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'best_advanced_model.pth')))
        model.eval()
        
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                outputs = model(batch_x).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        
        # Calculate additional metrics for multiclass
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Log final results
        logging.info(f"\nFinal Results:")
        logging.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        
        # Log final metrics to MLflow safely
        safe_log_metric("test_accuracy", test_acc, step=0)
        safe_log_metric("test_precision", precision, step=0)
        safe_log_metric("test_recall", recall, step=0)
        safe_log_metric("test_f1_score", f1, step=0)
        safe_log_metric("best_val_accuracy", best_val_acc, step=0)
        
        # Save model artifacts with consistent format (same as baseline model)
        # Create sample input for MLflow signature
        sample_input = torch.randn(1, sequence_length, input_size)
        model_cpu = model.cpu()
        sample_input_cpu = sample_input.cpu()
        
        # Create input example and signature for consistent input format
        with torch.no_grad():
            sample_output = model_cpu(sample_input_cpu)
        
        registered_model_name = f"FinancialDirectionPredictor_{MODEL_VARIANT.title()}"
        mlflow.pytorch.log_model(
            pytorch_model=model_cpu,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=sample_input_cpu.numpy(),
            signature=mlflow.models.infer_signature(sample_input_cpu.numpy(), sample_output.detach().numpy())
        )
        
        # Save results
        results = {
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model_params': sum(p.numel() for p in model.parameters()),
            'n_features': input_size,
            'sequence_length': sequence_length,
            'n_tickers': len(ticker_names),
            'ticker_names': ticker_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(MODEL_SAVE_DIR, 'advanced_model_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info("Training completed successfully!")
        
        return results

if __name__ == "__main__":
    train_advanced_model()
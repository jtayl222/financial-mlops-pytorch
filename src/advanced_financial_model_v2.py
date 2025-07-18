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
import mlflow
import mlflow.pytorch
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            if 'weight' in name:
                if 'lstm' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                else:
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
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_processed_datasets(processed_data_dir):
    """Load the processed datasets - use same approach as regular training"""
    
    logging.info(f"Loading processed data from {processed_data_dir}")
    
    # Load the combined CSV file that regular training creates
    combined_csv_path = os.path.join(processed_data_dir, 'combined_features.csv')
    if not os.path.exists(combined_csv_path):
        raise FileNotFoundError(f"Combined features file not found: {combined_csv_path}")
    
    df = pd.read_csv(combined_csv_path, index_col='Date', parse_dates=True)
    logging.info(f"Loaded combined data: {len(df)} rows, {len(df.columns)} columns")
    logging.info(f"Columns: {list(df.columns)}")
    
    # Use financial ML splits (same as regular training)
    from financial_ml_splits import create_financial_splits
    
    # Create proper financial splits
    train_df, val_df, test_df = create_financial_splits(
        df, 
        train_ratio=0.7,
        val_ratio=0.15,
        purge_period=30  # 30-day purge gap
    )
    
    logging.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    sequence_length = 10
    train_dataset = FinancialTimeSeriesDataset(train_df, sequence_length)
    val_dataset = FinancialTimeSeriesDataset(val_df, sequence_length)
    test_dataset = FinancialTimeSeriesDataset(test_df, sequence_length)
    
    # Create metadata
    feature_cols = [col for col in df.columns if not col.startswith('Target_')]
    metadata = {
        'n_features': len(feature_cols),
        'sequence_length': sequence_length,
        'ticker_names': list(set([col.split('_')[-1] for col in feature_cols if '_' in col])),
        'feature_names': feature_cols
    }
    
    logging.info(f"Features: {metadata['n_features']}, Sequence length: {metadata['sequence_length']}")
    
    return train_dataset, val_dataset, test_dataset, metadata

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
    
    # Model parameters
    input_size = metadata['n_features']
    sequence_length = metadata['sequence_length']
    
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
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Class imbalance handling
    # Calculate positive class weight from training data
    total_samples = len(train_dataset)
    positive_samples = sum(1 for _, target in train_dataset if target == 1)
    pos_weight = torch.tensor([total_samples / (2 * positive_samples)]).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Positive weight: {pos_weight.item():.3f}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="advanced_financial_model_v2"):
        # Log parameters
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
        mlflow.log_param("n_tickers", len(metadata['ticker_names']))
        mlflow.log_param("ticker_names", ",".join(metadata['ticker_names']))
        
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
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            
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
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
        
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
        
        test_acc = test_correct / test_total
        
        # Calculate additional metrics
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        # Log final results
        logging.info(f"\nFinal Results:")
        logging.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        
        # Log final metrics to MLflow
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        
        # Save model artifacts
        mlflow.pytorch.log_model(model, "model")
        
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
            'n_tickers': len(metadata['ticker_names']),
            'ticker_names': metadata['ticker_names'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(MODEL_SAVE_DIR, 'advanced_model_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info("Training completed successfully!")
        
        return results

if __name__ == "__main__":
    train_advanced_model()
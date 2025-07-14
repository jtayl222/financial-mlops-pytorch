"""
Targeted improvements to get closer to 78-82% accuracy
Focus on the most impactful changes based on financial ML research
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from feature_engineering_pytorch import FinancialTimeSeriesDataset
from train_pytorch_model import load_processed_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_advanced_features(data_dir):
    """Create advanced features that are more predictive for financial data"""
    
    logging.info("Creating advanced features...")
    
    # Load raw data to engineer better features
    raw_data_dir = "data/raw"
    raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    
    all_data = []
    for file in raw_files:
        ticker = file.split('_')[0]
        df = pd.read_csv(os.path.join(raw_data_dir, file), index_col='Date', parse_dates=True)
        df = df.sort_index()
        
        close_col = f'Close_{ticker}'
        high_col = f'High_{ticker}'
        low_col = f'Low_{ticker}'
        open_col = f'Open_{ticker}'
        volume_col = f'Volume_{ticker}'
        
        # 1. Better momentum indicators
        # Price momentum at multiple timeframes
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df[close_col].pct_change(period)
        
        # 2. Volatility features (critical for financial prediction)
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df[close_col].rolling(window).std()
            df[f'vol_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(50).mean()
        
        # 3. Volume analysis (much more predictive)
        df['volume_sma'] = df[volume_col].rolling(20).mean()
        df['volume_ratio'] = df[volume_col] / df['volume_sma']
        df['price_volume'] = df[close_col] * df[volume_col]  # Money flow proxy
        
        # 4. Mean reversion indicators
        for window in [10, 20, 50]:
            sma = df[close_col].rolling(window).mean()
            df[f'price_vs_sma_{window}'] = (df[close_col] - sma) / sma
            
        # 5. High-low analysis
        df['high_low_ratio'] = (df[high_col] - df[low_col]) / df[close_col]
        df['close_vs_high'] = (df[close_col] - df[high_col]) / df[high_col]
        df['close_vs_low'] = (df[close_col] - df[low_col]) / df[low_col]
        
        # 6. Gap analysis
        df['gap'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)
        df['intraday_return'] = (df[close_col] - df[open_col]) / df[open_col]
        
        # 7. RSI with multiple timeframes
        for period in [9, 14, 21]:
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 8. MACD-like indicators
        ema_12 = df[close_col].ewm(span=12).mean()
        ema_26 = df[close_col].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 9. Bollinger Bands
        sma_20 = df[close_col].rolling(20).mean()
        std_20 = df[close_col].rolling(20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 10. Market microstructure
        df['vwap'] = (df[close_col] * df[volume_col]).rolling(20).sum() / df[volume_col].rolling(20).sum()
        df['price_vs_vwap'] = df[close_col] / df['vwap']
        
        # 11. Enhanced target (more predictable)
        # Instead of simple up/down, use threshold-based target
        future_return = df[close_col].pct_change(1).shift(-1)
        threshold = 0.003  # 0.3% threshold - more realistic for daily prediction
        
        # Create multi-class target: strong_down, weak_down, weak_up, strong_up
        df['target_class'] = 1  # neutral/weak_up (baseline)
        df.loc[future_return > threshold, 'target_class'] = 1  # strong_up
        df.loc[future_return < -threshold, 'target_class'] = 0  # strong_down
        
        # Binary version (simplified)
        df['Target'] = (future_return > 0).astype(float)
        
        all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, axis=0).sort_index()
    
    # Drop original price columns (keep engineered features)
    feature_cols = [col for col in combined_df.columns if not any(
        ticker in col for ticker in ['AAPL', 'MSFT']
    ) or col in ['Target']]
    feature_cols = [col for col in feature_cols if col != 'Target']
    
    # Keep only engineered features + target
    features_df = combined_df[feature_cols + ['Target']].copy()
    
    # Drop NaN rows
    features_df = features_df.dropna()
    
    logging.info(f"Advanced features created. Shape: {features_df.shape}")
    logging.info(f"Features: {len(feature_cols)}")
    
    return features_df

class FinancialLSTM(torch.nn.Module):
    """Specialized LSTM for financial time series with domain-specific improvements"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_prob=0.2):
        super(FinancialLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-scale LSTM processing
        self.lstm_short = torch.nn.LSTM(input_size, hidden_size//2, 1, 
                                       batch_first=True, dropout=dropout_prob)
        self.lstm_long = torch.nn.LSTM(input_size, hidden_size//2, 2, 
                                      batch_first=True, dropout=dropout_prob)
        
        # Feature processing
        self.feature_norm = torch.nn.LayerNorm(hidden_size)
        
        # Decision layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size//2, hidden_size//4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_size//4, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-scale processing
        short_out, _ = self.lstm_short(x)
        long_out, _ = self.lstm_long(x)
        
        # Combine outputs
        combined = torch.cat([short_out[:, -1, :], long_out[:, -1, :]], dim=1)
        
        # Normalize
        combined = self.feature_norm(combined)
        
        # Classify
        output = self.classifier(combined)
        
        return output

def improved_training_with_features():
    """Train model with advanced features and better regularization"""
    
    logging.info("Starting training with advanced features...")
    
    # Setup MLflow experiment
    from mlflow_utils import setup_mlflow_experiment, create_mlflow_run, log_training_params, log_training_metrics, log_model_with_artifacts, log_training_artifacts
    
    experiment_name = setup_mlflow_experiment("targeted-improvements")
    
    # Create advanced features
    features_df = create_advanced_features("data/processed")
    
    # Split data
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)
    
    train_df = features_df[:train_size]
    val_df = features_df[train_size:train_size + val_size]
    test_df = features_df[train_size + val_size:]
    
    # Prepare features and targets
    feature_cols = [col for col in features_df.columns if col != 'Target']
    
    # Normalize features
    scaler = StandardScaler()
    
    train_features = scaler.fit_transform(train_df[feature_cols])
    val_features = scaler.transform(val_df[feature_cols])
    test_features = scaler.transform(test_df[feature_cols])
    
    train_targets = train_df['Target'].values
    val_targets = val_df['Target'].values
    test_targets = test_df['Target'].values
    
    logging.info(f"Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")
    logging.info(f"Features: {len(feature_cols)}")
    
    # Create sequences
    sequence_length = 15  # Shorter sequences for better generalization
    
    def create_sequences(features, targets, seq_len):
        X, y = [], []
        for i in range(len(features) - seq_len + 1):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len-1])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_features, train_targets, sequence_length)
    X_val, y_val = create_sequences(val_features, val_targets, sequence_length)
    X_test, y_test = create_sequences(test_features, test_targets, sequence_length)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_size = train_features.shape[1]
    model = FinancialLSTM(input_size, hidden_size=96, num_layers=2, dropout_prob=0.3)
    
    # Training setup
    from device_utils import get_device
    device = get_device()
    model.to(device)
    
    # Better optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Weighted loss for class imbalance
    pos_weight = torch.tensor([len(y_train) / (2 * y_train.sum())]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Adjust model for logits
    model.classifier[-1] = torch.nn.Identity()  # Remove sigmoid, use with BCEWithLogitsLoss
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Positive weight: {pos_weight.item():.3f}")
    
    # Start MLflow run
    with create_mlflow_run("targeted_improvements", "advanced"):
        # Log training configuration
        config = {
            "model_type": "FinancialLSTM",
            "input_size": input_size,
            "hidden_size": 96,
            "num_layers": 2,
            "dropout_prob": 0.3,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "sequence_length": 15,
            "num_features": len(feature_cols),
            "epochs": 100,
            "patience": 15
        }
        log_training_params(config)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
        
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step()
            
            # Early stopping based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_targeted_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                
            # Log metrics to MLflow
            log_training_metrics({
                "train_loss": train_loss / len(train_loader),
                "train_accuracy": train_acc,
                "val_loss": val_loss / len(val_loader),
                "val_accuracy": val_acc
            }, step=epoch)
        
        # Load best model and test
        model.load_state_dict(torch.load('best_targeted_model.pth'))
        model.eval()
    
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
        test_acc = test_correct / test_total
        
        # Calculate additional metrics
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        logging.info(f"\nFinal Results:")
        logging.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        
        # Log final test metrics
        log_training_metrics({
            "test_accuracy": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
            "best_val_accuracy": best_val_acc
        })
        
        # Save and log model with artifacts
        sample_input = torch.randn(1, 15, len(feature_cols)).to(device)
        results = {
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_val_acc': best_val_acc,
            'num_features': len(feature_cols),
            'model_params': sum(p.numel() for p in model.parameters()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log model to MLflow
        log_model_with_artifacts(model, sample_input, "targeted", results)
        
        # Log training artifacts
        log_training_artifacts()
        
        # Save local results file
        with open('targeted_improvement_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

if __name__ == "__main__":
    improved_training_with_features()
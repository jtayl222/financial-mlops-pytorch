"""
Simple improvements script that works with existing processed data
Focus on model architecture and training improvements without requiring new dependencies
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from feature_engineering_pytorch import FinancialTimeSeriesDataset
from train_pytorch_model import load_processed_data
from models import StockPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedStockPredictor(torch.nn.Module):
    """Improved version of the stock predictor with better architecture"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1, dropout_prob=0.3):
        super(ImprovedStockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM for better context
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                 batch_first=True, dropout=dropout_prob, bidirectional=True)
        
        # Additional layers for better representation
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = torch.nn.Linear(hidden_size // 2, num_classes)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size * 2)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Additional processing layers
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

def improved_training_loop(model, train_loader, val_loader, device, config):
    """Improved training loop with better optimization"""
    
    # Better optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Loss function
    criterion = torch.nn.BCELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_improved_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def run_improved_training():
    """Run training with improved model and training strategy"""
    
    logging.info("Starting improved training...")
    
    # Configuration
    model_variant = os.environ.get("MODEL_VARIANT", "enhanced")
    
    if model_variant == "enhanced":
        config = {
            'batch_size': 128,
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'patience': 20,
            'sequence_length': 20,  # Increased from 10
            'hidden_size': 128,     # Increased from 32/64
            'num_layers': 3,        # Increased from 1/2
            'dropout_prob': 0.3     # Increased regularization
        }
    else:  # baseline
        config = {
            'batch_size': 64,
            'epochs': 80,
            'learning_rate': 0.001,
            'weight_decay': 1e-6,
            'patience': 15,
            'sequence_length': 10,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout_prob': 0.2
        }
    
    # Load processed data
    processed_data_dir = os.getenv("PROCESSED_DATA_DIR", "/mnt/shared-data/processed")
    if not os.path.exists(processed_data_dir):
        processed_data_dir = "data/processed"
    
    if not os.path.exists(processed_data_dir):
        logging.error(f"Processed data directory not found: {processed_data_dir}")
        logging.info("Please run feature engineering first or set PROCESSED_DATA_DIR")
        return
    
    data = load_processed_data(processed_data_dir)
    
    # Create datasets
    sequence_length = config['sequence_length']
    train_dataset = FinancialTimeSeriesDataset(
        data['train_features'], data['train_targets'], sequence_length
    )
    val_dataset = FinancialTimeSeriesDataset(
        data['val_features'], data['val_targets'], sequence_length
    )
    test_dataset = FinancialTimeSeriesDataset(
        data['test_features'], data['test_targets'], sequence_length
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    # Create improved model
    input_size = data['train_features'].shape[1]
    
    if model_variant == "enhanced":
        model = ImprovedStockPredictor(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout_prob=config['dropout_prob']
        )
    else:
        # Use original model for baseline
        model = StockPredictor(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout_prob=config['dropout_prob']
        )
    
    # Setup device
    from device_utils import get_device
    device = get_device()
    model.to(device)
    
    logging.info(f"Model variant: {model_variant}")
    logging.info(f"Input size: {input_size}")
    logging.info(f"Training on: {device}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    training_history = improved_training_loop(model, train_loader, val_loader, device, config)
    
    # Load best model
    model.load_state_dict(torch.load('best_improved_model.pth'))
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs.squeeze(), targets)
            
            test_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    
    logging.info(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    logging.info(f"Final Test Loss: {test_loss:.4f}")
    
    # Save results
    results = {
        'model_variant': model_variant,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'training_history': training_history,
        'config': config,
        'input_size': input_size,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f'improved_results_{model_variant}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("Improved training completed successfully!")
    
    return results

def compare_models():
    """Compare baseline vs enhanced models"""
    
    logging.info("Running model comparison...")
    
    results = {}
    
    # Test baseline
    logging.info("Testing baseline model...")
    os.environ['MODEL_VARIANT'] = 'baseline'
    baseline_result = run_improved_training()
    results['baseline'] = baseline_result
    
    # Test enhanced
    logging.info("Testing enhanced model...")
    os.environ['MODEL_VARIANT'] = 'enhanced'
    enhanced_result = run_improved_training()
    results['enhanced'] = enhanced_result
    
    # Print comparison
    logging.info("\n" + "="*50)
    logging.info("MODEL COMPARISON RESULTS")
    logging.info("="*50)
    logging.info(f"Baseline Accuracy: {baseline_result['test_accuracy']:.4f} ({baseline_result['test_accuracy']*100:.1f}%)")
    logging.info(f"Enhanced Accuracy: {enhanced_result['test_accuracy']:.4f} ({enhanced_result['test_accuracy']*100:.1f}%)")
    logging.info(f"Improvement: {(enhanced_result['test_accuracy'] - baseline_result['test_accuracy'])*100:.1f} percentage points")
    logging.info("="*50)
    
    # Save comparison
    with open('model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_models()
    else:
        run_improved_training()
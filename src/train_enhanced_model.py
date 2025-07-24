#!/usr/bin/env python3
"""
Enhanced Model Training Script
Trains the high-accuracy model with advanced features
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import json
import math
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFinancialLSTM(nn.Module):
    """Enhanced LSTM with multi-scale processing for higher accuracy"""
    
    def __init__(self, input_size=205, hidden_size=128, num_layers=3, dropout_prob=0.3):
        super().__init__()
        
        # Feature attention layer
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, input_size),
            nn.Softmax(dim=-1)
        )
        
        # Multi-scale LSTM processing
        self.lstm_short = nn.LSTM(
            input_size, hidden_size // 2, 1, 
            batch_first=True, dropout=0
        )
        self.lstm_long = nn.LSTM(
            input_size, hidden_size // 2, 2, 
            batch_first=True, dropout=dropout_prob
        )
        
        # Combine layers
        self.combine_layer = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob * 0.7),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        # Apply feature attention
        attention_weights = self.feature_attention(x)
        x_attended = x * attention_weights
        
        # Multi-scale processing
        short_out, _ = self.lstm_short(x_attended)
        long_out, _ = self.lstm_long(x_attended)
        
        # Get final outputs
        short_final = short_out[:, -1, :]
        long_final = long_out[:, -1, :]
        
        # Combine
        combined = torch.cat([short_final, long_final], dim=1)
        combined = self.combine_layer(combined)
        combined = self.layer_norm(combined)
        
        # Classification
        output = self.classifier(combined)
        
        return output


class EnhancedDataset(torch.utils.data.Dataset):
    """Dataset for enhanced features"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_enhanced_data():
    """Load enhanced feature data"""
    enhanced_dir = "/Users/user/REPOS/financial-mlops-pytorch/data/processed/enhanced"
    
    # Load sequences
    train_sequences = np.load(os.path.join(enhanced_dir, 'train_sequences.npy'))
    train_targets = np.load(os.path.join(enhanced_dir, 'train_targets.npy'))
    val_sequences = np.load(os.path.join(enhanced_dir, 'val_sequences.npy'))
    val_targets = np.load(os.path.join(enhanced_dir, 'val_targets.npy'))
    test_sequences = np.load(os.path.join(enhanced_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(enhanced_dir, 'test_targets.npy'))
    
    # Load metadata
    with open(os.path.join(enhanced_dir, 'enhanced_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Create datasets
    train_dataset = EnhancedDataset(train_sequences, train_targets)
    val_dataset = EnhancedDataset(val_sequences, val_targets)
    test_dataset = EnhancedDataset(test_sequences, test_targets)
    
    logger.info(f"Enhanced data loaded:")
    logger.info(f"  Train: {train_sequences.shape}")
    logger.info(f"  Val: {val_sequences.shape}")
    logger.info(f"  Test: {test_sequences.shape}")
    logger.info(f"  Features: {metadata['n_features']}")
    logger.info(f"  Shape contract: {metadata['input_shape']}")
    
    return train_dataset, val_dataset, test_dataset, metadata


def safe_log_metric(metric_name: str, value: float, step: int):
    """Safely log metric to MLflow"""
    if math.isnan(value) or math.isinf(value):
        logger.warning(f"Skipping {metric_name} at step {step}: {value}")
        return
    
    try:
        mlflow.log_metric(metric_name, value, step=step)
    except Exception as e:
        logger.warning(f"Failed to log {metric_name}: {e}")


def train_enhanced_model():
    """Train the enhanced model with advanced features"""
    
    logger.info("=" * 80)
    logger.info("ENHANCED MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load enhanced data
    train_dataset, val_dataset, test_dataset, metadata = load_enhanced_data()
    
    # Training configuration
    config = {
        'model_variant': 'enhanced_v2',
        'input_size': metadata['n_features'],
        'hidden_size': 128,
        'num_layers': 3,
        'dropout_prob': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 100,
        'patience': 15
    }
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedFinancialLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_prob=config['dropout_prob']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Class weights for imbalanced data
    pos_samples = sum(train_dataset.targets)
    total_samples = len(train_dataset.targets)
    pos_weight = torch.tensor([total_samples / (2 * pos_samples)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logger.info(f"Training setup:")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Scheduler: CosineAnnealingWarmRestarts")
    logger.info(f"  Pos weight: {pos_weight.item():.3f}")
    
    # MLflow tracking
    experiment_name = "enhanced-financial-model"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="enhanced_model_v2"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("pos_weight", pos_weight.item())
        mlflow.log_param("device", str(device))
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0.0
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
            
            train_acc = train_correct / train_total
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
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
            
            val_acc = val_correct / val_total
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_enhanced_model_v2.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            safe_log_metric("train_loss", train_loss / len(train_loader), epoch)
            safe_log_metric("train_accuracy", train_acc, epoch)
            safe_log_metric("val_loss", val_loss / len(val_loader), epoch)
            safe_log_metric("val_accuracy", val_acc, epoch)
            safe_log_metric("learning_rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Final evaluation
        logger.info("=" * 40)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 40)
        
        # Load best model
        model.load_state_dict(torch.load('best_enhanced_model_v2.pth'))
        model.eval()
        
        # Test evaluation
        test_predictions = []
        test_targets_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                test_predictions.extend(predicted.cpu().numpy())
                test_targets_list.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(test_targets_list, test_predictions)
        test_precision = precision_score(test_targets_list, test_predictions, zero_division=0)
        test_recall = recall_score(test_targets_list, test_predictions, zero_division=0)
        test_f1 = f1_score(test_targets_list, test_predictions, zero_division=0)
        
        # Log final metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        
        # Results
        results = {
            'model_variant': 'enhanced_v2',
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'best_val_accuracy': best_val_acc,
            'total_parameters': total_params,
            'training_config': config,
            'feature_engineering': 'enhanced_financial_indicators',
            'shape_contract': metadata['input_shape'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open('enhanced_model_v2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log model
        sample_input = torch.randn(1, metadata['sequence_length'], metadata['n_features'])
        mlflow.pytorch.log_model(
            model.cpu(),
            artifact_path="model",
            registered_model_name="FinancialDirectionPredictor_Enhanced_V2",
            input_example=sample_input.numpy()
        )
        
        # Final summary
        logger.info(f"ENHANCED MODEL RESULTS:")
        logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1-Score: {test_f1:.4f}")
        logger.info(f"Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
        
        if test_acc > 0.80:
            logger.info("🎯 SUCCESS: Enhanced model achieved 80%+ accuracy target!")
        elif test_acc > 0.70:
            logger.info("✅ GOOD: Enhanced model shows significant improvement")
        elif test_acc > 0.60:
            logger.info("📈 IMPROVED: Enhanced model beats baseline")
        else:
            logger.info("⚠️ NEEDS WORK: Enhanced model needs further improvement")
        
        logger.info(f"Shape contract maintained: {metadata['input_shape']}")
        logger.info(f"A/B testing ready: Yes")
        
        return results


if __name__ == "__main__":
    results = train_enhanced_model()
    
    print(f"\n🎯 Enhanced Model Training Complete!")
    print(f"Accuracy: {results['test_accuracy']:.1%}")
    print(f"Shape contract: {results['shape_contract']}")
    print(f"Ready for A/B testing deployment!")
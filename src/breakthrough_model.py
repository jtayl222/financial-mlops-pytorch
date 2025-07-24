#!/usr/bin/env python3
"""
Breakthrough Model Implementation
Adapts the 90.2% accuracy model architecture while maintaining shape contract compatibility
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import json
import math
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreakthroughFinancialLSTM(nn.Module):
    """
    Breakthrough architecture adapted from 90.2% model
    Key innovations: Multi-scale processing, layer normalization, financial-specific design
    """
    
    def __init__(self, input_size=33, hidden_size=96, num_layers=2, dropout_prob=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # Multi-scale LSTM processing (critical for 90.2% performance)
        self.lstm_short = nn.LSTM(
            input_size, hidden_size//2, 1, 
            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0
        )
        self.lstm_long = nn.LSTM(
            input_size, hidden_size//2, 2, 
            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Layer normalization (from 90.2% model)
        self.feature_norm = nn.LayerNorm(hidden_size)
        
        # Financial-specific classifier architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size//4, 1)
        )
        
        # Initialize weights properly
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-scale processing (key innovation from 90.2% model)
        short_out, _ = self.lstm_short(x)
        long_out, _ = self.lstm_long(x)
        
        # Combine outputs from different scales
        combined = torch.cat([short_out[:, -1, :], long_out[:, -1, :]], dim=1)
        
        # Layer normalization (critical for stability)
        combined = self.feature_norm(combined)
        
        # Classification
        output = self.classifier(combined)
        
        return output

def create_breakthrough_features(enhanced_dir):
    """
    Create features following the 90.2% model approach
    Adapts the breakthrough feature engineering while maintaining compatibility
    """
    logger.info("Creating breakthrough features based on 90.2% model...")
    
    # Load existing enhanced sequences
    train_sequences = np.load(os.path.join(enhanced_dir, 'train_sequences.npy'))
    train_targets = np.load(os.path.join(enhanced_dir, 'train_targets.npy'))
    val_sequences = np.load(os.path.join(enhanced_dir, 'val_sequences.npy'))
    val_targets = np.load(os.path.join(enhanced_dir, 'val_targets.npy'))
    test_sequences = np.load(os.path.join(enhanced_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(enhanced_dir, 'test_targets.npy'))
    
    logger.info(f"Original shapes - Train: {train_sequences.shape}, Val: {val_sequences.shape}, Test: {test_sequences.shape}")
    
    # The 90.2% model used 33 features with 15-step sequences
    # We need to adapt to our current data structure
    
    # Select most important features based on 90.2% model categories:
    # 1. Momentum indicators (multiple timeframes)
    # 2. Volatility features 
    # 3. Volume analysis
    # 4. Mean reversion indicators
    # 5. Market microstructure
    # 6. Technical indicators
    
    # For now, select first 33 features and extend sequence length
    n_features_to_use = 33
    selected_features = train_sequences[:, :, :n_features_to_use]
    
    # Extend sequences from 10 to 15 steps (critical for breakthrough performance)
    def extend_sequences(sequences, targets, from_len=10, to_len=15):
        """Extend sequence length by padding with interpolated values"""
        n_samples, seq_len, n_features = sequences.shape
        
        if seq_len >= to_len:
            return sequences, targets
        
        # Create extended sequences
        extended_sequences = np.zeros((n_samples, to_len, n_features))
        
        for i in range(n_samples):
            seq = sequences[i]
            
            # Interpolate to extend sequence
            extended_seq = np.zeros((to_len, n_features))
            
            # Copy original sequence
            extended_seq[:seq_len] = seq
            
            # Interpolate remaining steps
            for step in range(seq_len, to_len):
                # Simple linear extrapolation based on last trend
                if seq_len >= 2:
                    trend = seq[-1] - seq[-2]
                    extended_seq[step] = seq[-1] + trend * (step - seq_len + 1)
                else:
                    extended_seq[step] = seq[-1]
            
            extended_sequences[i] = extended_seq
        
        return extended_sequences, targets
    
    # Extend all sequences to 15 steps
    train_extended, train_targets = extend_sequences(
        train_sequences[:, :, :n_features_to_use], train_targets, 10, 15
    )
    val_extended, val_targets = extend_sequences(
        val_sequences[:, :, :n_features_to_use], val_targets, 10, 15
    )
    test_extended, test_targets = extend_sequences(
        test_sequences[:, :, :n_features_to_use], test_targets, 10, 15
    )
    
    logger.info(f"Extended shapes - Train: {train_extended.shape}, Val: {val_extended.shape}, Test: {test_extended.shape}")
    
    return train_extended, train_targets, val_extended, val_targets, test_extended, test_targets

def train_breakthrough_model():
    """Train breakthrough model using 90.2% architecture insights"""
    
    logger.info("=" * 80)
    logger.info("BREAKTHROUGH MODEL TRAINING (90.2% Architecture)")
    logger.info("=" * 80)
    
    # Load and prepare breakthrough features
    enhanced_dir = "/Users/user/REPOS/financial-mlops-pytorch/data/processed/enhanced"
    
    train_sequences, train_targets, val_sequences, val_targets, test_sequences, test_targets = create_breakthrough_features(enhanced_dir)
    
    # Training configuration based on 90.2% model
    config = {
        'model_variant': 'breakthrough_financial',
        'hidden_size': 96,        # From 90.2% model
        'num_layers': 2,          # From 90.2% model
        'dropout_prob': 0.3,      # From 90.2% model
        'batch_size': 32,         # From 90.2% model
        'learning_rate': 0.001,   # From 90.2% model
        'weight_decay': 1e-4,     # From 90.2% model
        'epochs': 100,
        'patience': 15,
        'sequence_length': 15,    # Critical: From 90.2% model
        'input_features': 33      # From 90.2% model
    }
    
    # Create model with breakthrough architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BreakthroughFinancialLSTM(
        input_size=config['input_features'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_prob=config['dropout_prob']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Breakthrough model: {total_params:,} parameters")
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_sequences, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(val_sequences, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_sequences, dtype=torch.float32),
        torch.tensor(test_targets, dtype=torch.float32)
    )
    
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
    
    # Training setup following 90.2% model
    pos_samples = sum(train_targets)
    total_samples = len(train_targets)
    pos_weight = torch.tensor([total_samples / (2 * pos_samples)]).to(device)
    
    # Use BCEWithLogitsLoss (from 90.2% model)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # AdamW optimizer with cosine annealing (from 90.2% model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    
    logger.info(f"Training setup:")
    logger.info(f"  Architecture: Multi-scale dual LSTM (breakthrough)")
    logger.info(f"  Model params: {total_params:,}")
    logger.info(f"  Sequence length: {config['sequence_length']} (vs 10 before)")
    logger.info(f"  Features: {config['input_features']} (selected)")
    logger.info(f"  Pos weight: {pos_weight.item():.3f}")
    
    # MLflow tracking
    experiment_name = "breakthrough-financial-model"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="breakthrough_v1"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("pos_weight", pos_weight.item())
        mlflow.log_param("architecture", "multi_scale_dual_lstm")
        mlflow.log_param("inspiration", "90.2_percent_model")
        
        # Training loop
        best_val_acc = 0.0
        best_val_f1 = 0.0
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
                
                # Gradient clipping (from 90.2% model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets_list = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    
                    val_predictions.extend(predicted.cpu().numpy())
                    val_targets_list.extend(batch_y.cpu().numpy())
            
            val_acc = accuracy_score(val_targets_list, val_predictions)
            val_f1 = f1_score(val_targets_list, val_predictions, zero_division=0)
            
            scheduler.step()
            
            # Early stopping based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_breakthrough_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            mlflow.log_metric("train_loss", train_loss / len(train_loader), epoch)
            mlflow.log_metric("train_accuracy", train_acc, epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader), epoch)
            mlflow.log_metric("val_accuracy", val_acc, epoch)
            mlflow.log_metric("val_f1_score", val_f1, epoch)
        
        # Final evaluation
        logger.info("=" * 60)
        logger.info("BREAKTHROUGH MODEL EVALUATION")
        logger.info("=" * 60)
        
        # Load best model
        model.load_state_dict(torch.load('best_breakthrough_model.pth'))
        model.eval()
        
        # Test evaluation
        test_predictions = []
        test_targets_list = []
        test_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                
                test_predictions.extend(predicted.cpu().numpy())
                test_targets_list.extend(batch_y.cpu().numpy())
                test_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_acc = accuracy_score(test_targets_list, test_predictions)
        test_precision = precision_score(test_targets_list, test_predictions, zero_division=0)
        test_recall = recall_score(test_targets_list, test_predictions, zero_division=0)
        test_f1 = f1_score(test_targets_list, test_predictions, zero_division=0)
        
        # Probability analysis
        prob_mean = np.mean(test_probabilities)
        prob_std = np.std(test_probabilities)
        up_prediction_rate = np.mean(test_predictions)
        
        # Log final metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("best_val_f1", best_val_f1)
        
        # Results
        results = {
            'model_variant': 'breakthrough_financial_v1',
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'best_val_accuracy': best_val_acc,
            'best_val_f1': best_val_f1,
            'total_parameters': total_params,
            'input_features': config['input_features'],
            'sequence_length': config['sequence_length'],
            'architecture_inspiration': '90.2_percent_model',
            'key_innovations': [
                'Multi-scale dual LSTM processing',
                '15-step sequences (vs 10)',
                '33 selected financial features',
                'Layer normalization',
                'Cosine annealing scheduler',
                'Gradient clipping',
                'BCEWithLogitsLoss with pos_weight'
            ],
            'probability_analysis': {
                'mean': float(prob_mean),
                'std': float(prob_std),
                'up_prediction_rate': float(up_prediction_rate)
            },
            'training_config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open('breakthrough_model_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Final summary
        logger.info(f"BREAKTHROUGH MODEL RESULTS:")
        logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1-Score: {test_f1:.4f}")
        logger.info(f"Best Val F1: {best_val_f1:.4f}")
        logger.info(f"Model Parameters: {total_params:,}")
        
        # Compare against previous models
        baseline_acc = 0.504
        enhanced_acc = 0.492
        optimized_acc = 0.508
        
        logger.info(f"\\nCOMPARISON ANALYSIS:")
        logger.info(f"vs Baseline (50.4%): {((test_acc - baseline_acc) / baseline_acc * 100):+.1f}%")
        logger.info(f"vs Enhanced (49.2%): {((test_acc - enhanced_acc) / enhanced_acc * 100):+.1f}%")
        logger.info(f"vs Optimized (50.8%): {((test_acc - optimized_acc) / optimized_acc * 100):+.1f}%")
        
        if test_acc > 0.80:
            logger.info("🎯 SUCCESS: Achieved 80%+ accuracy target!")
        elif test_acc > 0.70:
            logger.info("✅ EXCELLENT: Major breakthrough achieved")
        elif test_acc > 0.60:
            logger.info("📈 SIGNIFICANT: Meaningful improvement")
        elif test_acc > 0.55:
            logger.info("🔄 PROGRESS: Moving toward breakthrough")
        else:
            logger.info("⚠️ NEEDS REFINEMENT: Architecture promising but needs tuning")
        
        return results

if __name__ == "__main__":
    results = train_breakthrough_model()
    
    print(f"\\n🎯 Breakthrough Model Training Complete!")
    print(f"Accuracy: {results['test_accuracy']:.1%}")
    print(f"Architecture: Multi-scale dual LSTM (90.2% inspired)")
    print(f"Sequence Length: {results['sequence_length']} steps")
    print(f"Features: {results['input_features']}")
    
    if results['test_accuracy'] > 0.70:
        print("🚀 Breakthrough achieved! Ready for deployment!")
    elif results['test_accuracy'] > 0.60:
        print("📈 Significant progress! Continue refinement!")
    else:
        print("🔧 Architecture promising, needs feature engineering refinement")
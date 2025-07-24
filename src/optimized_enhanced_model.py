#!/usr/bin/env python3
"""
Optimized Enhanced Model Implementation
Implements immediate improvements for higher accuracy
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
from sklearn.feature_selection import SelectKBest, f_classif
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss to combat majority class bias"""
    
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Apply pos_weight if provided
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

class OptimizedEnhancedLSTM(nn.Module):
    """Optimized Enhanced LSTM with reduced capacity and better regularization"""
    
    def __init__(self, input_size=205, hidden_size=64, num_layers=2, dropout_prob=0.5):
        super().__init__()
        
        # Reduced capacity to prevent overfitting
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature attention (lightweight)
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(32, input_size),
            nn.Sigmoid()  # Changed to sigmoid for gating
        )
        
        # Simplified LSTM architecture
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Simplified classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob * 0.8),
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
        batch_size = x.size(0)
        
        # Feature attention (element-wise gating)
        attention_weights = self.feature_attention(x)
        x_attended = x * attention_weights
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_attended)
        
        # Use last output
        final_output = lstm_out[:, -1, :]
        
        # Batch normalization
        final_output = self.batch_norm(final_output)
        
        # Classification
        output = self.classifier(final_output)
        
        return output

class ImprovedDataset(torch.utils.data.Dataset):
    """Dataset with data augmentation and improved preprocessing"""
    
    def __init__(self, sequences, targets, training=True, noise_factor=0.01):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.training = training
        self.noise_factor = noise_factor
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Add noise during training for regularization
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(sequence) * self.noise_factor
            sequence = sequence + noise
        
        return sequence, target

def select_best_features(sequences, targets, k=100):
    """Select top k features using statistical tests"""
    logger.info(f"Selecting top {k} features...")
    
    # Reshape for sklearn
    n_samples, seq_len, n_features = sequences.shape
    X_reshaped = sequences.reshape(n_samples, seq_len * n_features)
    
    # Select features
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_reshaped, targets)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    
    # Convert back to sequence format
    selected_features_per_timestep = k // seq_len
    
    logger.info(f"Selected {len(selected_indices)} features")
    return X_selected.reshape(n_samples, seq_len, selected_features_per_timestep), selected_indices

def load_and_preprocess_data():
    """Load and preprocess enhanced data with optimizations"""
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
    
    logger.info(f"Original data shapes:")
    logger.info(f"  Train: {train_sequences.shape}")
    logger.info(f"  Val: {val_sequences.shape}")
    logger.info(f"  Test: {test_sequences.shape}")
    
    # Feature selection on training data
    # Combine train and val for feature selection
    combined_sequences = np.concatenate([train_sequences, val_sequences], axis=0)
    combined_targets = np.concatenate([train_targets, val_targets], axis=0)
    
    # Select top 100 features
    selected_combined, feature_indices = select_best_features(combined_sequences, combined_targets, k=100)
    
    # Split back
    train_size = len(train_sequences)
    train_selected = selected_combined[:train_size]
    val_selected = selected_combined[train_size:]
    
    # Apply same feature selection to test set
    test_reshaped = test_sequences.reshape(test_sequences.shape[0], -1)
    test_selected_flat = test_reshaped[:, feature_indices]
    test_selected = test_selected_flat.reshape(test_sequences.shape[0], 10, -1)
    
    # Create datasets with augmentation
    train_dataset = ImprovedDataset(train_selected, train_targets, training=True, noise_factor=0.01)
    val_dataset = ImprovedDataset(val_selected, val_targets, training=False)
    test_dataset = ImprovedDataset(test_selected, test_targets, training=False)
    
    # Update metadata
    metadata['n_features_selected'] = train_selected.shape[2]
    metadata['feature_selection'] = 'SelectKBest_f_classif'
    metadata['selected_feature_indices'] = feature_indices.tolist()
    
    logger.info(f"After feature selection:")
    logger.info(f"  Train: {train_selected.shape}")
    logger.info(f"  Val: {val_selected.shape}")
    logger.info(f"  Test: {test_selected.shape}")
    
    return train_dataset, val_dataset, test_dataset, metadata

def create_optimized_training_config():
    """Create optimized training configuration"""
    return {
        'model_variant': 'optimized_enhanced',
        'hidden_size': 64,  # Reduced from 128
        'num_layers': 2,    # Reduced from 3
        'dropout_prob': 0.5,  # Increased from 0.3
        'batch_size': 16,   # Reduced for better gradients
        'learning_rate': 0.001,
        'weight_decay': 1e-3,  # Increased from 1e-4
        'epochs': 150,
        'patience': 20,
        'focal_loss_gamma': 2.0,
        'focal_loss_alpha': 1.0,
        'gradient_clip': 0.1,  # Reduced from 0.5
        'label_smoothing': 0.1
    }

def train_optimized_model():
    """Train optimized enhanced model with improvements"""
    
    logger.info("=" * 80)
    logger.info("OPTIMIZED ENHANCED MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load and preprocess data
    train_dataset, val_dataset, test_dataset, metadata = load_and_preprocess_data()
    
    # Training configuration
    config = create_optimized_training_config()
    
    # Update input size based on feature selection
    config['input_size'] = metadata['n_features_selected']
    
    # Create model with reduced capacity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedEnhancedLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_prob=config['dropout_prob']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Optimized model: {total_params:,} parameters (vs 225k before)")
    
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
    
    # Calculate class weights
    pos_samples = sum(train_dataset.targets)
    total_samples = len(train_dataset.targets)
    pos_weight = torch.tensor([total_samples / (2 * pos_samples)]).to(device)
    
    # Use Focal Loss instead of BCE
    criterion = FocalLoss(
        alpha=config['focal_loss_alpha'],
        gamma=config['focal_loss_gamma'],
        pos_weight=pos_weight
    )
    
    # Optimizer with higher weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # OneCycle learning rate schedule
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'] * 5,
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    logger.info(f"Training setup:")
    logger.info(f"  Input size: {config['input_size']}")
    logger.info(f"  Model params: {total_params:,}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Loss function: Focal Loss (γ={config['focal_loss_gamma']})")
    logger.info(f"  Pos weight: {pos_weight.item():.3f}")
    
    # MLflow tracking
    experiment_name = "optimized-enhanced-model"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="optimized_enhanced_v1"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("pos_weight", pos_weight.item())
        mlflow.log_param("device", str(device))
        
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
                
                optimizer.step()
                scheduler.step()
                
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
            
            # Early stopping based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_optimized_enhanced_model.pth')
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
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Final evaluation
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)
        
        # Load best model
        model.load_state_dict(torch.load('best_optimized_enhanced_model.pth'))
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
        mlflow.log_metric("prob_mean", prob_mean)
        mlflow.log_metric("prob_std", prob_std)
        mlflow.log_metric("up_prediction_rate", up_prediction_rate)
        
        # Results
        results = {
            'model_variant': 'optimized_enhanced_v1',
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'best_val_accuracy': best_val_acc,
            'best_val_f1': best_val_f1,
            'total_parameters': total_params,
            'input_features': config['input_size'],
            'improvements': [
                'Feature selection (205 → 100)',
                'Reduced model capacity (225k → ~15k params)',
                'Focal Loss for class imbalance',
                'OneCycle learning rate schedule',
                'Enhanced regularization',
                'Data augmentation with noise'
            ],
            'probability_analysis': {
                'mean': prob_mean,
                'std': prob_std,
                'up_prediction_rate': up_prediction_rate
            },
            'training_config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        with open('optimized_enhanced_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Final summary
        logger.info(f"OPTIMIZED ENHANCED MODEL RESULTS:")
        logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1-Score: {test_f1:.4f}")
        logger.info(f"Best Val F1: {best_val_f1:.4f}")
        logger.info(f"Model Parameters: {total_params:,}")
        logger.info(f"Up Prediction Rate: {up_prediction_rate:.3f}")
        logger.info(f"Probability Std: {prob_std:.3f}")
        
        improvement_over_baseline = (test_acc - 0.504) / 0.504 * 100
        improvement_over_enhanced = (test_acc - 0.492) / 0.492 * 100
        
        logger.info(f"\nIMPROVEMENT ANALYSIS:")
        logger.info(f"vs Baseline (50.4%): {improvement_over_baseline:+.1f}%")
        logger.info(f"vs Enhanced (49.2%): {improvement_over_enhanced:+.1f}%")
        
        if test_acc > 0.80:
            logger.info("🎯 SUCCESS: Achieved 80%+ accuracy target!")
        elif test_acc > 0.70:
            logger.info("✅ EXCELLENT: Significant improvement achieved")
        elif test_acc > 0.60:
            logger.info("📈 GOOD: Meaningful improvement over baseline")
        elif test_acc > 0.55:
            logger.info("🔄 PROGRESS: Moving in right direction")
        else:
            logger.info("⚠️ NEEDS MORE WORK: Continue optimizations")
        
        return results

if __name__ == "__main__":
    results = train_optimized_model()
    
    print(f"\n🎯 Optimized Enhanced Model Training Complete!")
    print(f"Accuracy: {results['test_accuracy']:.1%}")
    print(f"Parameters: {results['total_parameters']:,}")
    print(f"Features: {results['input_features']}")
    
    if results['test_accuracy'] > 0.60:
        print("🚀 Ready for next optimization phase!")
    else:
        print("🔧 Continue systematic improvements")
"""
Enhanced training configuration to improve model performance from 52.7% to 78%+
Focus on better training strategies, loss functions, and optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import logging

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in financial data"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class EnhancedTrainingConfig:
    """Enhanced training configuration with better hyperparameters"""
    
    def __init__(self, model_variant="enhanced"):
        self.model_variant = model_variant
        self.setup_config()
        
    def setup_config(self):
        """Setup training configuration based on model variant"""
        
        if self.model_variant == "enhanced":
            self.config = {
                # Model architecture
                'model_type': 'enhanced_lstm',
                'hidden_size': 256,
                'num_layers': 3,
                'dropout_prob': 0.3,
                
                # Training parameters
                'batch_size': 128,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'patience': 15,
                
                # Data parameters
                'sequence_length': 30,  # Increased from 10
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                
                # Loss function
                'loss_function': 'focal',
                'focal_alpha': 1.0,
                'focal_gamma': 2.0,
                
                # Optimization
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'warmup_epochs': 10,
                
                # Regularization
                'gradient_clipping': 1.0,
                'label_smoothing': 0.1,
                'mixup_alpha': 0.2,
            }
            
        elif self.model_variant == "transformer":
            self.config = {
                # Model architecture
                'model_type': 'transformer',
                'hidden_size': 128,
                'num_layers': 6,
                'dropout_prob': 0.1,
                'nhead': 8,
                
                # Training parameters
                'batch_size': 64,
                'epochs': 150,
                'learning_rate': 0.0001,
                'weight_decay': 1e-4,
                'patience': 20,
                
                # Data parameters
                'sequence_length': 60,  # Longer sequences for transformer
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                
                # Loss function
                'loss_function': 'focal',
                'focal_alpha': 1.0,
                'focal_gamma': 2.0,
                
                # Optimization
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'warmup_epochs': 15,
                
                # Regularization
                'gradient_clipping': 0.5,
                'label_smoothing': 0.1,
            }
            
        elif self.model_variant == "ensemble":
            self.config = {
                # Model architecture
                'model_type': 'ensemble',
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_prob': 0.25,
                
                # Training parameters
                'batch_size': 64,
                'epochs': 120,
                'learning_rate': 0.0005,
                'weight_decay': 1e-5,
                'patience': 25,
                
                # Data parameters
                'sequence_length': 45,
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                
                # Loss function
                'loss_function': 'focal',
                'focal_alpha': 1.0,
                'focal_gamma': 2.0,
                
                # Optimization
                'optimizer': 'adamw',
                'scheduler': 'reduce_on_plateau',
                'warmup_epochs': 10,
                
                # Regularization
                'gradient_clipping': 1.0,
                'label_smoothing': 0.1,
            }
            
        else:  # baseline
            self.config = {
                # Model architecture
                'model_type': 'enhanced_lstm',
                'hidden_size': 64,
                'num_layers': 2,
                'dropout_prob': 0.2,
                
                # Training parameters
                'batch_size': 64,
                'epochs': 80,
                'learning_rate': 0.001,
                'weight_decay': 1e-6,
                'patience': 10,
                
                # Data parameters
                'sequence_length': 20,
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                
                # Loss function
                'loss_function': 'bce',
                
                # Optimization
                'optimizer': 'adam',
                'scheduler': 'reduce_on_plateau',
                'warmup_epochs': 5,
                
                # Regularization
                'gradient_clipping': 0.5,
            }

class EnhancedTrainer:
    """Enhanced trainer with better training strategies"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.setup_training()
        
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        
        # Optimizer
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        
        # Scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['patience'] // 2,
                verbose=True
            )
        
        # Loss function
        if self.config['loss_function'] == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config['focal_alpha'],
                gamma=self.config['focal_gamma']
            )
        else:
            self.criterion = nn.BCELoss()
    
    def train_epoch(self, train_loader):
        """Train for one epoch with enhanced strategies"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Apply mixup augmentation if configured
            if hasattr(self.config, 'mixup_alpha') and self.config['mixup_alpha'] > 0:
                data, targets = self.mixup_data(data, targets, self.config['mixup_alpha'])
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Apply label smoothing if configured
            if hasattr(self.config, 'label_smoothing') and self.config['label_smoothing'] > 0:
                targets = self.smooth_labels(targets, self.config['label_smoothing'])
            
            loss = self.criterion(outputs.squeeze(), targets)
            loss.backward()
            
            # Gradient clipping
            if 'gradient_clipping' in self.config:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs.squeeze(), targets)
                
                total_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def mixup_data(self, x, y, alpha):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    def smooth_labels(self, targets, smoothing):
        """Apply label smoothing"""
        return targets * (1 - smoothing) + 0.5 * smoothing
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Learning rate scheduling
            if self.config['scheduler'] == 'reduce_on_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['patience']:
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

def get_enhanced_training_config(model_variant="enhanced"):
    """Get enhanced training configuration"""
    return EnhancedTrainingConfig(model_variant).config

def calculate_class_weights(targets):
    """Calculate class weights for imbalanced data"""
    targets_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    unique_classes = np.unique(targets_np)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=targets_np)
    return dict(zip(unique_classes, class_weights))
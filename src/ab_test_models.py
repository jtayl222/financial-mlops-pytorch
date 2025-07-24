#!/usr/bin/env python3
"""
A/B Testing Framework: Multi-scale Dual LSTM vs Optimized LSTM
Using market-aware features to beat coin flip performance
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Unified shape contract for A/B testing
SEQUENCE_LENGTH = 10  # Standardized for all models
TARGET_FEATURES = 50  # Reasonable feature count

class MultiScaleDualLSTM(nn.Module):
    """Champion Model: Multi-scale architecture from 90.2% model"""
    
    def __init__(self, input_size=50, hidden_size=64, dropout_prob=0.3):
        super().__init__()
        
        # Dual-scale processing
        self.lstm_short = nn.LSTM(
            input_size, hidden_size//2, 1,
            batch_first=True, dropout=0
        )
        self.lstm_long = nn.LSTM(
            input_size, hidden_size//2, 2,
            batch_first=True, dropout=dropout_prob
        )
        
        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, x):
        # Multi-scale processing
        short_out, _ = self.lstm_short(x)
        long_out, _ = self.lstm_long(x)
        
        # Combine scales
        combined = torch.cat([short_out[:, -1, :], long_out[:, -1, :]], dim=1)
        
        # Normalize and classify
        normalized = self.layer_norm(combined)
        output = self.classifier(normalized)
        
        return output

class OptimizedLSTM(nn.Module):
    """Challenger Model: Feature-selected regularized LSTM"""
    
    def __init__(self, input_size=50, hidden_size=64, dropout_prob=0.4):
        super().__init__()
        
        # Single optimized LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, 2,
            batch_first=True, dropout=dropout_prob
        )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Regularized classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        final_output = lstm_out[:, -1, :]
        
        # Normalize and classify
        normalized = self.batch_norm(final_output)
        output = self.classifier(normalized)
        
        return output

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()

def load_market_aware_data():
    """Load market-aware features for A/B testing"""
    
    data_dir = "data/processed/market_aware"
    
    # Check if data exists, if not create it
    if not os.path.exists(os.path.join(data_dir, 'market_aware_features.csv')):
        logger.info("Creating market-aware features...")
        from market_aware_features import prepare_ab_test_data
        features_df, feature_cols = prepare_ab_test_data()
    else:
        # Load existing data
        features_df = pd.read_csv(
            os.path.join(data_dir, 'market_aware_features.csv'),
            index_col=0, parse_dates=True
        )
        feature_cols = [col for col in features_df.columns if col not in ['target', 'target_return']]
    
    return features_df, feature_cols

def prepare_unified_data(features_df, feature_cols, target_features=50):
    """Prepare data with unified shape contract for A/B testing"""
    
    # Split data
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)
    
    train_df = features_df[:train_size]
    val_df = features_df[train_size:train_size + val_size]
    test_df = features_df[train_size + val_size:]
    
    # Feature selection to get best features
    selector = SelectKBest(f_classif, k=target_features)
    
    # Normalize features
    scaler = StandardScaler()
    
    # Fit on train data
    train_features = scaler.fit_transform(train_df[feature_cols])
    train_features_selected = selector.fit_transform(train_features, train_df['target'])
    
    # Transform val/test
    val_features = scaler.transform(val_df[feature_cols])
    val_features_selected = selector.transform(val_features)
    
    test_features = scaler.transform(test_df[feature_cols])
    test_features_selected = selector.transform(test_features)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_cols[i] for i in selected_indices]
    
    logger.info(f"Selected top {target_features} features:")
    for i, feat in enumerate(selected_feature_names[:10]):
        logger.info(f"  {i+1}. {feat}")
    
    # Create sequences
    def create_sequences(features, targets, seq_len=SEQUENCE_LENGTH):
        X, y = [], []
        for i in range(len(features) - seq_len + 1):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len-1])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_features_selected, train_df['target'].values)
    X_val, y_val = create_sequences(val_features_selected, val_df['target'].values)
    X_test, y_test = create_sequences(test_features_selected, test_df['target'].values)
    
    logger.info(f"Unified data shapes:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), selected_feature_names

def train_model(model, train_data, val_data, model_name, use_focal_loss=False):
    """Train a model with consistent settings"""
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Loss function
    pos_weight = torch.tensor([len(y_train) / (2 * sum(y_train))]).to(device)
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    logger.info(f"Training {model_name}...")
    
    for epoch in range(100):
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
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_predictions)
        
        # Learning rate scheduling
        scheduler.step(val_loss / len(val_loader))
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ", "_")}_ab.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{model_name.lower().replace(" ", "_")}_ab.pth'))
    
    return model, best_val_acc

def evaluate_model(model, test_data, model_name):
    """Evaluate model on test data"""
    
    X_test, y_test = test_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    targets = []
    probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    results = {
        'model': model_name,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'test_samples': len(targets),
        'up_prediction_rate': np.mean(predictions),
        'probability_std': np.std(probabilities)
    }
    
    return results

def run_ab_test():
    """Run complete A/B test between models"""
    
    logger.info("=" * 60)
    logger.info("A/B TESTING: Multi-scale Dual LSTM vs Optimized LSTM")
    logger.info("=" * 60)
    
    # Load market-aware data
    features_df, feature_cols = load_market_aware_data()
    
    # Prepare unified data
    train_data, val_data, test_data, selected_features = prepare_unified_data(
        features_df, feature_cols, TARGET_FEATURES
    )
    
    # Initialize models
    model_a = MultiScaleDualLSTM(input_size=TARGET_FEATURES)
    model_b = OptimizedLSTM(input_size=TARGET_FEATURES)
    
    # MLflow tracking
    mlflow.set_experiment("ab-test-market-aware")
    
    with mlflow.start_run(run_name="ab_test_comparison"):
        # Log configuration
        config = {
            'sequence_length': SEQUENCE_LENGTH,
            'n_features': TARGET_FEATURES,
            'train_samples': len(train_data[0]),
            'val_samples': len(val_data[0]),
            'test_samples': len(test_data[0]),
            'feature_engineering': 'market_aware'
        }
        mlflow.log_params(config)
        
        # Train Model A: Multi-scale Dual LSTM
        model_a_trained, val_acc_a = train_model(
            model_a, train_data, val_data, "Multi-scale Dual LSTM"
        )
        
        # Train Model B: Optimized LSTM with Focal Loss
        model_b_trained, val_acc_b = train_model(
            model_b, train_data, val_data, "Optimized LSTM", use_focal_loss=True
        )
        
        # Evaluate both models
        results_a = evaluate_model(model_a_trained, test_data, "Multi-scale Dual LSTM")
        results_b = evaluate_model(model_b_trained, test_data, "Optimized LSTM")
        
        # Log results
        mlflow.log_metric("model_a_test_accuracy", results_a['accuracy'])
        mlflow.log_metric("model_b_test_accuracy", results_b['accuracy'])
        mlflow.log_metric("model_a_val_accuracy", val_acc_a)
        mlflow.log_metric("model_b_val_accuracy", val_acc_b)
        
        # Display results
        print("\n" + "=" * 60)
        print("A/B TEST RESULTS")
        print("=" * 60)
        
        for results in [results_a, results_b]:
            print(f"\n{results['model']}:")
            print(f"  Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
            print(f"  Up Prediction Rate: {results['up_prediction_rate']:.3f}")
        
        # Winner determination
        acc_diff = results_a['accuracy'] - results_b['accuracy']
        
        print(f"\n🏆 WINNER ANALYSIS:")
        print(f"Accuracy Difference: {acc_diff:+.4f} ({acc_diff*100:+.1f} percentage points)")
        
        if acc_diff > 0.02:
            winner = "Multi-scale Dual LSTM"
            confidence = "Strong"
        elif acc_diff > 0.01:
            winner = "Multi-scale Dual LSTM"
            confidence = "Moderate"
        elif acc_diff < -0.02:
            winner = "Optimized LSTM"
            confidence = "Strong"
        elif acc_diff < -0.01:
            winner = "Optimized LSTM"
            confidence = "Moderate"
        else:
            winner = "Tie"
            confidence = "No significant difference"
        
        print(f"Winner: {winner}")
        print(f"Confidence: {confidence}")
        
        # Check if we beat coin flip
        best_acc = max(results_a['accuracy'], results_b['accuracy'])
        
        print(f"\n🎯 COIN FLIP ANALYSIS:")
        if best_acc > 0.55:
            print(f"✅ SUCCESS: {best_acc:.1%} accuracy beats coin flip!")
            print(f"📈 Market-aware features working!")
        elif best_acc > 0.52:
            print(f"🔄 PROGRESS: {best_acc:.1%} shows promise")
            print(f"💡 Continue feature engineering")
        else:
            print(f"⚠️ CHALLENGE: {best_acc:.1%} still near random")
            print(f"🔧 Need external data sources")
        
        # Save A/B test results
        ab_results = {
            'model_a': results_a,
            'model_b': results_b,
            'winner': winner,
            'confidence': confidence,
            'best_accuracy': best_acc,
            'selected_features': selected_features[:20],  # Top 20 features
            'beats_coin_flip': best_acc > 0.52
        }
        
        with open('ab_test_results.json', 'w') as f:
            json.dump(ab_results, f, indent=2)
        
        print(f"\n💾 Results saved to: ab_test_results.json")
        print(f"📊 MLflow run: {mlflow.active_run().info.run_id}")
        
        return ab_results

if __name__ == "__main__":
    results = run_ab_test()
    
    if results['beats_coin_flip']:
        print("\n🚀 Ready for production A/B testing!")
        print("✅ Deploy both models with traffic splitting")
        print("📊 Monitor real-world performance")
    else:
        print("\n🔬 Continue experimentation")
        print("💡 Consider: Options flow, sentiment, macro indicators")
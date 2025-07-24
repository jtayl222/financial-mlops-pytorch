#!/usr/bin/env python3
"""
Quick A/B Test using existing data
Test Multi-scale Dual LSTM vs Optimized LSTM with available enhanced features
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiScaleDualLSTM(nn.Module):
    """Multi-scale Dual LSTM for A/B testing"""
    
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
    """Optimized LSTM for A/B testing"""
    
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

def load_enhanced_data():
    """Load the enhanced features we already have"""
    
    enhanced_dir = "data/processed/enhanced"
    
    # Load sequences
    train_sequences = np.load(os.path.join(enhanced_dir, 'train_sequences.npy'))
    train_targets = np.load(os.path.join(enhanced_dir, 'train_targets.npy'))
    val_sequences = np.load(os.path.join(enhanced_dir, 'val_sequences.npy'))
    val_targets = np.load(os.path.join(enhanced_dir, 'val_targets.npy'))
    test_sequences = np.load(os.path.join(enhanced_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(enhanced_dir, 'test_targets.npy'))
    
    logger.info(f"Enhanced data loaded:")
    logger.info(f"  Train: {train_sequences.shape}")
    logger.info(f"  Val: {val_sequences.shape}")
    logger.info(f"  Test: {test_sequences.shape}")
    
    return (train_sequences, train_targets), (val_sequences, val_targets), (test_sequences, test_targets)

def prepare_ab_data(train_data, val_data, test_data, target_features=50):
    """Prepare data for A/B testing with unified shape"""
    
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data
    
    # Feature selection to reduce from 205 to target_features
    # Reshape for feature selection
    train_X_flat = train_X.reshape(train_X.shape[0], -1)
    val_X_flat = val_X.reshape(val_X.shape[0], -1)
    test_X_flat = test_X.reshape(test_X.shape[0], -1)
    
    # Select best features
    selector = SelectKBest(f_classif, k=target_features * train_X.shape[1])  # target_features per timestep
    
    train_X_selected = selector.fit_transform(train_X_flat, train_y)
    val_X_selected = selector.transform(val_X_flat)
    test_X_selected = selector.transform(test_X_flat)
    
    # Reshape back to sequences
    train_X_final = train_X_selected.reshape(train_X.shape[0], train_X.shape[1], target_features)
    val_X_final = val_X_selected.reshape(val_X.shape[0], val_X.shape[1], target_features)
    test_X_final = test_X_selected.reshape(test_X.shape[0], test_X.shape[1], target_features)
    
    logger.info(f"A/B data prepared:")
    logger.info(f"  Train: {train_X_final.shape}")
    logger.info(f"  Val: {val_X_final.shape}")
    logger.info(f"  Test: {test_X_final.shape}")
    
    return (train_X_final, train_y), (val_X_final, val_y), (test_X_final, test_y)

def train_model(model, train_data, val_data, model_name):
    """Train a model for A/B testing"""
    
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
    
    # Loss and optimizer
    pos_weight = torch.tensor([len(y_train) / (2 * sum(y_train))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    logger.info(f"Training {model_name}...")
    
    for epoch in range(50):  # Shorter training for quick test
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
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_predictions)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ", "_")}_quick.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{model_name.lower().replace(" ", "_")}_quick.pth'))
    
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

def quick_ab_test():
    """Run quick A/B test with existing enhanced features"""
    
    logger.info("=" * 60)
    logger.info("QUICK A/B TEST: Multi-scale vs Optimized")
    logger.info("=" * 60)
    
    # Load enhanced data
    train_data, val_data, test_data = load_enhanced_data()
    
    # Prepare for A/B testing
    train_ab, val_ab, test_ab = prepare_ab_data(train_data, val_data, test_data, target_features=50)
    
    # Initialize models
    input_size = train_ab[0].shape[2]  # Should be 50
    model_a = MultiScaleDualLSTM(input_size=input_size)
    model_b = OptimizedLSTM(input_size=input_size)
    
    # Train Model A: Multi-scale Dual LSTM
    model_a_trained, val_acc_a = train_model(model_a, train_ab, val_ab, "Multi-scale Dual LSTM")
    
    # Train Model B: Optimized LSTM
    model_b_trained, val_acc_b = train_model(model_b, train_ab, val_ab, "Optimized LSTM")
    
    # Evaluate both models
    results_a = evaluate_model(model_a_trained, test_ab, "Multi-scale Dual LSTM")
    results_b = evaluate_model(model_b_trained, test_ab, "Optimized LSTM")
    
    # Display results
    print("\\n" + "=" * 60)
    print("QUICK A/B TEST RESULTS")
    print("=" * 60)
    
    for results in [results_a, results_b]:
        print(f"\\n{results['model']}:")
        print(f"  Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  Up Prediction Rate: {results['up_prediction_rate']:.3f}")
    
    # Winner determination
    acc_diff = results_a['accuracy'] - results_b['accuracy']
    
    print(f"\\n🏆 WINNER ANALYSIS:")
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
    
    # Check performance vs previous models
    best_acc = max(results_a['accuracy'], results_b['accuracy'])
    
    print(f"\\n📊 PERFORMANCE COMPARISON:")
    print(f"• Simple 902 (previous best): 53.8%")
    print(f"• A/B Test best: {best_acc:.1%}")
    
    improvement = (best_acc - 0.538) / 0.538 * 100
    print(f"• Improvement: {improvement:+.1f}%")
    
    if best_acc > 0.55:
        print(f"✅ SUCCESS: Beating previous best!")
        print(f"🚀 Ready for production A/B testing")
    elif best_acc > 0.52:
        print(f"🔄 PROGRESS: Competitive with best model")
        print(f"📊 Ready for infrastructure A/B testing")
    else:
        print(f"⚠️ BASELINE: Still around coin flip")
        print(f"🔧 Focus on feature engineering")
    
    # Save results
    ab_results = {
        'model_a': results_a,
        'model_b': results_b,
        'winner': winner,
        'confidence': confidence,
        'best_accuracy': best_acc,
        'improvement_vs_simple902': improvement,
        'input_shape': list(train_ab[0].shape),
        'unified_shape_contract': True
    }
    
    with open('quick_ab_test_results.json', 'w') as f:
        json.dump(ab_results, f, indent=2)
    
    print(f"\\n💾 Results saved to: quick_ab_test_results.json")
    
    return ab_results

if __name__ == "__main__":
    results = quick_ab_test()
    
    print(f"\\n🎯 FINAL VERDICT:")
    print(f"Winner: {results['winner']}")
    print(f"Best Accuracy: {results['best_accuracy']:.1%}")
    
    if results['best_accuracy'] > 0.55:
        print("🚀 Deploy winner for production A/B testing!")
        print("✅ Unified shape contract ready")
    else:
        print("🔧 Continue feature engineering for >55% target")
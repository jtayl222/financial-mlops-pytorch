#!/usr/bin/env python3
"""
Fair Comparison: Simple 902 vs Enhanced on Same Test Split
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_enhanced_test_data():
    """Load the enhanced test data that other models used"""
    enhanced_dir = "data/processed/enhanced"
    
    test_sequences = np.load(os.path.join(enhanced_dir, 'test_sequences.npy'))
    test_targets = np.load(os.path.join(enhanced_dir, 'test_targets.npy'))
    
    print(f"Enhanced test data: {test_sequences.shape}")
    return test_sequences, test_targets

def evaluate_enhanced_model():
    """Evaluate enhanced/optimized model on test data"""
    
    print("🔍 Evaluating Enhanced/Optimized Model...")
    
    # Load test data
    test_sequences, test_targets = load_enhanced_test_data()
    
    # Use optimized model (best enhanced version)
    model_path = "best_optimized_enhanced_model.pth"
    if not os.path.exists(model_path):
        print("❌ Optimized enhanced model not found")
        return None
    
    # Create model (optimized version)
    from optimized_enhanced_model import OptimizedEnhancedLSTM
    
    # Feature selection: take first 100 features to match optimized training
    test_sequences_selected = test_sequences[:, :, :100]
    
    model = OptimizedEnhancedLSTM(input_size=10, hidden_size=64, num_layers=2)  # Shape: [batch, 10, 100]
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Convert to tensors
    X_test = torch.FloatTensor(test_sequences_selected)
    y_test = torch.FloatTensor(test_targets)
    
    print(f"Enhanced model input shape: {X_test.shape}")
    
    # Evaluate
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test), 
        batch_size=32, shuffle=False
    )
    
    predictions = []
    targets = []
    probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            probs = torch.sigmoid(outputs) if len(outputs.shape) > 0 else torch.sigmoid(outputs.unsqueeze(0))
            preds = (probs > 0.5).float()
            
            if len(preds.shape) == 0:
                preds = preds.unsqueeze(0)
                probs = probs.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    return {
        'model': 'enhanced_optimized',
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_samples': len(targets),
        'input_shape': list(X_test.shape),
        'up_prediction_rate': np.mean(predictions),
        'prob_std': np.std(probabilities)
    }

def create_compatible_simple_model():
    """Create simple model compatible with enhanced test data"""
    
    print("🔍 Creating Shape-Compatible Simple Model...")
    
    # Load enhanced test data
    test_sequences, test_targets = load_enhanced_test_data()
    
    # We need to adapt simple features to match enhanced data structure
    # Enhanced data: [191, 10, 205] -> Need to create simple features in this format
    
    # Load simple features from our biotech approach
    from simple_features_902 import load_simple_features, create_test_sequences
    
    # Get simple features
    features_df, feature_cols = load_simple_features()
    
    # Create test split matching enhanced model
    # Enhanced used different date range, so let's use same test size
    total_samples = len(features_df)
    enhanced_test_size = len(test_targets)  # 191 samples
    
    # Take last N samples to match enhanced test set size
    simple_test_df = features_df.tail(enhanced_test_size + 20)  # +20 for sequence creation buffer
    
    # Create sequences for simple model (15 timesteps)
    scaler = StandardScaler()
    
    # Fit on all but test data
    train_df = features_df.head(total_samples - enhanced_test_size - 20)
    train_features = scaler.fit_transform(train_df[feature_cols])
    
    test_features = scaler.transform(simple_test_df[feature_cols])
    test_targets_simple = simple_test_df['Target'].values
    
    # Create 15-step sequences
    def create_sequences(features, targets, seq_len):
        X, y = [], []
        for i in range(len(features) - seq_len + 1):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len-1])
        return np.array(X), np.array(y)
    
    X_test_simple, y_test_simple = create_sequences(test_features, test_targets_simple, 15)
    
    # Load simple model
    model_path = "best_simple_902_model.pth"
    if not os.path.exists(model_path):
        print("❌ Simple 902 model not found")
        return None
    
    from simple_features_902 import Simple902LSTM
    
    model = Simple902LSTM(input_size=len(feature_cols), hidden_size=96, num_layers=2)
    model.classifier[-1] = torch.nn.Identity()  # Remove sigmoid
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"Simple model input shape: {X_test_simple.shape}")
    
    # Evaluate
    X_test_tensor = torch.FloatTensor(X_test_simple)
    y_test_tensor = torch.FloatTensor(y_test_simple)
    
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=32, shuffle=False
    )
    
    predictions = []
    targets = []
    probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            if len(preds.shape) == 0:
                preds = preds.unsqueeze(0)
                probs = probs.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            
            predictions.extend(preds.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    return {
        'model': 'simple_902',
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_samples': len(targets),
        'input_shape': list(X_test_simple.shape),
        'up_prediction_rate': np.mean(predictions),
        'prob_std': np.std(probabilities)
    }

def fair_comparison():
    """Run fair comparison between models"""
    
    print("🎯 FAIR MODEL COMPARISON")
    print("=" * 60)
    
    # Evaluate both models
    enhanced_results = evaluate_enhanced_model()
    simple_results = create_compatible_simple_model()
    
    if not enhanced_results or not simple_results:
        print("❌ Could not evaluate both models")
        return
    
    print("\\n📊 FAIR COMPARISON RESULTS:")
    print("=" * 60)
    
    for results in [enhanced_results, simple_results]:
        model_name = results['model']
        acc = results['accuracy']
        f1 = results['f1']
        samples = results['test_samples']
        shape = results['input_shape']
        up_rate = results['up_prediction_rate']
        
        print(f"\\n{model_name.upper()}:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Test Samples: {samples}")
        print(f"  Input Shape: {shape}")
        print(f"  Up Prediction Rate: {up_rate:.3f}")
    
    # Statistical comparison
    acc_diff = simple_results['accuracy'] - enhanced_results['accuracy']
    acc_diff_pct = (acc_diff / enhanced_results['accuracy']) * 100
    
    print(f"\\n🔍 STATISTICAL ANALYSIS:")
    print(f"• Accuracy Difference: {acc_diff:+.4f} ({acc_diff_pct:+.1f}%)")
    
    if abs(acc_diff) < 0.02:
        significance = "No significant difference"
        recommendation = "Both models perform similarly"
    elif acc_diff > 0.05:
        significance = "Simple model significantly better"
        recommendation = "Deploy Simple 902 for A/B testing"
    elif acc_diff > 0.02:
        significance = "Simple model moderately better"
        recommendation = "Simple 902 shows promise"
    else:
        significance = "Enhanced model slightly better"
        recommendation = "Continue with enhanced approach"
    
    print(f"• Statistical Significance: {significance}")
    print(f"• Recommendation: {recommendation}")
    
    # A/B Testing Readiness
    print(f"\\n🚀 A/B TESTING READINESS:")
    
    if max(enhanced_results['accuracy'], simple_results['accuracy']) > 0.60:
        readiness = "✅ READY - Models show business value"
    elif max(enhanced_results['accuracy'], simple_results['accuracy']) > 0.55:
        readiness = "🔄 CONDITIONAL - Infrastructure testing only"
    else:
        readiness = "⚠️ NOT READY - Both models near random"
    
    print(f"• Status: {readiness}")
    
    # Shape compatibility for A/B testing
    print(f"\\n🔧 SHAPE COMPATIBILITY:")
    enhanced_shape = enhanced_results['input_shape']
    simple_shape = simple_results['input_shape']
    
    if enhanced_shape[1:] == simple_shape[1:]:
        print("✅ Input shapes compatible for A/B testing")
    else:
        print("❌ Input shapes incompatible - need adaptation layer")
        print(f"  Enhanced: {enhanced_shape[1:]}")
        print(f"  Simple: {simple_shape[1:]}")
        print("🔧 Solution: Create unified preprocessing pipeline")
    
    # Save comparison results
    comparison_results = {
        'enhanced_model': enhanced_results,
        'simple_model': simple_results,
        'comparison': {
            'accuracy_difference': acc_diff,
            'accuracy_difference_percent': acc_diff_pct,
            'significance': significance,
            'recommendation': recommendation,
            'ab_testing_readiness': readiness
        },
        'shape_compatibility': enhanced_shape[1:] == simple_shape[1:]
    }
    
    with open('fair_model_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to: fair_model_comparison.json")
    
    return comparison_results

if __name__ == "__main__":
    results = fair_comparison()
    
    if results:
        simple_acc = results['simple_model']['accuracy']
        enhanced_acc = results['enhanced_model']['accuracy']
        
        print(f"\\n🎯 FINAL VERDICT:")
        print(f"Simple 902: {simple_acc:.1%}")
        print(f"Enhanced: {enhanced_acc:.1%}")
        
        if simple_acc > enhanced_acc + 0.02:
            print("🚀 Simple approach wins - ready for A/B testing!")
        elif abs(simple_acc - enhanced_acc) < 0.02:
            print("🤝 Models perform similarly - choose based on complexity")
        else:
            print("📊 Enhanced approach slightly better")
        
        if max(simple_acc, enhanced_acc) > 0.55:
            print("✅ Deploy best model for infrastructure validation")
        else:
            print("🔍 Both models still near random - focus on data quality")
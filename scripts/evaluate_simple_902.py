#!/usr/bin/env python3
"""
Quick evaluation of Simple 902 Model
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_simple_features():
    """Load simple features exactly as created by the training script"""
    
    tickers = ['IBB', 'XBI', 'XLV']
    raw_data_dir = "data/raw"
    
    all_data = {}
    
    # Load each ticker's data (replicate training logic)
    for ticker in tickers:
        file_path = os.path.join(raw_data_dir, f"{ticker}_raw_2018-01-01_2023-12-31.csv")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df = df.sort_index()
        
        features_df = pd.DataFrame(index=df.index)
        
        # Basic OHLCV
        features_df[f'Close_{ticker}'] = df[f'Close_{ticker}']
        features_df[f'High_{ticker}'] = df[f'High_{ticker}']
        features_df[f'Low_{ticker}'] = df[f'Low_{ticker}']
        features_df[f'Open_{ticker}'] = df[f'Open_{ticker}']
        features_df[f'Volume_{ticker}'] = df[f'Volume_{ticker}']
        
        # Price lags (1-5 days)
        for lag in range(1, 6):
            features_df[f'Close_{ticker}_lag_{lag}'] = df[f'Close_{ticker}'].shift(lag)
        
        # Volume lags (1-3 days)
        for lag in range(1, 4):
            features_df[f'Volume_{ticker}_lag_{lag}'] = df[f'Volume_{ticker}'].shift(lag)
        
        # SMA indicators (5, 10, 20 periods)
        for period in [5, 10, 20]:
            features_df[f'SMA_Close_{ticker}_{period}'] = df[f'Close_{ticker}'].rolling(period).mean()
        
        # RSI (14 period)
        delta = df[f'Close_{ticker}'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df[f'RSI_Close_{ticker}_14'] = 100 - (100 / (1 + rs))
        
        all_data[ticker] = features_df
    
    # Combine all ticker data
    combined_features = pd.concat(all_data.values(), axis=1)
    
    # Create target
    ibb_close = combined_features['Close_IBB']
    daily_return = ibb_close.pct_change(1).shift(-1)
    combined_features['Daily_Return'] = daily_return
    combined_features['Target'] = (daily_return > 0).astype(float)
    
    # Drop NaN rows
    combined_features = combined_features.dropna()
    
    # Get feature names
    feature_cols = [col for col in combined_features.columns 
                   if col not in ['Daily_Return', 'Target']]
    
    return combined_features, feature_cols

def create_test_sequences(features_df, feature_cols):
    """Create test sequences matching training format"""
    
    # Split data (70/15/15)
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)
    
    test_df = features_df[train_size + val_size:]
    
    # Prepare features and targets
    scaler = StandardScaler()
    
    # We need to fit scaler on train data for proper evaluation
    train_df = features_df[:train_size]
    train_features = scaler.fit_transform(train_df[feature_cols])
    
    test_features = scaler.transform(test_df[feature_cols])
    test_targets = test_df['Target'].values
    
    # Create sequences (15 steps)
    sequence_length = 15
    
    def create_sequences(features, targets, seq_len):
        X, y = [], []
        for i in range(len(features) - seq_len + 1):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len-1])
        return np.array(X), np.array(y)
    
    X_test, y_test = create_sequences(test_features, test_targets, sequence_length)
    
    return X_test, y_test, len(feature_cols)

def evaluate_simple_902():
    """Evaluate the saved simple 902 model"""
    
    print("🔍 Evaluating Simple 902 Model...")
    
    # Check if model exists
    model_path = "best_simple_902_model.pth"
    if not os.path.exists(model_path):
        print("❌ Simple 902 model not found")
        return None
    
    # Load test data
    print("📊 Loading test data...")
    features_df, feature_cols = load_simple_features()
    X_test, y_test, input_size = create_test_sequences(features_df, feature_cols)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {input_size}")
    print(f"Test samples: {len(y_test)}")
    
    # Load model architecture (need to recreate)
    from simple_features_902 import Simple902LSTM
    
    model = Simple902LSTM(input_size, hidden_size=96, num_layers=2, dropout_prob=0.3)
    
    # Remove sigmoid for BCEWithLogitsLoss compatibility
    model.classifier[-1] = torch.nn.Identity()
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Convert test data to tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create test loader
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    print("🧪 Running evaluation...")
    test_predictions = []
    test_targets_list = []
    test_probabilities = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            
            test_predictions.extend(predicted.cpu().numpy())
            test_targets_list.extend(batch_y.cpu().numpy())
            test_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_targets_list, test_predictions)
    test_precision = precision_score(test_targets_list, test_predictions, zero_division=0)
    test_recall = recall_score(test_targets_list, test_predictions, zero_division=0)
    test_f1 = f1_score(test_targets_list, test_predictions, zero_division=0)
    
    # Analysis
    prob_mean = np.mean(test_probabilities)
    prob_std = np.std(test_probabilities)
    up_prediction_rate = np.mean(test_predictions)
    
    # Results
    print("\n" + "="*60)
    print("SIMPLE 90.2% MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"✅ Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"📊 Test Precision: {test_precision:.4f}")
    print(f"📊 Test Recall: {test_recall:.4f}")
    print(f"📊 Test F1-Score: {test_f1:.4f}")
    print(f"🎯 Model Parameters: {total_params:,}")
    print(f"📈 Features: {input_size} (simple)")
    print(f"🔄 Up Prediction Rate: {up_prediction_rate:.3f}")
    print(f"📊 Probability Std: {prob_std:.3f}")
    
    # Compare with previous models
    print(f"\n📈 COMPARISON WITH ALL APPROACHES:")
    print(f"• Complex Enhanced (205 features): 49.2%")
    print(f"• Optimized (100 features): 50.8%")
    print(f"• Breakthrough (33 features): 49.7%")
    print(f"• Simple 902 ({input_size} features): {test_acc:.1%}")
    
    # Analysis
    if test_acc > 0.80:
        print("\n🎯 SUCCESS: Simple approach achieved 80%+ accuracy!")
        status = "breakthrough_achieved"
    elif test_acc > 0.70:
        print("\n✅ BREAKTHROUGH: Simple features work significantly better!")
        status = "major_improvement"
    elif test_acc > 0.60:
        print("\n📈 SIGNIFICANT: Simple approach shows clear promise")
        status = "promising_improvement"
    elif test_acc > 0.55:
        print("\n🔄 PROGRESS: Simple approach slightly better")
        status = "modest_improvement"
    else:
        print("\n🔍 INVESTIGATION: All approaches still around random")
        status = "still_random"
    
    # Save results
    results = {
        'model_variant': 'simple_902_evaluated',
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1_score': test_f1,
        'total_parameters': total_params,
        'input_features': input_size,
        'tickers': ['IBB', 'XBI', 'XLV'],
        'approach': 'simple_902_biotech_focus',
        'probability_analysis': {
            'mean': float(prob_mean),
            'std': float(prob_std),
            'up_prediction_rate': float(up_prediction_rate)
        },
        'status': status,
        'comparison': {
            'complex_enhanced': 0.492,
            'optimized': 0.508,
            'breakthrough': 0.497,
            'simple_902': test_acc
        }
    }
    
    with open('simple_902_evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == "__main__":
    results = evaluate_simple_902()
    
    if results:
        print(f"\n🎯 Simple 902 Model Evaluation Complete!")
        print(f"Final Accuracy: {results['test_accuracy']:.1%}")
        print(f"Status: {results['status']}")
        
        if results['test_accuracy'] > 0.70:
            print("🚀 Simple approach breakthrough confirmed!")
        elif results['test_accuracy'] > 0.60:
            print("📈 Simple features show significant promise!")
        else:
            print("🔍 Need deeper investigation of 90.2% model data")
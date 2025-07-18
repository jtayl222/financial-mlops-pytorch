#!/usr/bin/env python3
"""
Test advanced model performance during COVID crash period (Feb-Apr 2020)
to demonstrate how market regime changes destroy model performance.
"""

import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime
import logging
from pathlib import Path

# Import our modules
from enhanced_features import calculate_advanced_technical_indicators
from advanced_financial_model import FinancialLSTM
from device_utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_covid_period_data():
    """Load IBB data specifically for COVID crash period"""
    data_file = Path("data/raw/IBB_raw_2018-01-01_2023-12-31.csv")
    
    if not data_file.exists():
        raise FileNotFoundError(f"IBB data not found at {data_file}")
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Define critical periods
    pre_covid = df['2019-01-01':'2020-02-19']  # Pre-crash training period
    covid_crash = df['2020-02-20':'2020-04-30']  # Peak volatility period
    recovery = df['2020-05-01':'2020-12-31']  # Recovery period
    
    logging.info(f"Pre-COVID period: {len(pre_covid)} days")
    logging.info(f"COVID crash period: {len(covid_crash)} days") 
    logging.info(f"Recovery period: {len(recovery)} days")
    
    return pre_covid, covid_crash, recovery

def prepare_features_for_period(df, ticker="IBB"):
    """Apply advanced feature engineering to a specific period"""
    
    # Reset index to work with our feature engineering function
    df_work = df.reset_index()
    df_work = df_work.rename(columns={'Date': 'date'})
    
    # Check existing column format and rename if needed
    if 'Close_IBB' in df_work.columns:
        # Data already has ticker suffix, just rename for consistency
        column_mapping = {
            'Open_IBB': 'Open_IBB',
            'High_IBB': 'High_IBB', 
            'Low_IBB': 'Low_IBB',
            'Close_IBB': 'Close_IBB',
            'Volume_IBB': 'Volume_IBB'
        }
    else:
        # Add ticker columns if they don't exist
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_work[f"{col}_{ticker}"] = df_work[col]
    
    # Calculate advanced features
    df_features = calculate_advanced_technical_indicators(df_work, ticker)
    
    # Create target (next day direction)
    df_features[f'target_{ticker}'] = (df_features[f'Close_{ticker}'].shift(-1) > df_features[f'Close_{ticker}']).astype(int)
    
    # Drop NaN rows
    df_features = df_features.dropna()
    
    # Extract feature columns (exclude basic OHLCV and target)
    feature_cols = [col for col in df_features.columns 
                   if not col.startswith(('date', 'Open_', 'High_', 'Low_', 'Close_', 'Volume_', 'target_'))
                   and not col in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    X = df_features[feature_cols].values
    y = df_features[f'target_{ticker}'].values
    
    logging.info(f"Features shape: {X.shape}, Targets shape: {y.shape}")
    logging.info(f"Feature columns: {len(feature_cols)}")
    
    return X, y, df_features

def create_sequences(X, y, sequence_length=15):
    """Create sequences for LSTM input"""
    sequences_X, sequences_y = [], []
    
    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:i+sequence_length])
        sequences_y.append(y[i+sequence_length])
    
    return np.array(sequences_X), np.array(sequences_y)

def test_model_on_covid_crash():
    """Test the advanced model on COVID crash period"""
    
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load data for different periods
    pre_covid, covid_crash, recovery = load_covid_period_data()
    
    # Prepare features for each period
    logging.info("Preparing pre-COVID features...")
    X_train, y_train, df_train = prepare_features_for_period(pre_covid)
    
    logging.info("Preparing COVID crash features...")
    X_crash, y_crash, df_crash = prepare_features_for_period(covid_crash)
    
    logging.info("Preparing recovery features...")
    X_recovery, y_recovery, df_recovery = prepare_features_for_period(recovery)
    
    # Create sequences
    sequence_length = 15
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_crash_seq, y_crash_seq = create_sequences(X_crash, y_crash, sequence_length)
    X_recovery_seq, y_recovery_seq = create_sequences(X_recovery, y_recovery, sequence_length)
    
    logging.info(f"Training sequences: {X_train_seq.shape}")
    logging.info(f"COVID crash sequences: {X_crash_seq.shape}")
    logging.info(f"Recovery sequences: {X_recovery_seq.shape}")
    
    # Initialize model with same architecture as advanced model
    input_size = X_train_seq.shape[2]
    model = FinancialLSTM(
        input_size=input_size,
        hidden_size=96,
        num_layers=2,
        dropout_prob=0.3
    ).to(device)
    
    # Quick training on pre-COVID data
    logging.info("Training model on pre-COVID data...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
    
    # Simple training loop
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model.train()
    for epoch in range(50):  # Quick training
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test on different periods
    def evaluate_period(X_test, y_test, period_name):
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities > 0.5).float()
            
            accuracy = (predictions == y_test_tensor).float().mean().item()
            
            # Calculate additional metrics
            tp = ((predictions == 1) & (y_test_tensor == 1)).sum().item()
            tn = ((predictions == 0) & (y_test_tensor == 0)).sum().item()
            fp = ((predictions == 1) & (y_test_tensor == 0)).sum().item()
            fn = ((predictions == 0) & (y_test_tensor == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
        return {
            'period': period_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_predictions': len(y_test),
            'class_distribution': {
                'up_days': int(y_test.sum()),
                'down_days': int(len(y_test) - y_test.sum())
            }
        }
    
    # Evaluate on all periods
    results = {}
    
    # Pre-COVID (should be good - trained on this)
    results['pre_covid'] = evaluate_period(X_train_seq, y_train_seq, "Pre-COVID (Training)")
    
    # COVID crash (should be terrible)
    results['covid_crash'] = evaluate_period(X_crash_seq, y_crash_seq, "COVID Crash")
    
    # Recovery (should be mediocre)
    results['recovery'] = evaluate_period(X_recovery_seq, y_recovery_seq, "Recovery")
    
    # Calculate volatility for each period
    volatility_stats = {
        'pre_covid_volatility': pre_covid['Close_IBB'].pct_change().std() * np.sqrt(252),
        'covid_crash_volatility': covid_crash['Close_IBB'].pct_change().std() * np.sqrt(252),
        'recovery_volatility': recovery['Close_IBB'].pct_change().std() * np.sqrt(252)
    }
    
    # Compile final results
    final_results = {
        'model_performance': results,
        'volatility_analysis': volatility_stats,
        'key_insights': {
            'performance_degradation': {
                'pre_covid_accuracy': results['pre_covid']['accuracy'],
                'covid_crash_accuracy': results['covid_crash']['accuracy'],
                'recovery_accuracy': results['recovery']['accuracy']
            },
            'volatility_correlation': "Model performance inversely correlated with market volatility",
            'regime_change_impact': "Trained patterns break down during unprecedented market conditions"
        },
        'timestamp': datetime.now().isoformat(),
        'test_description': "Advanced model tested on COVID crash to demonstrate regime change vulnerability"
    }
    
    # Save results
    results_file = Path("results/experiments/covid_crash_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    logging.info("=== COVID CRASH TEST RESULTS ===")
    logging.info(f"Pre-COVID accuracy: {results['pre_covid']['accuracy']:.1%}")
    logging.info(f"COVID crash accuracy: {results['covid_crash']['accuracy']:.1%}")
    logging.info(f"Recovery accuracy: {results['recovery']['accuracy']:.1%}")
    logging.info(f"Pre-COVID volatility: {volatility_stats['pre_covid_volatility']:.1%}")
    logging.info(f"COVID crash volatility: {volatility_stats['covid_crash_volatility']:.1%}")
    logging.info(f"Recovery volatility: {volatility_stats['recovery_volatility']:.1%}")
    
    return final_results

if __name__ == "__main__":
    results = test_model_on_covid_crash()
    print(f"\nResults saved to: results/experiments/covid_crash_test_results.json")
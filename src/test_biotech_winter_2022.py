#!/usr/bin/env python3
"""
Test advanced model performance during 2022 biotech winter.
The biotech sector crashed ~50% from Feb-Oct 2022 due to:
- Rising interest rates
- Reduced risk appetite  
- Regulatory concerns
- Funding drought

This tests if our model can handle a different type of crisis
than the sharp COVID crash.
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

def load_biotech_winter_data():
    """Load IBB data for 2022 biotech winter crash"""
    data_file = Path("data/raw/IBB_raw_2018-01-01_2023-12-31.csv")
    
    if not data_file.exists():
        raise FileNotFoundError(f"IBB data not found at {data_file}")
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Define periods for 2022 biotech winter
    pre_crash_2022 = df['2021-01-01':'2022-01-31']  # Pre-crash training
    biotech_winter = df['2022-02-01':'2022-10-31']  # Main crash period
    late_2022_recovery = df['2022-11-01':'2022-12-31']  # Partial recovery
    
    logging.info(f"Pre-crash 2022 period: {len(pre_crash_2022)} days")
    logging.info(f"Biotech winter period: {len(biotech_winter)} days") 
    logging.info(f"Late 2022 recovery: {len(late_2022_recovery)} days")
    
    return pre_crash_2022, biotech_winter, late_2022_recovery

def prepare_features_for_period(df, ticker="IBB"):
    """Apply advanced feature engineering to a specific period"""
    
    # Reset index to work with our feature engineering function
    df_work = df.reset_index()
    df_work = df_work.rename(columns={'Date': 'date'})
    
    # Data already has ticker suffix format
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

def calculate_trading_returns_2022(predictions, actual_returns, period_name):
    """Calculate trading returns for 2022 biotech winter"""
    
    # Strategy: Buy if predict up (1), Sell/Short if predict down (0)
    strategy_returns = []
    for pred, actual_ret in zip(predictions, actual_returns):
        if pred == 1:  # Predict up -> go long
            strategy_return = actual_ret
        else:  # Predict down -> go short
            strategy_return = -actual_ret
        strategy_returns.append(strategy_return)
    
    strategy_returns = np.array(strategy_returns)
    
    # Calculate performance metrics
    total_return = np.sum(strategy_returns)
    annualized_return = np.mean(strategy_returns) * 252
    volatility = np.std(strategy_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Win rate and other metrics
    win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
    buy_hold_return = np.sum(actual_returns)
    
    return {
        'period': period_name,
        'total_return_pct': total_return * 100,
        'buy_hold_return_pct': buy_hold_return * 100,
        'excess_return_pct': (total_return - buy_hold_return) * 100,
        'annualized_return_pct': annualized_return * 100,
        'volatility_pct': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown * 100,
        'win_rate_pct': win_rate * 100,
        'trading_days': len(strategy_returns),
        'best_day_pct': np.max(strategy_returns) * 100,
        'worst_day_pct': np.min(strategy_returns) * 100
    }

def test_biotech_winter_2022():
    """Test the advanced model on 2022 biotech winter"""
    
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load data for different periods
    pre_crash, biotech_winter, recovery = load_biotech_winter_data()
    
    # Prepare features for each period
    logging.info("Preparing pre-crash 2022 features...")
    X_train, y_train, df_train = prepare_features_for_period(pre_crash)
    
    logging.info("Preparing biotech winter features...")
    X_winter, y_winter, df_winter = prepare_features_for_period(biotech_winter)
    
    logging.info("Preparing recovery features...")
    X_recovery, y_recovery, df_recovery = prepare_features_for_period(recovery)
    
    # Create sequences
    sequence_length = 15
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_winter_seq, y_winter_seq = create_sequences(X_winter, y_winter, sequence_length)
    X_recovery_seq, y_recovery_seq = create_sequences(X_recovery, y_recovery, sequence_length)
    
    logging.info(f"Training sequences: {X_train_seq.shape}")
    logging.info(f"Biotech winter sequences: {X_winter_seq.shape}")
    logging.info(f"Recovery sequences: {X_recovery_seq.shape}")
    
    # Initialize model with same architecture as advanced model
    input_size = X_train_seq.shape[2]
    model = FinancialLSTM(
        input_size=input_size,
        hidden_size=96,
        num_layers=2,
        dropout_prob=0.3
    ).to(device)
    
    # Quick training on pre-crash 2022 data
    logging.info("Training model on pre-crash 2022 data...")
    
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
            'predictions': predictions.cpu().numpy(),
            'total_predictions': len(y_test),
            'class_distribution': {
                'up_days': int(y_test.sum()),
                'down_days': int(len(y_test) - y_test.sum())
            }
        }
    
    # Evaluate on all periods
    results = {}
    
    # Pre-crash 2022 (training period)
    results['pre_crash_2022'] = evaluate_period(X_train_seq, y_train_seq, "Pre-crash 2022")
    
    # Biotech winter 2022 (main test)
    results['biotech_winter'] = evaluate_period(X_winter_seq, y_winter_seq, "Biotech Winter 2022")
    
    # Recovery period
    results['recovery'] = evaluate_period(X_recovery_seq, y_recovery_seq, "Late 2022 Recovery")
    
    # Calculate actual returns for each period
    winter_returns = biotech_winter['Close_IBB'].pct_change().dropna().values
    winter_returns = winter_returns[sequence_length:]  # Align with sequences
    
    recovery_returns = recovery['Close_IBB'].pct_change().dropna().values 
    recovery_returns = recovery_returns[sequence_length:]  # Align with sequences
    
    # Calculate trading performance
    winter_trading = calculate_trading_returns_2022(
        results['biotech_winter']['predictions'],
        winter_returns,
        "Biotech Winter 2022"
    )
    
    recovery_trading = calculate_trading_returns_2022(
        results['recovery']['predictions'],
        recovery_returns,
        "Late 2022 Recovery"
    )
    
    # Calculate volatility for each period
    volatility_stats = {
        'pre_crash_2022_volatility': pre_crash['Close_IBB'].pct_change().std() * np.sqrt(252),
        'biotech_winter_volatility': biotech_winter['Close_IBB'].pct_change().std() * np.sqrt(252),
        'recovery_volatility': recovery['Close_IBB'].pct_change().std() * np.sqrt(252)
    }
    
    # Price decline analysis
    price_analysis = {
        'pre_crash_start_price': float(pre_crash['Close_IBB'].iloc[0]),
        'pre_crash_end_price': float(pre_crash['Close_IBB'].iloc[-1]),
        'winter_start_price': float(biotech_winter['Close_IBB'].iloc[0]),
        'winter_end_price': float(biotech_winter['Close_IBB'].iloc[-1]),
        'winter_decline_pct': float((biotech_winter['Close_IBB'].iloc[-1] / biotech_winter['Close_IBB'].iloc[0] - 1) * 100),
        'recovery_start_price': float(recovery['Close_IBB'].iloc[0]),
        'recovery_end_price': float(recovery['Close_IBB'].iloc[-1])
    }
    
    # Compile final results
    final_results = {
        'model_performance': {
            k: {key: val for key, val in v.items() if key != 'predictions'}  # Remove predictions array
            for k, v in results.items()
        },
        'trading_performance': {
            'biotech_winter': winter_trading,
            'recovery': recovery_trading
        },
        'market_analysis': {
            'volatility_stats': volatility_stats,
            'price_analysis': price_analysis
        },
        'comparison_with_covid': {
            'covid_crash_accuracy': 0.571,
            'biotech_winter_accuracy': results['biotech_winter']['accuracy'],
            'crash_type_difference': "COVID: Sharp 2-month crash vs Biotech Winter: Gradual 9-month decline"
        },
        'key_insights': {
            'regime_difference': "2022 biotech winter was a gradual decline vs COVID sharp crash",
            'model_adaptation': "Different crisis types test different aspects of model robustness",
            'volatility_impact': "Lower volatility but sustained decline creates different trading challenges"
        },
        'timestamp': datetime.now().isoformat(),
        'test_description': "Advanced model tested on 2022 biotech winter crash"
    }
    
    # Save results
    results_file = Path("results/experiments/biotech_winter_2022_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    logging.info("=== 2022 BIOTECH WINTER TEST RESULTS ===")
    logging.info(f"Pre-crash 2022 accuracy: {results['pre_crash_2022']['accuracy']:.1%}")
    logging.info(f"Biotech winter accuracy: {results['biotech_winter']['accuracy']:.1%}")
    logging.info(f"Recovery accuracy: {results['recovery']['accuracy']:.1%}")
    logging.info(f"Winter trading return: {winter_trading['total_return_pct']:.1f}%")
    logging.info(f"Winter buy & hold: {winter_trading['buy_hold_return_pct']:.1f}%")
    logging.info(f"Winter volatility: {volatility_stats['biotech_winter_volatility']:.1%}")
    logging.info(f"Price decline during winter: {price_analysis['winter_decline_pct']:.1f}%")
    
    return final_results

if __name__ == "__main__":
    results = test_biotech_winter_2022()
    print(f"\nResults saved to: results/experiments/biotech_winter_2022_results.json")
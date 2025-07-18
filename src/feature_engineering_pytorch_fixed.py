#!/usr/bin/env python3
"""
Fixed Feature Engineering with Proper Financial ML Splits
Addresses data leakage issues in the original implementation
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import mlflow
from financial_ml_splits import FinancialMLSplitter

# Configure logging
loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, loglevel), format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration from environment variables
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
SCALER_DIR = os.environ.get("SCALER_DIR", "data/scalers")
N_TICKERS = int(os.environ.get("N_TICKERS", 12))

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# Time series parameters
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 10))
PREDICTION_HORIZON = 1

# Financial ML split parameters
TRAIN_END_DATE = os.environ.get("TRAIN_END_DATE", "2021-12-31")
VAL_END_DATE = os.environ.get("VAL_END_DATE", "2022-12-31")
TEST_END_DATE = os.environ.get("TEST_END_DATE", "2023-12-31")
PURGE_GAP_DAYS = int(os.environ.get("PURGE_GAP_DAYS", 5))

# Feature Engineering parameters
SMA_WINDOWS = [5, 10, 20]
RSI_WINDOWS = [14]

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def process_ticker_data(file_path):
    """Process individual ticker data with comprehensive feature engineering"""
    
    ticker_name = os.path.basename(file_path).replace('.csv', '').replace('_raw_2018-01-01_2024-12-31', '')
    logging.info(f"Processing ticker: {ticker_name}")
    
    try:
        # Load data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Ensure we have the required columns with ticker-specific names
        required_columns = [f'Open_{ticker_name}', f'High_{ticker_name}', f'Low_{ticker_name}', f'Close_{ticker_name}', f'Volume_{ticker_name}']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns in {file_path}. Expected: {required_columns}")
            logging.error(f"Found columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Sort by date to ensure chronological order
        df = df.sort_index()
        
        # Get ticker-specific column names
        close_col = f'Close_{ticker_name}'
        high_col = f'High_{ticker_name}'
        low_col = f'Low_{ticker_name}'
        open_col = f'Open_{ticker_name}'
        volume_col = f'Volume_{ticker_name}'
        
        # Calculate returns
        df[f'Returns_{ticker_name}'] = df[close_col].pct_change()
        df[f'Log_Returns_{ticker_name}'] = np.log(df[close_col] / df[close_col].shift(1))
        
        # Price-based features
        df[f'Price_Range_{ticker_name}'] = (df[high_col] - df[low_col]) / df[close_col]
        df[f'Price_Change_{ticker_name}'] = (df[close_col] - df[open_col]) / df[open_col]
        
        # Volume features
        df[f'Volume_SMA_10_{ticker_name}'] = df[volume_col].rolling(window=10).mean()
        df[f'Volume_Ratio_{ticker_name}'] = df[volume_col] / df[f'Volume_SMA_10_{ticker_name}']
        
        # Technical indicators
        for window in SMA_WINDOWS:
            df[f'SMA_{window}_{ticker_name}'] = df[close_col].rolling(window=window).mean()
            df[f'Price_to_SMA_{window}_{ticker_name}'] = df[close_col] / df[f'SMA_{window}_{ticker_name}']
        
        for window in RSI_WINDOWS:
            df[f'RSI_{window}_{ticker_name}'] = calculate_rsi(df[close_col], window)
        
        # Bollinger Bands
        bb_upper, bb_lower = calculate_bollinger_bands(df[close_col])
        df[f'BB_Upper_{ticker_name}'] = bb_upper
        df[f'BB_Lower_{ticker_name}'] = bb_lower
        df[f'BB_Width_{ticker_name}'] = (bb_upper - bb_lower) / df[close_col]
        df[f'BB_Position_{ticker_name}'] = (df[close_col] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd, signal, histogram = calculate_macd(df[close_col])
        df[f'MACD_{ticker_name}'] = macd
        df[f'MACD_Signal_{ticker_name}'] = signal
        df[f'MACD_Histogram_{ticker_name}'] = histogram
        
        # Volatility features
        df[f'Volatility_10_{ticker_name}'] = df[f'Returns_{ticker_name}'].rolling(window=10).std()
        df[f'Volatility_20_{ticker_name}'] = df[f'Returns_{ticker_name}'].rolling(window=20).std()
        
        # Momentum features
        for period in [3, 5, 10, 20]:
            df[f'Momentum_{period}_{ticker_name}'] = df[close_col] / df[close_col].shift(period) - 1
        
        # Target: Next day's return direction (use primary ticker for target)
        df[f'Target_{ticker_name}'] = (df[close_col].shift(-PREDICTION_HORIZON) > df[close_col]).astype(int)
        
        # Add ticker identifier
        df['ticker'] = ticker_name
        
        # Remove rows with NaN values
        df = df.dropna()
        
        logging.info(f"Processed {ticker_name}: {len(df)} samples, {len(df.columns)} features")
        
        return df
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def create_financial_ml_splits(ticker_data_dict):
    """Create proper financial ML splits using FinancialMLSplitter"""
    
    # Initialize splitter
    splitter = FinancialMLSplitter(
        purge_gap_days=PURGE_GAP_DAYS,
        embargo_days=2,
        min_train_samples=500
    )
    
    # Create per-ticker splits
    per_ticker_splits = splitter.create_per_ticker_splits(
        ticker_data_dict, 
        split_method="time_based",
        train_end_date=TRAIN_END_DATE,
        val_end_date=VAL_END_DATE,
        test_end_date=TEST_END_DATE
    )
    
    # Combine splits
    combined_splits = splitter.combine_ticker_splits(per_ticker_splits)
    
    return combined_splits

def create_sequences(features_df, target_series, sequence_length):
    """Create sequences for LSTM training"""
    
    sequences = []
    targets = []
    
    # Group by ticker to ensure sequences don't cross ticker boundaries
    for ticker in features_df['ticker'].unique():
        ticker_data = features_df[features_df['ticker'] == ticker].copy()
        ticker_targets = target_series[target_series.index.isin(ticker_data.index)]
        
        # Drop ticker column for sequence creation
        ticker_features = ticker_data.drop('ticker', axis=1)
        
        # Create sequences
        for i in range(len(ticker_features) - sequence_length):
            seq = ticker_features.iloc[i:i+sequence_length].values
            target = ticker_targets.iloc[i+sequence_length-1]
            
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

class FinancialTimeSeriesDataset(Dataset):
    """PyTorch Dataset for financial time series with proper structure"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def main():
    """Main feature engineering pipeline with proper financial ML splits"""
    
    # Start MLflow run
    with mlflow.start_run(run_name="feature_engineering_pytorch_fixed"):
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("prediction_horizon", PREDICTION_HORIZON)
        mlflow.log_param("n_tickers", N_TICKERS)
        mlflow.log_param("train_end_date", TRAIN_END_DATE)
        mlflow.log_param("val_end_date", VAL_END_DATE)
        mlflow.log_param("test_end_date", TEST_END_DATE)
        mlflow.log_param("purge_gap_days", PURGE_GAP_DAYS)
        
        # Process each ticker separately
        ticker_data_dict = {}
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
        
        if not raw_files:
            logging.error(f"No raw CSV files found in {RAW_DATA_DIR}")
            return
        
        for filename in raw_files:
            ticker_df = process_ticker_data(os.path.join(RAW_DATA_DIR, filename))
            if not ticker_df.empty:
                ticker_name = filename.replace('.csv', '')
                ticker_data_dict[ticker_name] = ticker_df
            else:
                logging.warning(f"Skipping {filename} due to empty processed data")
        
        if not ticker_data_dict:
            logging.error("No data processed for any ticker")
            return
        
        logging.info(f"Successfully processed {len(ticker_data_dict)} tickers")
        
        # Create proper financial ML splits
        splits = create_financial_ml_splits(ticker_data_dict)
        
        # Process each split
        processed_splits = {}
        scalers = {}
        
        for split_name, split_df in splits.items():
            logging.info(f"Processing {split_name} split: {len(split_df)} samples")
            
            # Separate features and targets
            target_cols = [col for col in split_df.columns if col.startswith('Target_')]
            feature_cols = [col for col in split_df.columns if col not in target_cols + ['ticker']]
            features_df = split_df[feature_cols + ['ticker']].copy()
            
            # Use IBB as primary target for prediction (biotech-focused model)
            if 'Target_IBB' in split_df.columns:
                target_series = split_df['Target_IBB'].copy()
                logging.info(f"Using IBB as primary target for {split_name} split")
            else:
                # Fall back to first available target
                target_series = split_df[target_cols[0]].copy()
                logging.info(f"Using {target_cols[0]} as primary target for {split_name} split")
            
            # Scale features (fit scaler only on training data)
            if split_name == 'train':
                scaler = MinMaxScaler()
                features_scaled = scaler.fit_transform(features_df.drop('ticker', axis=1))
                scalers['feature_scaler'] = scaler
                
                # Save scaler
                with open(os.path.join(SCALER_DIR, 'feature_scaler.pkl'), 'wb') as f:
                    pickle.dump(scaler, f)
                    
            else:
                # Use fitted scaler for validation and test
                features_scaled = scalers['feature_scaler'].transform(features_df.drop('ticker', axis=1))
            
            # Reconstruct features DataFrame with ticker column
            features_scaled_df = pd.DataFrame(
                features_scaled, 
                index=features_df.index,
                columns=feature_cols
            )
            features_scaled_df['ticker'] = features_df['ticker'].values
            
            # Create sequences
            sequences, targets = create_sequences(features_scaled_df, target_series, SEQUENCE_LENGTH)
            
            # Create PyTorch dataset
            dataset = FinancialTimeSeriesDataset(sequences, targets)
            
            # Save processed data
            torch.save(dataset, os.path.join(PROCESSED_DATA_DIR, f'{split_name}_dataset.pt'))
            
            processed_splits[split_name] = {
                'sequences': sequences,
                'targets': targets,
                'dataset': dataset
            }
            
            logging.info(f"{split_name} sequences shape: {sequences.shape}")
            logging.info(f"{split_name} targets shape: {targets.shape}")
            logging.info(f"{split_name} positive rate: {targets.mean():.3f}")
            
            # Log metrics
            mlflow.log_metric(f"{split_name}_samples", len(targets))
            mlflow.log_metric(f"{split_name}_positive_rate", targets.mean())
        
        # Log feature information
        mlflow.log_metric("n_features", len(feature_cols))
        mlflow.log_metric("sequence_length", SEQUENCE_LENGTH)
        
        # Save metadata
        metadata = {
            'feature_columns': feature_cols,
            'n_features': len(feature_cols),
            'sequence_length': SEQUENCE_LENGTH,
            'n_tickers': len(ticker_data_dict),
            'ticker_names': list(ticker_data_dict.keys()),
            'splits': {name: len(data['targets']) for name, data in processed_splits.items()}
        }
        
        with open(os.path.join(PROCESSED_DATA_DIR, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info("Feature engineering completed successfully!")
        logging.info(f"Results saved to: {PROCESSED_DATA_DIR}")
        
        return processed_splits

if __name__ == "__main__":
    main()
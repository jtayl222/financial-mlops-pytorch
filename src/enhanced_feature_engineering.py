#!/usr/bin/env python3
"""
Enhanced Feature Engineering Implementation
Creates advanced features for 80%+ accuracy while maintaining shape contract
"""

import os
import numpy as np
import pandas as pd
import talib
import logging
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering that maintains [10, 205] shape contract
    Implements sophisticated financial features for high accuracy
    """
    
    def __init__(self, data_dir: str = "/Users/user/REPOS/financial-mlops-pytorch/data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Enhanced ticker selection (6 most informative)
        self.selected_tickers = ['IBB', 'XBI', 'SPY', 'QQQ', 'XLV', 'MRNA']
        self.features_per_ticker = 33
        self.market_features = 7
        self.sequence_length = 10
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def load_ticker_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for selected tickers"""
        ticker_data = {}
        
        for ticker in self.selected_tickers:
            file_path = os.path.join(self.raw_dir, f"{ticker}_raw_2018-01-01_2023-12-31.csv")
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                ticker_data[ticker] = df
                logger.info(f"Loaded {ticker}: {df.shape}")
            else:
                logger.warning(f"Missing data file: {file_path}")
        
        return ticker_data
    
    def calculate_price_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate price-based features (7 features)"""
        features = pd.DataFrame(index=df.index)
        
        # Basic OHLC (using actual column names)
        features[f'{ticker}_open'] = df[f'Open_{ticker}']
        features[f'{ticker}_high'] = df[f'High_{ticker}']
        features[f'{ticker}_low'] = df[f'Low_{ticker}']
        features[f'{ticker}_close'] = df[f'Close_{ticker}']
        
        # Returns
        features[f'{ticker}_daily_return'] = df[f'Close_{ticker}'].pct_change()
        features[f'{ticker}_log_return'] = np.log(df[f'Close_{ticker}'] / df[f'Close_{ticker}'].shift(1))
        features[f'{ticker}_intraday_return'] = (df[f'Close_{ticker}'] - df[f'Open_{ticker}']) / df[f'Open_{ticker}']
        
        return features
    
    def calculate_momentum_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate momentum features (4 features)"""
        features = pd.DataFrame(index=df.index)
        
        # Multi-period momentum
        for period in [3, 5, 10, 20]:
            features[f'{ticker}_momentum_{period}'] = df[f'Close_{ticker}'].pct_change(period)
        
        return features
    
    def calculate_volatility_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate volatility features (5 features)"""
        features = pd.DataFrame(index=df.index)
        
        # Rolling volatility
        for window in [5, 10, 20]:
            features[f'{ticker}_volatility_{window}'] = df[f'Close_{ticker}'].rolling(window).std()
        
        # Volatility ratio
        features[f'{ticker}_vol_ratio'] = (
            df[f'Close_{ticker}'].rolling(10).std() / 
            df[f'Close_{ticker}'].rolling(50).std()
        )
        
        # ATR using talib
        features[f'{ticker}_atr'] = talib.ATR(
            df[f'High_{ticker}'].values, df[f'Low_{ticker}'].values, df[f'Close_{ticker}'].values, timeperiod=14
        )
        
        return features
    
    def calculate_volume_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate volume features (4 features)"""
        features = pd.DataFrame(index=df.index)
        
        # Volume analysis
        features[f'{ticker}_volume'] = df[f'Volume_{ticker}']
        features[f'{ticker}_volume_ratio'] = df[f'Volume_{ticker}'] / df[f'Volume_{ticker}'].rolling(20).mean()
        features[f'{ticker}_price_volume'] = df[f'Close_{ticker}'] * df[f'Volume_{ticker}']
        
        # On-Balance Volume
        obv = (np.sign(df[f'Close_{ticker}'].diff()) * df[f'Volume_{ticker}']).cumsum()
        features[f'{ticker}_obv_change'] = obv.pct_change()
        
        return features
    
    def calculate_technical_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate technical indicator features (9 features)"""
        features = pd.DataFrame(index=df.index)
        
        # RSI at multiple periods
        for period in [9, 14, 21]:
            features[f'{ticker}_rsi_{period}'] = talib.RSI(df[f'Close_{ticker}'].values, timeperiod=period)
        
        # MACD
        macd, signal, hist = talib.MACD(df[f'Close_{ticker}'].values)
        features[f'{ticker}_macd'] = macd
        features[f'{ticker}_macd_signal'] = signal
        features[f'{ticker}_macd_hist'] = hist
        
        # Bollinger Bands position
        upper, middle, lower = talib.BBANDS(df[f'Close_{ticker}'].values)
        features[f'{ticker}_bb_position'] = (df[f'Close_{ticker}'] - lower) / (upper - lower)
        
        # Williams %R
        features[f'{ticker}_williams_r'] = talib.WILLR(
            df[f'High_{ticker}'].values, df[f'Low_{ticker}'].values, df[f'Close_{ticker}'].values
        )
        
        # Stochastic
        slowk, slowd = talib.STOCH(df[f'High_{ticker}'].values, df[f'Low_{ticker}'].values, df[f'Close_{ticker}'].values)
        features[f'{ticker}_stoch_k'] = slowk
        
        return features
    
    def calculate_microstructure_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate market microstructure features (4 features)"""
        features = pd.DataFrame(index=df.index)
        
        # High-low ratio
        features[f'{ticker}_high_low_ratio'] = (df[f'High_{ticker}'] - df[f'Low_{ticker}']) / df[f'Close_{ticker}']
        
        # Gap analysis
        features[f'{ticker}_gap'] = (df[f'Open_{ticker}'] - df[f'Close_{ticker}'].shift(1)) / df[f'Close_{ticker}'].shift(1)
        
        # Close position in daily range
        features[f'{ticker}_close_position'] = (df[f'Close_{ticker}'] - df[f'Low_{ticker}']) / (df[f'High_{ticker}'] - df[f'Low_{ticker}'])
        
        # Spread proxy
        features[f'{ticker}_spread_proxy'] = 2 * (df[f'High_{ticker}'] - df[f'Low_{ticker}']) / (df[f'High_{ticker}'] + df[f'Low_{ticker}'])
        
        return features
    
    def calculate_ticker_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate all 33 features for a single ticker"""
        feature_groups = [
            self.calculate_price_features(df, ticker),
            self.calculate_momentum_features(df, ticker),
            self.calculate_volatility_features(df, ticker),
            self.calculate_volume_features(df, ticker),
            self.calculate_technical_features(df, ticker),
            self.calculate_microstructure_features(df, ticker)
        ]
        
        combined = pd.concat(feature_groups, axis=1)
        
        # Verify we have exactly 33 features
        ticker_cols = [col for col in combined.columns if ticker in col]
        if len(ticker_cols) != self.features_per_ticker:
            logger.warning(f"Expected {self.features_per_ticker} features for {ticker}, got {len(ticker_cols)}")
        
        return combined
    
    def calculate_market_features(self, all_ticker_features: pd.DataFrame) -> pd.DataFrame:
        """Calculate 7 market-wide features"""
        market_features = pd.DataFrame(index=all_ticker_features.index)
        
        # 1. Market-wide momentum
        return_cols = [col for col in all_ticker_features.columns if 'daily_return' in col]
        market_features['market_momentum'] = all_ticker_features[return_cols].mean(axis=1)
        
        # 2. Market volatility
        vol_cols = [col for col in all_ticker_features.columns if 'volatility_10' in col]
        market_features['market_volatility'] = all_ticker_features[vol_cols].mean(axis=1)
        
        # 3. Return dispersion
        market_features['return_dispersion'] = all_ticker_features[return_cols].std(axis=1)
        
        # 4. Volume surge
        volume_ratio_cols = [col for col in all_ticker_features.columns if 'volume_ratio' in col]
        market_features['volume_surge'] = all_ticker_features[volume_ratio_cols].max(axis=1)
        
        # 5. Technical strength
        rsi_cols = [col for col in all_ticker_features.columns if 'rsi_14' in col]
        market_features['technical_strength'] = all_ticker_features[rsi_cols].mean(axis=1)
        
        # 6. Volatility regime
        vol_threshold = market_features['market_volatility'].rolling(50).mean()
        market_features['volatility_regime'] = (market_features['market_volatility'] > vol_threshold).astype(float)
        
        # 7. Market stress (high dispersion + high volatility)
        market_features['market_stress'] = (
            (market_features['return_dispersion'] > market_features['return_dispersion'].rolling(20).mean()) &
            (market_features['market_volatility'] > market_features['market_volatility'].rolling(20).mean())
        ).astype(float)
        
        return market_features
    
    def create_enhanced_features(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create all enhanced features maintaining shape contract"""
        logger.info("Creating enhanced features...")
        
        # Load data
        ticker_data = self.load_ticker_data()
        
        if not ticker_data:
            raise ValueError("No ticker data loaded")
        
        # Align all data to common date range
        common_dates = None
        for ticker, df in ticker_data.items():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
        
        logger.info(f"Common date range: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} days)")
        
        # Calculate features for each ticker
        all_features = []
        for ticker in self.selected_tickers:
            if ticker in ticker_data:
                df_aligned = ticker_data[ticker].loc[common_dates]
                ticker_features = self.calculate_ticker_features(df_aligned, ticker)
                all_features.append(ticker_features)
                logger.info(f"Created {ticker_features.shape[1]} features for {ticker}")
            else:
                # Create zero features if ticker missing
                zero_features = pd.DataFrame(
                    np.zeros((len(common_dates), self.features_per_ticker)),
                    index=common_dates,
                    columns=[f'{ticker}_feat_{i}' for i in range(self.features_per_ticker)]
                )
                all_features.append(zero_features)
                logger.warning(f"Using zero features for missing ticker: {ticker}")
        
        # Combine ticker features
        combined_features = pd.concat(all_features, axis=1)
        
        # Add market features
        market_features = self.calculate_market_features(combined_features)
        
        # Final feature matrix
        final_features = pd.concat([combined_features, market_features], axis=1)
        
        # Clean data
        final_features = final_features.fillna(method='ffill').fillna(0)
        
        # Create target (IBB next day return > 0)
        target_return = ticker_data['IBB']['Close_IBB'].pct_change().shift(-1)
        target_return = target_return.loc[common_dates]
        targets = (target_return > 0).astype(float)
        
        # Remove last row (no target)
        final_features = final_features[:-1]
        targets = targets[:-1]
        
        # Verify shape contract
        expected_features = len(self.selected_tickers) * self.features_per_ticker + self.market_features
        logger.info(f"Final features shape: {final_features.shape}")
        logger.info(f"Expected features: {expected_features}, actual: {final_features.shape[1]}")
        
        if final_features.shape[1] != 205:
            logger.warning(f"Shape contract violation: expected 205 features, got {final_features.shape[1]}")
        
        return final_features, targets.values
    
    def create_sequences(self, features: pd.DataFrame, targets: np.ndarray) -> dict:
        """Create sequences for training"""
        logger.info("Creating sequences...")
        
        # Convert to numpy
        feature_array = features.values
        
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - self.sequence_length + 1):
            seq = feature_array[i:i + self.sequence_length]
            target = targets[i + self.sequence_length - 1]
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        sequence_targets = np.array(sequence_targets, dtype=np.float32)
        
        logger.info(f"Created sequences: {sequences.shape}")
        logger.info(f"Shape contract: {sequences.shape[1:]} == (10, 205)")
        
        return {
            'sequences': sequences,
            'targets': sequence_targets,
            'feature_names': features.columns.tolist()
        }
    
    def prepare_enhanced_data(self) -> dict:
        """Full pipeline to prepare enhanced training data"""
        logger.info("=" * 60)
        logger.info("ENHANCED FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        # Create features
        features, targets = self.create_enhanced_features()
        
        # Create sequences
        sequence_data = self.create_sequences(features, targets)
        
        # Train/val/test split (temporal)
        total_sequences = len(sequence_data['sequences'])
        train_size = int(total_sequences * 0.7)
        val_size = int(total_sequences * 0.15)
        
        train_sequences = sequence_data['sequences'][:train_size]
        train_targets = sequence_data['targets'][:train_size]
        
        val_sequences = sequence_data['sequences'][train_size:train_size + val_size]
        val_targets = sequence_data['targets'][train_size:train_size + val_size]
        
        test_sequences = sequence_data['sequences'][train_size + val_size:]
        test_targets = sequence_data['targets'][train_size + val_size:]
        
        # Save enhanced data
        enhanced_dir = os.path.join(self.processed_dir, 'enhanced')
        os.makedirs(enhanced_dir, exist_ok=True)
        
        np.save(os.path.join(enhanced_dir, 'train_sequences.npy'), train_sequences)
        np.save(os.path.join(enhanced_dir, 'train_targets.npy'), train_targets)
        np.save(os.path.join(enhanced_dir, 'val_sequences.npy'), val_sequences)
        np.save(os.path.join(enhanced_dir, 'val_targets.npy'), val_targets)
        np.save(os.path.join(enhanced_dir, 'test_sequences.npy'), test_sequences)
        np.save(os.path.join(enhanced_dir, 'test_targets.npy'), test_targets)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': features.shape[1],
            'input_shape': [self.sequence_length, features.shape[1]],
            'selected_tickers': self.selected_tickers,
            'features_per_ticker': self.features_per_ticker,
            'market_features': self.market_features,
            'n_train_sequences': len(train_sequences),
            'n_val_sequences': len(val_sequences),
            'n_test_sequences': len(test_sequences),
            'target_ticker': 'IBB',
            'feature_engineering': 'enhanced_financial_indicators',
            'pipeline_version': '2.0_enhanced_features'
        }
        
        with open(os.path.join(enhanced_dir, 'enhanced_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature names
        with open(os.path.join(enhanced_dir, 'enhanced_feature_names.txt'), 'w') as f:
            for name in sequence_data['feature_names']:
                f.write(f"{name}\n")
        
        logger.info(f"Enhanced data saved to: {enhanced_dir}")
        logger.info(f"Train: {train_sequences.shape}")
        logger.info(f"Val: {val_sequences.shape}")
        logger.info(f"Test: {test_sequences.shape}")
        
        return {
            'train_sequences': train_sequences,
            'train_targets': train_targets,
            'val_sequences': val_sequences,
            'val_targets': val_targets,
            'test_sequences': test_sequences,
            'test_targets': test_targets,
            'metadata': metadata,
            'feature_names': sequence_data['feature_names']
        }


if __name__ == "__main__":
    # Create enhanced features
    engineer = EnhancedFeatureEngineer()
    data = engineer.prepare_enhanced_data()
    
    print(f"\n✅ Enhanced feature engineering complete!")
    print(f"Shape contract: {data['metadata']['input_shape']}")
    print(f"Features: {data['metadata']['n_features']}")
    print(f"Ready for high-accuracy training!")
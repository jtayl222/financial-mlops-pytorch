#!/usr/bin/env python3
"""
Advanced Feature Engineering with Shape Contract Compliance
Implements sophisticated financial features while maintaining [10, 205] shape
Targets 80%+ accuracy improvement from baseline 52.7%
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List
import talib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeaturesShapeContract:
    """
    Advanced feature engineering that maintains shape contract [10, 205]
    Uses fewer tickers with richer features per ticker
    """
    
    def __init__(self, sequence_length: int = 10, target_features: int = 205):
        self.sequence_length = sequence_length
        self.target_features = target_features
        
        # Select most informative tickers for focused approach
        self.selected_tickers = ['IBB', 'XBI', 'SPY', 'QQQ', 'XLV', 'MRNA']
        self.features_per_ticker = 33  # Advanced features
        self.market_features = 7  # Cross-market features
        
        logger.info(f"Advanced Feature Engineering Configuration:")
        logger.info(f"  Tickers: {len(self.selected_tickers)} ({self.selected_tickers})")
        logger.info(f"  Features per ticker: {self.features_per_ticker}")
        logger.info(f"  Market features: {self.market_features}")
        logger.info(f"  Total features: {len(self.selected_tickers) * self.features_per_ticker + self.market_features}")
        
    def calculate_advanced_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate 33 advanced features for a single ticker"""
        
        features = pd.DataFrame(index=df.index)
        
        # Price data
        open_col = f'Open_{ticker}'
        high_col = f'High_{ticker}'
        low_col = f'Low_{ticker}'
        close_col = f'Close_{ticker}'
        volume_col = f'Volume_{ticker}'
        
        # Ensure columns exist
        if close_col not in df.columns:
            logger.warning(f"Missing data for {ticker}")
            return pd.DataFrame(np.zeros((len(df), self.features_per_ticker)), 
                               index=df.index,
                               columns=[f'{ticker}_feat_{i}' for i in range(self.features_per_ticker)])
        
        # 1. Price-based features (7)
        features[f'{ticker}_open'] = df[open_col]
        features[f'{ticker}_high'] = df[high_col]
        features[f'{ticker}_low'] = df[low_col]
        features[f'{ticker}_close'] = df[close_col]
        features[f'{ticker}_daily_return'] = df[close_col].pct_change()
        features[f'{ticker}_log_return'] = np.log(df[close_col] / df[close_col].shift(1))
        features[f'{ticker}_intraday_return'] = (df[close_col] - df[open_col]) / df[open_col]
        
        # 2. Momentum features (4)
        for period in [3, 5, 10, 20]:
            features[f'{ticker}_momentum_{period}'] = df[close_col].pct_change(period)
        
        # 3. Volatility features (5)
        for window in [5, 10, 20]:
            features[f'{ticker}_volatility_{window}'] = df[close_col].rolling(window).std()
        
        # Volatility ratio
        features[f'{ticker}_vol_ratio_10_50'] = (
            df[close_col].rolling(10).std() / 
            df[close_col].rolling(50).std()
        )
        
        # ATR (Average True Range)
        high_low = df[high_col] - df[low_col]
        high_close = np.abs(df[high_col] - df[close_col].shift(1))
        low_close = np.abs(df[low_col] - df[close_col].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features[f'{ticker}_atr'] = true_range.rolling(14).mean()
        
        # 4. Volume analysis (4)
        features[f'{ticker}_volume'] = df[volume_col]
        features[f'{ticker}_volume_ratio'] = df[volume_col] / df[volume_col].rolling(20).mean()
        features[f'{ticker}_price_volume'] = df[close_col] * df[volume_col]
        
        # On-Balance Volume change
        obv = (np.sign(df[close_col].diff()) * df[volume_col]).cumsum()
        features[f'{ticker}_obv_change'] = obv.pct_change()
        
        # 5. Technical indicators (9)
        # RSI at multiple periods
        for period in [9, 14, 21]:
            features[f'{ticker}_rsi_{period}'] = talib.RSI(df[close_col].values, timeperiod=period)
        
        # MACD
        macd, signal, hist = talib.MACD(df[close_col].values)
        features[f'{ticker}_macd'] = macd
        features[f'{ticker}_macd_signal'] = signal
        features[f'{ticker}_macd_hist'] = hist
        
        # Bollinger Bands position
        upper, middle, lower = talib.BBANDS(df[close_col].values)
        bb_position = (df[close_col] - lower) / (upper - lower)
        features[f'{ticker}_bb_position'] = bb_position
        
        # Williams %R
        features[f'{ticker}_williams_r'] = talib.WILLR(
            df[high_col].values, df[low_col].values, df[close_col].values
        )
        
        # Stochastic
        slowk, slowd = talib.STOCH(df[high_col].values, df[low_col].values, df[close_col].values)
        features[f'{ticker}_stoch_k'] = slowk
        
        # 6. Market microstructure (4)
        features[f'{ticker}_high_low_ratio'] = (df[high_col] - df[low_col]) / df[close_col]
        features[f'{ticker}_gap'] = (df[open_col] - df[close_col].shift(1)) / df[close_col].shift(1)
        features[f'{ticker}_close_position'] = (df[close_col] - df[low_col]) / (df[high_col] - df[low_col])
        
        # Spread proxy
        features[f'{ticker}_spread_proxy'] = 2 * (df[high_col] - df[low_col]) / (df[high_col] + df[low_col])
        
        # Ensure we have exactly 33 features
        feature_cols = [col for col in features.columns if ticker in col]
        assert len(feature_cols) == self.features_per_ticker, f"Expected 33 features, got {len(feature_cols)}"
        
        return features
    
    def calculate_market_features(self, ticker_features: pd.DataFrame) -> pd.DataFrame:
        """Calculate 7 market-wide features"""
        
        market_features = pd.DataFrame(index=ticker_features.index)
        
        # 1. Average market return
        returns = [col for col in ticker_features.columns if 'daily_return' in col]
        market_features['market_avg_return'] = ticker_features[returns].mean(axis=1)
        
        # 2. Market volatility
        vols = [col for col in ticker_features.columns if 'volatility_10' in col]
        market_features['market_volatility'] = ticker_features[vols].mean(axis=1)
        
        # 3. Dispersion (cross-sectional volatility)
        market_features['return_dispersion'] = ticker_features[returns].std(axis=1)
        
        # 4. Market momentum
        mom_cols = [col for col in ticker_features.columns if 'momentum_5' in col]
        market_features['market_momentum'] = ticker_features[mom_cols].mean(axis=1)
        
        # 5. Volume surge indicator
        vol_ratios = [col for col in ticker_features.columns if 'volume_ratio' in col]
        market_features['volume_surge'] = ticker_features[vol_ratios].max(axis=1)
        
        # 6. Technical strength (average RSI)
        rsi_cols = [col for col in ticker_features.columns if 'rsi_14' in col]
        market_features['market_rsi'] = ticker_features[rsi_cols].mean(axis=1)
        
        # 7. Volatility regime (high vol = 1, low vol = 0)
        vol_threshold = market_features['market_volatility'].rolling(50).mean()
        market_features['vol_regime'] = (market_features['market_volatility'] > vol_threshold).astype(float)
        
        return market_features
    
    def create_features_with_shape_contract(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create advanced features maintaining [10, 205] shape contract
        
        Args:
            data: Dictionary with ticker dataframes
            
        Returns:
            features: Array of shape (n_samples, n_features) = (n, 205)
            targets: Array of shape (n_samples,)
        """
        
        all_features = []
        
        # Calculate features for each selected ticker
        for ticker in self.selected_tickers:
            if ticker in data:
                ticker_features = self.calculate_advanced_features(data[ticker], ticker)
                all_features.append(ticker_features)
            else:
                # Create zero features if ticker missing
                logger.warning(f"Ticker {ticker} not found in data")
                zero_features = pd.DataFrame(
                    np.zeros((len(list(data.values())[0]), self.features_per_ticker)),
                    columns=[f'{ticker}_feat_{i}' for i in range(self.features_per_ticker)]
                )
                all_features.append(zero_features)
        
        # Combine ticker features
        combined_features = pd.concat(all_features, axis=1)
        
        # Add market-wide features
        market_features = self.calculate_market_features(combined_features)
        
        # Final feature matrix
        final_features = pd.concat([combined_features, market_features], axis=1)
        
        # Handle NaN values
        final_features = final_features.fillna(method='ffill').fillna(0)
        
        # Verify shape contract
        logger.info(f"Final feature shape: {final_features.shape}")
        assert final_features.shape[1] == self.target_features, \
            f"Shape contract violation: expected {self.target_features} features, got {final_features.shape[1]}"
        
        return final_features
    
    def prepare_training_data(self, features: pd.DataFrame, target_ticker: str = 'IBB') -> dict:
        """
        Prepare data for training with sequences
        Maintains shape contract: (n_sequences, 10, 205)
        """
        
        # Create target (next day return > 0)
        target_return = features[f'{target_ticker}_daily_return'].shift(-1)
        targets = (target_return > 0).astype(float)
        
        # Remove last row (no target)
        features = features[:-1]
        targets = targets[:-1]
        
        # Create sequences
        feature_array = features.values
        target_array = targets.values
        
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - self.sequence_length + 1):
            seq = feature_array[i:i + self.sequence_length]
            target = target_array[i + self.sequence_length - 1]
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        sequences = np.array(sequences)
        sequence_targets = np.array(sequence_targets)
        
        logger.info(f"Created sequences with shape: {sequences.shape}")
        logger.info(f"Shape contract compliance: {sequences.shape[1:]} == (10, 205)")
        
        return {
            'sequences': sequences,
            'targets': sequence_targets,
            'feature_names': features.columns.tolist()
        }


def demonstrate_advanced_features():
    """Demonstrate how advanced features maintain shape contract"""
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    sample_data = {}
    
    for ticker in ['IBB', 'XBI', 'SPY', 'QQQ', 'XLV', 'MRNA']:
        df = pd.DataFrame({
            f'Open_{ticker}': np.random.randn(len(dates)).cumsum() + 100,
            f'High_{ticker}': np.random.randn(len(dates)).cumsum() + 101,
            f'Low_{ticker}': np.random.randn(len(dates)).cumsum() + 99,
            f'Close_{ticker}': np.random.randn(len(dates)).cumsum() + 100,
            f'Volume_{ticker}': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        sample_data[ticker] = df
    
    # Create advanced features
    feature_engineer = AdvancedFeaturesShapeContract()
    features = feature_engineer.create_features_with_shape_contract(sample_data)
    
    # Prepare training data
    training_data = feature_engineer.prepare_training_data(features)
    
    print("\n" + "="*60)
    print("ADVANCED FEATURES WITH SHAPE CONTRACT")
    print("="*60)
    print(f"Features per ticker: {feature_engineer.features_per_ticker}")
    print(f"Selected tickers: {feature_engineer.selected_tickers}")
    print(f"Market features: {feature_engineer.market_features}")
    print(f"Total features: {features.shape[1]}")
    print(f"Sequence shape: {training_data['sequences'].shape}")
    print(f"✅ Shape contract maintained: {training_data['sequences'].shape[1:]} == (10, 205)")
    print("\nFeature categories:")
    print("- Price-based: 7 features")
    print("- Momentum: 4 features")
    print("- Volatility: 5 features")
    print("- Volume: 4 features")
    print("- Technical: 9 features")
    print("- Microstructure: 4 features")
    print("- Market-wide: 7 features")
    print(f"\nTotal: {6*33 + 7} = 205 features ✅")

if __name__ == "__main__":
    demonstrate_advanced_features()
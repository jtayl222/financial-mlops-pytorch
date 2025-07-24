#!/usr/bin/env python3
"""
Market-Aware Feature Engineering
Features designed to beat coin flip performance by capturing market structure
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_market_data():
    """Download AAPL, MSFT, SPY, QQQ for market-aware features"""
    
    logger.info("Downloading market-aware data...")
    
    # Core stocks + market indicators
    tickers = ['AAPL', 'MSFT', 'SPY', 'QQQ', 'VIX', 'DXY', 'TLT']
    start_date = '2018-01-01'
    end_date = '2023-12-31'
    
    all_data = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Downloading {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(data) > 0:
                all_data[ticker] = data
                logger.info(f"✅ {ticker}: {len(data)} days")
            else:
                logger.warning(f"❌ No data for {ticker}")
        except Exception as e:
            logger.warning(f"❌ Error downloading {ticker}: {e}")
            # Use alternate tickers if primary fails
            if ticker == 'VIX':
                # Use VIXY ETF as proxy
                try:
                    data = yf.download('VIXY', start=start_date, end=end_date, progress=False)
                    all_data['VIX'] = data
                except:
                    pass
            elif ticker == 'DXY':
                # Use UUP as dollar proxy
                try:
                    data = yf.download('UUP', start=start_date, end=end_date, progress=False)
                    all_data['DXY'] = data
                except:
                    pass
    
    return all_data

def create_market_aware_features(data_dict):
    """
    Create features that capture market structure beyond simple price
    Goal: Beat 50% accuracy by understanding market context
    """
    
    logger.info("Creating market-aware features...")
    
    # Primary stocks
    aapl = data_dict.get('AAPL', pd.DataFrame())
    msft = data_dict.get('MSFT', pd.DataFrame())
    spy = data_dict.get('SPY', pd.DataFrame())
    qqq = data_dict.get('QQQ', pd.DataFrame())
    
    # Align all data to common dates
    common_dates = aapl.index
    for df in [msft, spy, qqq]:
        if len(df) > 0:
            common_dates = common_dates.intersection(df.index)
    
    features_list = []
    
    for ticker, df in [('AAPL', aapl), ('MSFT', msft)]:
        if len(df) == 0:
            continue
            
        # Align to common dates
        df = df.loc[common_dates]
        features = pd.DataFrame(index=common_dates)
        
        # 1. BASIC PRICE FEATURES (keep simple)
        features[f'{ticker}_close'] = df['Close']
        features[f'{ticker}_volume'] = df['Volume']
        features[f'{ticker}_returns'] = df['Close'].pct_change()
        
        # 2. VOLATILITY REGIME (critical for >50% accuracy)
        returns = features[f'{ticker}_returns']
        features[f'{ticker}_volatility_20'] = returns.rolling(20).std()
        features[f'{ticker}_volatility_5'] = returns.rolling(5).std()
        features[f'{ticker}_vol_ratio'] = features[f'{ticker}_volatility_5'] / features[f'{ticker}_volatility_20']
        
        # Volatility percentile (market regime)
        features[f'{ticker}_vol_percentile'] = features[f'{ticker}_volatility_20'].rolling(252).rank(pct=True)
        
        # 3. TREND INDICATORS
        close_prices = df['Close']
        features[f'{ticker}_sma_20'] = close_prices.rolling(20).mean()
        features[f'{ticker}_sma_50'] = close_prices.rolling(50).mean()
        features[f'{ticker}_price_vs_sma20'] = close_prices / features[f'{ticker}_sma_20'] - 1
        features[f'{ticker}_trend_strength'] = features[f'{ticker}_sma_20'] / features[f'{ticker}_sma_50'] - 1
        
        # 4. VOLUME PATTERNS (smart money)
        volume = df['Volume']
        features[f'{ticker}_volume_ratio'] = volume / volume.rolling(20).mean()
        features[f'{ticker}_dollar_volume'] = close_prices * volume
        
        # Price-volume divergence
        price_up = returns > 0
        volume_down = volume < volume.rolling(20).mean()
        features[f'{ticker}_divergence'] = (price_up & volume_down).astype(float)
        
        # 5. MARKET RELATIVE FEATURES (key for >50%)
        if len(spy) > 0:
            spy_returns = spy.loc[common_dates, 'Close'].pct_change()
            features[f'{ticker}_vs_spy'] = returns - spy_returns
            features[f'{ticker}_beta'] = returns.rolling(20).corr(spy_returns)
            features[f'{ticker}_relative_strength'] = (close_prices / spy.loc[common_dates, 'Close']).pct_change(20)
        
        # 6. RANGE AND MOMENTUM
        high_prices = df['High']
        low_prices = df['Low']
        features[f'{ticker}_high_low_range'] = (high_prices - low_prices) / close_prices
        features[f'{ticker}_close_range_position'] = (close_prices - low_prices) / (high_prices - low_prices)
        
        # Multi-timeframe momentum
        for period in [5, 10, 20]:
            features[f'{ticker}_momentum_{period}'] = close_prices.pct_change(period)
        
        # 7. TIME PATTERNS (market anomalies)
        features[f'{ticker}_day_of_week'] = df.index.dayofweek
        features[f'{ticker}_month'] = df.index.month
        features[f'{ticker}_is_monday'] = (df.index.dayofweek == 0).astype(float)
        features[f'{ticker}_is_friday'] = (df.index.dayofweek == 4).astype(float)
        features[f'{ticker}_month_end'] = (df.index.day > 25).astype(float)
        
        features_list.append(features)
    
    # 8. MARKET-WIDE FEATURES (critical for context)
    market_features = pd.DataFrame(index=common_dates)
    
    if len(spy) > 0 and len(qqq) > 0:
        spy_returns = spy.loc[common_dates, 'Close'].pct_change()
        qqq_returns = qqq.loc[common_dates, 'Close'].pct_change()
        
        # Market regime
        market_features['market_volatility'] = spy_returns.rolling(20).std()
        market_features['market_trend'] = spy.loc[common_dates, 'Close'] / spy.loc[common_dates, 'Close'].rolling(50).mean() - 1
        
        # Tech vs broad market
        market_features['tech_vs_market'] = qqq_returns.rolling(20).mean() - spy_returns.rolling(20).mean()
        
        # Market breadth proxy
        market_features['spy_qqq_correlation'] = spy_returns.rolling(20).corr(qqq_returns)
        
        # Fear indicator
        if 'VIX' in data_dict and len(data_dict['VIX']) > 0:
            vix = data_dict['VIX'].loc[common_dates, 'Close']
            market_features['vix_level'] = vix
            market_features['vix_percentile'] = vix.rolling(252).rank(pct=True)
    
    features_list.append(market_features)
    
    # Combine all features
    all_features = pd.concat(features_list, axis=1)
    
    # Create targets (next-day AAPL return > 0)
    if 'AAPL_returns' in all_features.columns:
        all_features['target_return'] = all_features['AAPL_returns'].shift(-1)
        all_features['target'] = (all_features['target_return'] > 0).astype(float)
    else:
        # Fallback to MSFT
        all_features['target_return'] = all_features['MSFT_returns'].shift(-1)
        all_features['target'] = (all_features['target_return'] > 0).astype(float)
    
    # Drop NaN rows
    all_features = all_features.dropna()
    
    # Get feature columns (exclude targets)
    feature_cols = [col for col in all_features.columns if col not in ['target', 'target_return']]
    
    logger.info(f"Market-aware features created:")
    logger.info(f"  Shape: {all_features.shape}")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Date range: {all_features.index[0]} to {all_features.index[-1]}")
    
    # Feature categories
    vol_features = [col for col in feature_cols if 'vol' in col.lower()]
    market_features = [col for col in feature_cols if 'market' in col or 'spy' in col or 'qqq' in col]
    time_features = [col for col in feature_cols if 'day' in col or 'month' in col or 'friday' in col]
    
    logger.info(f"  Volatility features: {len(vol_features)}")
    logger.info(f"  Market features: {len(market_features)}")
    logger.info(f"  Time features: {len(time_features)}")
    
    return all_features, feature_cols

def prepare_ab_test_data():
    """Prepare data specifically for A/B testing with better features"""
    
    # Download fresh market data
    data_dict = download_market_data()
    
    if not data_dict:
        logger.error("Failed to download market data")
        return None
    
    # Create market-aware features
    features_df, feature_cols = create_market_aware_features(data_dict)
    
    # Save for A/B testing
    output_dir = "data/processed/market_aware"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    features_df.to_csv(os.path.join(output_dir, 'market_aware_features.csv'))
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")
    
    # Create train/val/test splits
    train_size = int(len(features_df) * 0.7)
    val_size = int(len(features_df) * 0.15)
    
    train_df = features_df[:train_size]
    val_df = features_df[train_size:train_size + val_size]
    test_df = features_df[train_size + val_size:]
    
    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    return features_df, feature_cols

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MARKET-AWARE FEATURE ENGINEERING")
    logger.info("Goal: Beat 50% accuracy with context")
    logger.info("=" * 60)
    
    features_df, feature_cols = prepare_ab_test_data()
    
    if features_df is not None:
        print(f"\n✅ Market-aware features created successfully!")
        print(f"📊 Total features: {len(feature_cols)}")
        print(f"📈 Ready for A/B testing with >50% potential")
        
        # Quick analysis
        target_mean = features_df['target'].mean()
        print(f"\n🎯 Target distribution: {target_mean:.1%} positive days")
        
        # Feature importance hints
        print(f"\n💡 Key feature categories for >50% accuracy:")
        print(f"  • Volatility regimes (low/medium/high)")
        print(f"  • Market relative performance (stock vs SPY)")
        print(f"  • Volume-price divergences")
        print(f"  • Time-based patterns")
        print(f"  • Inter-market relationships")
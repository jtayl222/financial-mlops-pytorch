#!/usr/bin/env python3
"""
Financial ML Train/Test Splits with proper time series handling
Addresses data leakage issues common in financial ML
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

class FinancialMLSplitter:
    """
    Proper financial ML train/test splits with:
    - Walk-forward validation
    - Purging gaps to prevent data leakage
    - Per-ticker handling
    - Time-based splits (not random)
    """
    
    def __init__(self, 
                 purge_gap_days: int = 5,
                 embargo_days: int = 2,
                 min_train_samples: int = 1000):
        """
        Args:
            purge_gap_days: Days to purge between train/validation/test sets
            embargo_days: Days to embargo after test set (for realistic backtesting)
            min_train_samples: Minimum samples needed for training
        """
        self.purge_gap_days = purge_gap_days
        self.embargo_days = embargo_days
        self.min_train_samples = min_train_samples
        
    def create_time_based_splits(self, 
                                data: pd.DataFrame,
                                train_end_date: str = "2021-12-31",
                                val_end_date: str = "2022-12-31",
                                test_end_date: str = "2023-12-31") -> Dict[str, pd.DataFrame]:
        """
        Create time-based splits with proper financial ML practices
        
        Args:
            data: DataFrame with DatetimeIndex
            train_end_date: End date for training set
            val_end_date: End date for validation set
            test_end_date: End date for test set
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Convert string dates to datetime
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        test_end = pd.to_datetime(test_end_date)
        
        # Calculate purge periods
        train_purge_start = train_end + timedelta(days=1)
        train_purge_end = train_purge_start + timedelta(days=self.purge_gap_days)
        
        val_purge_start = val_end + timedelta(days=1)
        val_purge_end = val_purge_start + timedelta(days=self.purge_gap_days)
        
        # Create splits with purging
        train_data = data[data.index <= train_end]
        val_data = data[(data.index >= train_purge_end) & (data.index <= val_end)]
        test_data = data[(data.index >= val_purge_end) & (data.index <= test_end)]
        
        # Log split statistics
        logging.info(f"Train period: {train_data.index.min()} to {train_data.index.max()}")
        logging.info(f"Validation period: {val_data.index.min()} to {val_data.index.max()}")
        logging.info(f"Test period: {test_data.index.min()} to {test_data.index.max()}")
        logging.info(f"Train samples: {len(train_data)}")
        logging.info(f"Validation samples: {len(val_data)}")
        logging.info(f"Test samples: {len(test_data)}")
        
        # Validate minimum samples
        if len(train_data) < self.min_train_samples:
            logging.warning(f"Training set has only {len(train_data)} samples, less than minimum {self.min_train_samples}")
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def create_walk_forward_splits(self, 
                                  data: pd.DataFrame,
                                  initial_train_months: int = 36,
                                  validation_months: int = 6,
                                  test_months: int = 6,
                                  step_months: int = 3) -> List[Dict[str, pd.DataFrame]]:
        """
        Create walk-forward validation splits
        
        Args:
            data: DataFrame with DatetimeIndex
            initial_train_months: Initial training window in months
            validation_months: Validation window in months
            test_months: Test window in months
            step_months: Step size for walk-forward in months
            
        Returns:
            List of split dictionaries
        """
        
        splits = []
        data = data.sort_index()
        
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Calculate initial training end
        current_train_end = start_date + pd.DateOffset(months=initial_train_months)
        
        while current_train_end < end_date:
            # Define periods
            val_start = current_train_end + timedelta(days=self.purge_gap_days)
            val_end = val_start + pd.DateOffset(months=validation_months)
            
            test_start = val_end + timedelta(days=self.purge_gap_days)
            test_end = test_start + pd.DateOffset(months=test_months)
            
            # Check if we have enough data
            if test_end > end_date:
                break
                
            # Create splits
            train_data = data[data.index <= current_train_end]
            val_data = data[(data.index >= val_start) & (data.index <= val_end)]
            test_data = data[(data.index >= test_start) & (data.index <= test_end)]
            
            if len(train_data) >= self.min_train_samples and len(val_data) > 0 and len(test_data) > 0:
                splits.append({
                    'train': train_data,
                    'validation': val_data,
                    'test': test_data,
                    'train_end': current_train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
                
                logging.info(f"Walk-forward split {len(splits)}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            
            # Move forward
            current_train_end += pd.DateOffset(months=step_months)
        
        return splits
    
    def create_per_ticker_splits(self, 
                                ticker_data: Dict[str, pd.DataFrame],
                                split_method: str = "time_based",
                                **kwargs) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create splits for each ticker separately to avoid cross-contamination
        
        Args:
            ticker_data: Dictionary mapping ticker -> DataFrame
            split_method: "time_based" or "walk_forward"
            **kwargs: Arguments for split method
            
        Returns:
            Dictionary mapping ticker -> {train, validation, test}
        """
        
        ticker_splits = {}
        
        for ticker, data in ticker_data.items():
            logging.info(f"Creating splits for {ticker}")
            
            if split_method == "time_based":
                splits = self.create_time_based_splits(data, **kwargs)
            elif split_method == "walk_forward":
                splits = self.create_walk_forward_splits(data, **kwargs)
                # For per-ticker, use only the first walk-forward split
                if splits:
                    splits = splits[0]
                else:
                    logging.warning(f"No valid splits created for {ticker}")
                    continue
            else:
                raise ValueError(f"Unknown split method: {split_method}")
            
            ticker_splits[ticker] = splits
        
        return ticker_splits
    
    def combine_ticker_splits(self, 
                             ticker_splits: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Combine per-ticker splits into unified train/val/test sets
        
        Args:
            ticker_splits: Output from create_per_ticker_splits
            
        Returns:
            Combined train/validation/test DataFrames
        """
        
        combined_splits = {'train': [], 'validation': [], 'test': []}
        
        for ticker, splits in ticker_splits.items():
            for split_name, split_data in splits.items():
                if split_name in combined_splits:
                    # Add ticker column for identification
                    split_data_copy = split_data.copy()
                    split_data_copy['ticker'] = ticker
                    combined_splits[split_name].append(split_data_copy)
        
        # Concatenate and sort by date
        result = {}
        for split_name, split_list in combined_splits.items():
            if split_list:
                combined_df = pd.concat(split_list).sort_index()
                result[split_name] = combined_df
                logging.info(f"Combined {split_name} set: {len(combined_df)} samples from {len(split_list)} tickers")
        
        return result

def demonstrate_financial_splits():
    """
    Demonstrate proper financial ML splits vs naive approach
    """
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    
    # Sample ticker data
    ticker_data = {}
    for ticker in ['IBB', 'XBI', 'JNJ']:
        data = pd.DataFrame({
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Target': np.random.choice([0, 1], len(dates))
        }, index=dates)
        ticker_data[ticker] = data
    
    # Initialize splitter
    splitter = FinancialMLSplitter(purge_gap_days=5, embargo_days=2)
    
    # Method 1: Time-based splits per ticker
    print("=== Time-based splits per ticker ===")
    per_ticker_splits = splitter.create_per_ticker_splits(ticker_data, split_method="time_based")
    combined_splits = splitter.combine_ticker_splits(per_ticker_splits)
    
    # Method 2: Walk-forward validation
    print("\n=== Walk-forward validation (first ticker only) ===")
    walk_forward_splits = splitter.create_walk_forward_splits(ticker_data['IBB'], initial_train_months=36)
    print(f"Created {len(walk_forward_splits)} walk-forward splits")
    
    return combined_splits, walk_forward_splits

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_financial_splits()
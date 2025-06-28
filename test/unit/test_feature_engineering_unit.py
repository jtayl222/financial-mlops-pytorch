import pytest
import pandas as pd
import numpy as np
import torch
from src.feature_engineering_pytorch import (
    calculate_lagged_features, calculate_sma, calculate_rsi,
    create_target, FinancialTimeSeriesDataset, process_ticker_data,
    SEQUENCE_LENGTH, PROCESSED_DATA_DIR # Import relevant constants
)
from sklearn.preprocessing import MinMaxScaler
import os

@pytest.fixture
def sample_dataframe():
    # Increase periods to at least 40 to support all rolling/lags (e.g., SMA_20, RSI_14, lags up to 5, and target horizon)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=40, freq='D'))
    return pd.DataFrame({
        'Close': np.arange(100, 140).astype(float),
        'Volume': np.arange(1000, 1040).astype(float) * 10,
        'Open': np.arange(99, 139).astype(float),
        'High': np.arange(101, 141).astype(float),
        'Low': np.arange(98, 138).astype(float),
        'Adj Close': np.arange(100, 140).astype(float)
    }, index=dates)

def test_calculate_lagged_features(sample_dataframe):
    df_lagged = calculate_lagged_features(sample_dataframe.copy(), 'Close', lags=2)
    assert 'Close_lag_1' in df_lagged.columns
    assert 'Close_lag_2' in df_lagged.columns
    assert df_lagged['Close_lag_1'].iloc[1] == 100 # Lag of 100 for row with 101
    assert pd.isna(df_lagged['Close_lag_1'].iloc[0]) # First lag is NaN
    assert pd.isna(df_lagged['Close_lag_2'].iloc[1]) # Second lag at second row is NaN

def test_calculate_sma(sample_dataframe):
    df_sma = calculate_sma(sample_dataframe.copy(), 'Close', window=3)
    assert 'SMA_Close_3' in df_sma.columns
    assert df_sma['SMA_Close_3'].iloc[2] == (100 + 101 + 102) / 3
    assert pd.isna(df_sma['SMA_Close_3'].iloc[0])

def test_calculate_rsi(sample_dataframe):
    # RSI needs at least window + 1 data points for the first valid calculation
    df_rsi = calculate_rsi(sample_dataframe.copy(), 'Close', window=14)
    assert 'RSI_Close_14' in df_rsi.columns
    assert pd.isna(df_rsi['RSI_Close_14'].iloc[12])  # Last NaN at index 12
    assert not pd.isna(df_rsi['RSI_Close_14'].iloc[13])  # First valid at index 13

def test_create_target(sample_dataframe):
    df_target = create_target(sample_dataframe.copy(), 'Close', horizon=1)
    assert 'Target' in df_target.columns
    assert df_target['Target'].iloc[0] == 1 # 101 > 100 (True -> 1)
    assert pd.isna(df_target['Target'].iloc[-1]) # Last row has no future price

def test_financial_time_series_dataset():
    features = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    targets = np.array([0, 1, 0, 1, 0])
    seq_len = 3

    dataset = FinancialTimeSeriesDataset(features, targets, seq_len)
    assert len(dataset) == len(features) - seq_len + 1 # 5 - 3 + 1 = 3 sequences

    # First sequence: features = [[1,2], [3,4], [5,6]], target = 0 (target for index 2)
    seq0, target0 = dataset[0]
    np.testing.assert_array_equal(seq0.numpy(), [[1, 2], [3, 4], [5, 6]])
    assert target0.item() == 0

    # Second sequence: features = [[3,4], [5,6], [7,8]], target = 1 (target for index 3)
    seq1, target1 = dataset[1]
    np.testing.assert_array_equal(seq1.numpy(), [[3, 4], [5, 6], [7, 8]])
    assert target1.item() == 1

    # Test for index out of bounds
    with pytest.raises(IndexError):
        dataset[len(dataset)]

def test_process_ticker_data_integration(tmp_path, sample_dataframe):
    # Simulate data ingestion output
    sample_dataframe.index.name = "Date"  # Ensure index is named 'Date' for CSV
    temp_raw_file = tmp_path / "AAPL_raw_2023-01-01_2023-01-20.csv"
    sample_dataframe.to_csv(temp_raw_file, index=True)

    processed_df = process_ticker_data(str(temp_raw_file))

    # Debug output to help diagnose test failure
    print("Processed DataFrame columns:", processed_df.columns.tolist())
    print(processed_df.head())

    # Basic checks on processed_df
    assert not processed_df.empty
    assert 'Target' in processed_df.columns
    # Check for presence of engineered features
    assert 'Close_lag_1' in processed_df.columns
    assert 'SMA_Close_5' in processed_df.columns
    assert 'RSI_Close_14' in processed_df.columns
    assert 'Daily_Return' in processed_df.columns
    # Check for NaNs after processing (should be none due to dropna)
    assert not processed_df.isnull().any().any()
    # Check that original columns are removed
    assert 'Open' not in processed_df.columns
    assert 'Close' not in processed_df.columns
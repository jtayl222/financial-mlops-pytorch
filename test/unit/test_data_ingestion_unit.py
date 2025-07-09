import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os
from src.data_ingestion import download_stock_data, save_data, RAW_DATA_DIR


# Fixture to create a temporary directory for raw data
@pytest.fixture
def temp_raw_data_dir(tmp_path):
    global RAW_DATA_DIR
    # tmp_path is a built-in pytest fixture that provides a unique temporary directory
    original_raw_data_dir = RAW_DATA_DIR
    # Temporarily change RAW_DATA_DIR for testing
    RAW_DATA_DIR = str(tmp_path / "test_raw_data")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    yield RAW_DATA_DIR
    # Clean up (pytest tmp_path handles deletion, but reset global if necessary)
    RAW_DATA_DIR = original_raw_data_dir


@patch('yfinance.download')
def test_download_stock_data_success(mock_yf_download):
    # Mock successful download
    mock_data = pd.DataFrame({
        'Open': [100, 101],
        'High': [102, 103],
        'Low': [99, 100],
        'Close': [101, 102],
        'Adj Close': [101, 102],
        'Volume': [1000, 1100]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
    mock_yf_download.return_value = mock_data

    df = download_stock_data("TEST", "2023-01-01", "2023-01-02")
    assert not df.empty
    assert 'Close' in df.columns
    mock_yf_download.assert_called_once_with("TEST", start="2023-01-01", end="2023-01-02", progress=False)


@patch('yfinance.download')
def test_download_stock_data_empty(mock_yf_download):
    # Mock empty data returned
    mock_yf_download.return_value = pd.DataFrame()
    df = download_stock_data("NONEXISTENT", "2023-01-01", "2023-01-02")
    assert df.empty


@patch('yfinance.download', side_effect=Exception("Network error"))
def test_download_stock_data_failure(mock_yf_download):
    """Tests the behavior when yfinance.download raises an exception."""
    df = download_stock_data("FAIL", "2023-01-01", "2023-01-02")
    assert df.empty
    mock_yf_download.assert_called_once_with("FAIL", start="2023-01-01", end="2023-01-02", progress=False)


def test_save_data(temp_raw_data_dir):
    test_df = pd.DataFrame({
        'Col1': [1, 2],
        'Col2': [3, 4]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02']))

    file_path = os.path.join(temp_raw_data_dir, "test_ticker.csv")
    save_data(test_df, file_path, "TEST_TICKER")

    assert os.path.exists(file_path)
    loaded_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(test_df, loaded_df)

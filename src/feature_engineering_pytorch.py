# src/feature_engineering_pytorch.py
import pandas as pd
import numpy as np
import os
import logging
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler  # Or StandardScaler
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
SCALER_DIR = os.environ.get("SCALER_DIR", "artifacts/scalers")  # Directory to save scalers
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# Time series parameters
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 10))  # Number of past days to consider for prediction
PREDICTION_HORIZON = 1  # Predict the next day
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% for training + validation, 20% for testing
TRAIN_VALIDATION_SPLIT_RATIO = 0.85  # 85% of training data for training, 15% for validation

# Feature Engineering parameters
SMA_WINDOWS = [5, 10, 20]
RSI_WINDOWS = [14]  # Common RSI window


# --- PyTorch Custom Dataset Class ---
class FinancialTimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for financial time series data.
    Assumes data is already scaled and ready for sequence creation.
    """

    def __init__(self, features, targets, sequence_length):
        self.features = features  # NumPy array: (num_samples, num_features)
        self.targets = targets  # NumPy array: (num_samples,)
        self.sequence_length = sequence_length

        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets must have the same number of samples.")

    def __len__(self):
        # The number of available sequences
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        # idx refers to the starting index of the sequence
        if idx >= (len(self.features) - self.sequence_length + 1):
            raise IndexError("Index out of bounds for sequence creation.")

        sequence = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length - 1]  # Target is for the end of the sequence

        # Ensure tensors are float32 for features and float32 for targets (for BCELoss with Sigmoid output)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# --- Feature Engineering Functions ---

def calculate_lagged_features(df: pd.DataFrame, column: str, lags: int = 5) -> pd.DataFrame:
    """Calculates lagged values for a given column."""
    for i in range(1, lags + 1):
        df[f'{column}_lag_{i}'] = df[column].shift(i)
    return df


def calculate_sma(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Calculates Simple Moving Average."""
    df[f'SMA_{column}_{window}'] = df[column].rolling(window=window).mean()
    return df


def calculate_rsi(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df[f'RSI_{column}_{window}'] = 100 - (100 / (1 + rs))
    return df


def create_target(df: pd.DataFrame, column: str, horizon: int = 1) -> pd.DataFrame:
    """
    Creates a binary target: 1 if the price goes up, 0 otherwise.
    Predicts the direction 'horizon' days into the future.
    """
    future_price = df[column].shift(-horizon)
    df['Target'] = (future_price > df[column]).astype(float)
    df.loc[future_price.isna(), 'Target'] = np.nan
    return df


# --- Main Feature Engineering and Data Preparation Logic ---
def process_ticker_data(file_path: str):
    """
    Loads raw data for a ticker, performs feature engineering, creates target,
    and returns the processed DataFrame.
    """
    logging.info(f"Processing file: {file_path}")
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df = df.sort_index()  # Ensure chronological order

    # Drop rows with any NaN values that might prevent feature calculation
    initial_rows = len(df)
    df.dropna(inplace=True)
    if len(df) != initial_rows:
        logging.warning(f"Dropped {initial_rows - len(df)} rows due to initial NaNs for {os.path.basename(file_path)}")

    # 1. Calculate Lagged Features (e.g., lagged Close prices and Volume)
    df = calculate_lagged_features(df, 'Close', lags=5)
    df = calculate_lagged_features(df, 'Volume', lags=3)

    # 2. Calculate Moving Averages (on Close price)
    for window in SMA_WINDOWS:
        df = calculate_sma(df, 'Close', window)

    # 3. Calculate RSI (on Close price)
    for window in RSI_WINDOWS:
        df = calculate_rsi(df, 'Close', window)

    # 4. Calculate Daily Returns (Simple returns)
    df['Daily_Return'] = df['Close'].pct_change()

    # 5. Create Target Variable
    df = create_target(df, 'Close', horizon=PREDICTION_HORIZON)

    # Drop any remaining NaNs created by feature engineering and target creation
    # This will remove the initial rows where lags/rolling windows don't have enough data
    # and the last 'prediction_horizon' rows where target is NaN.
    initial_rows_after_fe = len(df)
    df.dropna(inplace=True)
    if len(df) != initial_rows_after_fe:
        logging.warning(
            f"Dropped {initial_rows_after_fe - len(df)} rows due to NaNs after feature engineering for {os.path.basename(file_path)}")

    # Select features for the model
    # Exclude original 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
    # And of course, 'Target' itself
    feature_columns = [col for col in df.columns if
                       col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']]

    # It's good practice to ensure all selected feature columns are numeric
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=feature_columns, inplace=True)  # Drop rows if features are NaN after coercion

    logging.info(f"Processed {len(df)} rows for {os.path.basename(file_path)}. Features: {feature_columns}")
    return df[feature_columns + ['Target']]


# --- Main execution ---
if __name__ == "__main__":
    logging.info("Starting feature engineering and PyTorch data preparation process.")

    # Start an MLflow run
    with mlflow.start_run(run_name="feature_engineering_pytorch"):
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("prediction_horizon", PREDICTION_HORIZON)
        mlflow.log_param("train_test_split_ratio", TRAIN_TEST_SPLIT_RATIO)
        mlflow.log_param("train_validation_split_ratio", TRAIN_VALIDATION_SPLIT_RATIO)
        mlflow.log_param("sma_windows", SMA_WINDOWS)
        mlflow.log_param("rsi_windows", RSI_WINDOWS)
        mlflow.log_param("raw_data_input_dir", RAW_DATA_DIR)
        mlflow.log_param("processed_data_output_dir", PROCESSED_DATA_DIR)
        mlflow.log_param("scaler_output_dir", SCALER_DIR)

        all_tickers_data = []
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]

        if not raw_files:
            logging.error(f"No raw CSV files found in {RAW_DATA_DIR}. Please run data_ingestion.py first.")
            exit()

        for filename in raw_files:
            ticker_df = process_ticker_data(os.path.join(RAW_DATA_DIR, filename))
            if not ticker_df.empty:
                all_tickers_data.append(ticker_df)
            else:
                logging.warning(f"Skipping {filename} due to empty processed data.")

        if not all_tickers_data:
            logging.error("No data processed for any ticker. Exiting.")
            exit()

        # Concatenate all ticker data
        # IMPORTANT: When combining multiple tickers, ensure that the time index is handled carefully
        # For an LSTM, you'd typically train one model per ticker, or a single model with a ticker ID.
        # For simplicity here, we'll combine and then split. For production, consider per-ticker models.
        combined_df = pd.concat(all_tickers_data).sort_index()

        logging.info(f"Combined data shape after feature engineering: {combined_df.shape}")

        # Define features and target after combining and ensuring all NaNs are handled
        features_df = combined_df.drop(columns=['Target'])
        target_series = combined_df['Target']

        # --- Data Splitting (Time-series aware) ---
        # Get the split index based on chronological order
        train_val_split_idx = int(len(combined_df) * TRAIN_TEST_SPLIT_RATIO)

        # Train + Validation set
        train_val_features_df = features_df.iloc[:train_val_split_idx]
        train_val_target_series = target_series.iloc[:train_val_split_idx]

        # Test set
        test_features_df = features_df.iloc[train_val_split_idx:]
        test_target_series = target_series.iloc[train_val_split_idx:]

        # Train and Validation split
        train_split_idx = int(len(train_val_features_df) * TRAIN_VALIDATION_SPLIT_RATIO)

        train_features_df = train_val_features_df.iloc[:train_split_idx]
        train_target_series = train_val_target_series.iloc[:train_split_idx]

        val_features_df = train_val_features_df.iloc[train_split_idx:]
        val_target_series = train_val_target_series.iloc[train_split_idx:]

        logging.info(f"Train features shape: {train_features_df.shape}")
        logging.info(f"Validation features shape: {val_features_df.shape}")
        logging.info(f"Test features shape: {test_features_df.shape}")

        # --- Feature Scaling ---
        # Fit scaler ONLY on the training data to prevent data leakage
        scaler = MinMaxScaler()  # Or StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features_df)
        val_features_scaled = scaler.transform(val_features_df)
        test_features_scaled = scaler.transform(test_features_df)

        # Save the scaler
        scaler_file_path = os.path.join(SCALER_DIR, 'minmax_scaler.pkl')
        with open(scaler_file_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_file_path, artifact_path="scalers")
        logging.info(f"Scaler saved to {scaler_file_path} and logged to MLflow.")

        # --- Create PyTorch Datasets and DataLoaders ---
        train_dataset = FinancialTimeSeriesDataset(
            features=train_features_scaled,
            targets=train_target_series.values,
            sequence_length=SEQUENCE_LENGTH
        )
        val_dataset = FinancialTimeSeriesDataset(
            features=val_features_scaled,
            targets=val_target_series.values,
            sequence_length=SEQUENCE_LENGTH
        )
        test_dataset = FinancialTimeSeriesDataset(
            features=test_features_scaled,
            targets=test_target_series.values,
            sequence_length=SEQUENCE_LENGTH
        )

        logging.info(f"Number of training sequences: {len(train_dataset)}")
        logging.info(f"Number of validation sequences: {len(val_dataset)}")
        logging.info(f"Number of test sequences: {len(test_dataset)}")

        # Save processed data as NumPy arrays or Parquet/CSV
        # NumPy .npy format is convenient for tensors
        np.save(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'), train_dataset.features)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'train_targets.npy'), train_dataset.targets)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'val_features.npy'), val_dataset.features)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'val_targets.npy'), val_dataset.targets)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'test_features.npy'), test_dataset.features)
        np.save(os.path.join(PROCESSED_DATA_DIR, 'test_targets.npy'), test_dataset.targets)

        # Log processed data as artifacts
        mlflow.log_artifact(os.path.join(PROCESSED_DATA_DIR, 'train_features.npy'), artifact_path="processed_data")
        mlflow.log_artifact(os.path.join(PROCESSED_DATA_DIR, 'train_targets.npy'), artifact_path="processed_data")
        mlflow.log_artifact(os.path.join(PROCESSED_DATA_DIR, 'val_features.npy'), artifact_path="processed_data")
        mlflow.log_artifact(os.path.join(PROCESSED_DATA_DIR, 'val_targets.npy'), artifact_path="processed_data")
        mlflow.log_artifact(os.path.join(PROCESSED_DATA_DIR, 'test_features.npy'), artifact_path="processed_data")
        mlflow.log_artifact(os.path.join(PROCESSED_DATA_DIR, 'test_targets.npy'), artifact_path="processed_data")
        logging.info("Processed data saved and logged to MLflow.")

        mlflow.log_metric("num_train_sequences", len(train_dataset))
        mlflow.log_metric("num_val_sequences", len(val_dataset))
        mlflow.log_metric("num_test_sequences", len(test_dataset))
        mlflow.log_metric("num_features", train_features_scaled.shape[1])

        logging.info("Feature engineering and PyTorch data preparation completed successfully.")
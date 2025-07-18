# src/data_ingestion.py
import yfinance as yf
import pandas as pd
import os
import datetime
import logging
import mlflow

# Configure logging
loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, loglevel), format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# List of ticker symbols to download - IBB + Healthcare + Market Context (12 assets)
TICKERS = [
    "IBB",   # iShares Biotech ETF (Primary target)
    "XBI",   # SPDR Biotech ETF (Alternative biotech)
    "JNJ",   # Johnson & Johnson (Healthcare mega-cap anchor)
    "SPY",   # S&P 500 (Broad market benchmark)
    "QQQ",   # Nasdaq 100 (Tech-heavy market exposure)
    "XLV",   # Health Care Select Sector (Healthcare sector)
    "AMGN",  # Amgen (Biotech leader)
    "PFE",   # Pfizer (Big pharma)
    "VTI",   # Total Stock Market (Broad market exposure)
    "XLRE",  # Real Estate Select Sector (Different sector correlation)
    "MRNA",  # Moderna (Individual biotech stock)
    "TMO",   # Thermo Fisher Scientific (Life sciences tools)
]

# Define the start and end dates for data ingestion
# Using a fixed start date for reproducibility, and current date as end
START_DATE = "2018-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d") # Current date

# Configuration from environment variables, with fallbacks for local dev/testing
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
START_DATE = os.getenv("INGESTION_START_DATE", "2018-01-01")
END_DATE = os.getenv("INGESTION_END_DATE", "2023-12-31")
TICKERS = os.getenv("TICKERS", "IBB,XBI,JNJ,SPY,QQQ,XLV,AMGN,PFE,VTI,XLRE,MRNA,TMO").split(',') # 12-asset biotech/healthcare ecosystem

# Output directory for raw data
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def run_ingestion_pipeline():
    # Your existing logic, now using the ENV-configurable variables
    with mlflow.start_run(run_name="data_ingestion"):
        mlflow.log_param("RAW_DATA_DIR", RAW_DATA_DIR)
        mlflow.log_param("tickers", TICKERS)
        mlflow.log_param("start_date", START_DATE)
        mlflow.log_param("end_date", END_DATE)
        for ticker in TICKERS:
            df = download_stock_data(ticker, START_DATE, END_DATE)
            if not df.empty:
                # Construct file path using the configurable RAW_DATA_DIR
                file_name = f"{ticker}_raw_{START_DATE}_{END_DATE}.csv"
                file_path = os.path.join(RAW_DATA_DIR, file_name)
                save_data(df, file_path, ticker)
                mlflow.log_artifact(file_path, artifact_path="raw_data")
        print(f"Data ingestion complete. Raw data saved to: {RAW_DATA_DIR}")


# --- Functions ---
def download_stock_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical stock data for a given ticker symbol.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing historical data, or an empty DataFrame if failed.
    """
    logging.info(f"Attempting to download data for {ticker_symbol} from {start_date} to {end_date}")
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            logging.warning(f"No data downloaded for {ticker_symbol}. It might be an invalid ticker or no data in the period.")
            return pd.DataFrame()
        logging.info(f"Successfully downloaded {len(data)} rows for {ticker_symbol}.")
        return data
    except Exception as e:
        logging.error(f"Error downloading data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def save_data(dataframe: pd.DataFrame, file_path: str, ticker: str):
    """
    Saves a DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        file_path (str): The full path to save the CSV file.
        ticker (str): The ticker symbol for logging purposes.
    """
    try:
        dataframe.index.name = "Date"  # Ensure index is named 'Date' for CSV compatibility
        # Flatten MultiIndex columns if present
        if isinstance(dataframe.columns, pd.MultiIndex):
            dataframe.columns = ['_'.join([str(i) for i in col if i]) for col in dataframe.columns.values]
        dataframe.to_csv(file_path, index=True) # Save with Date index
        logging.info(f"Data for {ticker} saved to {file_path}")
        # Log the columns written to the CSV for debugging
        logging.info(f"Columns written for {ticker}: {[dataframe.index.name] + list(dataframe.columns)}")
    except Exception as e:
        logging.error(f"Error saving data for {ticker} to {file_path}: {e}")

# --- Main execution ---
if __name__ == "__main__":
    logging.info("Starting data ingestion process.")

    # Print environment variables for debugging
    print(f"LOGLEVEL={os.environ.get('LOGLEVEL')}")
    for k, v in os.environ.items():
        logging.debug(f"{k}={v}")

    # Log output directory and tickers
    logging.info(f"Configured RAW_DATA_DIR: {RAW_DATA_DIR}")
    logging.info(f"Configured TICKERS: {TICKERS}")
    logging.info(f"Configured START_DATE: {START_DATE}")
    logging.info(f"Configured END_DATE: {END_DATE}")

    # Start an MLflow run
    # This ensures all parameters, metrics, and artifacts are logged under one run
    with mlflow.start_run(run_name="data_ingestion"):
        mlflow.log_param("start_date", START_DATE)
        mlflow.log_param("end_date", END_DATE)
        mlflow.log_param("tickers", TICKERS)
        mlflow.log_param("RAW_DATA_DIR", RAW_DATA_DIR)

        total_files_downloaded = 0
        total_rows_downloaded = 0

        for ticker in TICKERS:
            logging.info(f"Downloading data for ticker: {ticker}")
            df = download_stock_data(ticker, START_DATE, END_DATE)

            if not df.empty:
                file_name = f"{ticker}_raw_{START_DATE}_{END_DATE}.csv"
                file_path = os.path.join(RAW_DATA_DIR, file_name)
                save_data(df, file_path, ticker)
                logging.info(f"Saved data for {ticker} to {file_path} with {len(df)} rows.")

                total_files_downloaded += 1
                total_rows_downloaded += len(df)

                # Log each individual raw data file as an artifact
                mlflow.log_artifact(file_path, artifact_path="raw_data")
            else:
                logging.warning(f"No data downloaded for ticker: {ticker}")

        # Log overall metrics for the ingestion run
        mlflow.log_metric("total_tickers_attempted", len(TICKERS))
        mlflow.log_metric("total_files_downloaded", total_files_downloaded)
        mlflow.log_metric("total_rows_downloaded", total_rows_downloaded)

        logging.info(f"Total files downloaded: {total_files_downloaded}")
        logging.info(f"Total rows downloaded: {total_rows_downloaded}")

        if total_files_downloaded > 0:
            logging.info(f"Data ingestion completed. Downloaded data for {total_files_downloaded}/{len(TICKERS)} tickers.")
        else:
            logging.error("Data ingestion failed: No data downloaded for any ticker.")
import pandera as pa
import pandas as pd
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_schema():
    """Defines the Pandera schema for the raw stock data."""
    schema = pa.DataFrameSchema(
        {
            "Open": pa.Column(float, checks=pa.Check.ge(0)),
            "High": pa.Column(float, checks=pa.Check.ge(0)),
            "Low": pa.Column(float, checks=pa.Check.ge(0)),
            "Close": pa.Column(float, checks=pa.Check.ge(0)),
            "Adj Close": pa.Column(float, checks=pa.Check.ge(0)),
            "Volume": pa.Column(int, checks=pa.Check.ge(0)),
        },
        index=pa.Index(pd.DatetimeTZDtype(tz='America/New_York')),
        strict=True,  # Ensures no extra columns are present
        coerce=True,  # Coerces types to match schema
    )
    return schema

def validate_data(file_path: str):
    """Validates a given CSV file against the predefined schema."""
    try:
        logging.info(f"Reading data from {file_path}...")
        # Load data, ensuring the 'Date' column is parsed correctly as timezone-aware
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

        logging.info("Validating data against the schema...")
        schema = get_schema()
        schema.validate(df, lazy=True)  # lazy=True reports all validation errors
        logging.info(f"Successfully validated {file_path}. Data is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        logging.error(f"Data validation failed for {file_path}.")
        logging.error("Schema Error Details:\n" + str(err.failure_cases))
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during validation of {file_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_data.py <path_to_csv_file>")
        sys.exit(1)

    file_to_validate = sys.argv[1]
    if not validate_data(file_to_validate):
        sys.exit(1) # Exit with a non-zero code to indicate failure in a pipeline

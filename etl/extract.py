"""
ETL Extract Module — Load raw flight data from CSV.
Source: BTS On-Time Performance Dataset (synthetic)
"""

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

EXPECTED_COLUMNS = [
    'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
    'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'Origin', 'Dest',
    'CRSDepTime', 'DepDelay', 'DepDelayMinutes', 'TaxiOut', 'TaxiIn',
    'ArrDelay', 'ArrDelayMinutes', 'Cancelled', 'Diverted',
    'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Distance',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'
]


def extract(filepath: str = None) -> pd.DataFrame:
    """
    Load raw flight data from CSV.

    Args:
        filepath: Path to raw CSV. If None, uses default location.

    Returns:
        Raw DataFrame with validated schema.
    """
    if filepath is None:
        filepath = os.path.join(RAW_DATA_DIR, "flights_raw.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    logger.info(f"📂 Extracting data from: {filepath}")
    df = pd.read_csv(filepath)

    # Validate schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.warning(f"⚠️ Missing expected columns: {missing_cols}")

    logger.info(f"✅ Extracted {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"   Date range: {df['FlightDate'].min()} to {df['FlightDate'].max()}")
    logger.info(f"   Carriers: {df['Reporting_Airline'].nunique()}")

    return df


if __name__ == "__main__":
    df = extract()
    print(df.head())
    print(f"\nShape: {df.shape}")

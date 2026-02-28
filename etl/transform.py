"""
ETL Transform Module — Clean, feature-engineer, and prepare flight data for ML.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw flight data into ML-ready features.

    Steps:
    1. Drop cancelled/diverted flights
    2. Handle missing values
    3. Create target variable (is_delayed)
    4. Feature engineering
    5. Encode categorical variables
    6. Select final features

    Args:
        df: Raw DataFrame from extract step.

    Returns:
        Processed DataFrame ready for ML training.
    """
    logger.info(f"🔧 Transforming {len(df)} rows...")
    df = df.copy()

    # ---------- 1. Filter out cancelled/diverted ----------
    initial_size = len(df)
    df = df[df['Cancelled'] == 0]
    df = df[df['Diverted'] == 0]
    logger.info(f"   Removed {initial_size - len(df)} cancelled/diverted flights")

    # ---------- 2. Handle missing values ----------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ---------- 3. Create target variable ----------
    # FAA definition: a flight is "delayed" if arrival delay > 15 minutes
    df['is_delayed'] = (df['ArrDelay'] > 15).astype(int)
    delay_rate = df['is_delayed'].mean()
    logger.info(f"   Delay rate: {delay_rate:.1%}")

    # ---------- 4. Feature engineering ----------
    # Time features from CRSDepTime
    df['dep_hour'] = (df['CRSDepTime'] // 100).astype(int)
    df['dep_minute'] = (df['CRSDepTime'] % 100).astype(int)

    # Time of day buckets
    df['time_of_day'] = pd.cut(
        df['dep_hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )

    # Weekend flag
    df['is_weekend'] = (df['DayOfWeek'] >= 6).astype(int)

    # Season
    df['season'] = df['Month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })

    # Distance buckets
    df['distance_group'] = pd.cut(
        df['Distance'],
        bins=[0, 500, 1000, 2000, 6000],
        labels=['short', 'medium', 'long', 'ultra_long']
    )

    # Departure delay indicator (useful as feature for arrival delay)
    df['dep_delay_flag'] = (df['DepDelay'] > 0).astype(int)

    # Log distance
    df['log_distance'] = np.log1p(df['Distance'])

    # ---------- 5. Encode categorical variables ----------
    # Label encode carriers
    carrier_map = {c: i for i, c in enumerate(sorted(df['Reporting_Airline'].unique()))}
    df['carrier_encoded'] = df['Reporting_Airline'].map(carrier_map)

    # Frequency encode origin/dest
    origin_freq = df['Origin'].value_counts(normalize=True).to_dict()
    dest_freq = df['Dest'].value_counts(normalize=True).to_dict()
    df['origin_freq'] = df['Origin'].map(origin_freq)
    df['dest_freq'] = df['Dest'].map(dest_freq)

    # One-hot encode time_of_day and season
    df = pd.get_dummies(df, columns=['time_of_day', 'season', 'distance_group'], drop_first=True)

    # ---------- 6. Select final feature columns ----------
    feature_cols = [
        'Month', 'DayofMonth', 'DayOfWeek', 'dep_hour', 'dep_minute',
        'carrier_encoded', 'origin_freq', 'dest_freq',
        'Distance', 'log_distance', 'is_weekend', 'dep_delay_flag',
        'DepDelay', 'TaxiOut', 'CRSElapsedTime'
    ]

    # Add one-hot encoded columns
    ohe_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in
        ['time_of_day_', 'season_', 'distance_group_']
    )]
    feature_cols.extend(ohe_cols)

    # Ensure all feature columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Keep metadata columns for tracking
    meta_cols = ['Year', 'FlightDate', 'Reporting_Airline', 'Origin', 'Dest',
                 'ArrDelay', 'Flight_Number_Reporting_Airline']
    meta_cols = [c for c in meta_cols if c in df.columns]

    output_cols = meta_cols + feature_cols + ['is_delayed']
    df_out = df[output_cols].copy()

    # Ensure no NaN in features
    df_out[feature_cols] = df_out[feature_cols].fillna(0)

    logger.info(f"✅ Transform complete: {len(df_out)} rows, {len(feature_cols)} features")
    logger.info(f"   Features: {feature_cols}")

    return df_out


if __name__ == "__main__":
    from extract import extract
    raw_df = extract()
    processed_df = transform(raw_df)
    print(processed_df.head())
    print(f"\nShape: {processed_df.shape}")
    print(f"Target distribution:\n{processed_df['is_delayed'].value_counts(normalize=True)}")

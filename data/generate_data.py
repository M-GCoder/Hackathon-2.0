"""
Generate realistic synthetic BTS On-Time Performance flight data.
Simulates data from the U.S. Bureau of Transportation Statistics.
Source format: https://transtats.bts.gov/
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Configuration
NUM_RECORDS = 50000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "raw", "flights_raw.csv")

# Realistic airline carriers (IATA codes)
CARRIERS = {
    'AA': 'American Airlines',
    'DL': 'Delta Air Lines',
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'NK': 'Spirit Airlines',
    'F9': 'Frontier Airlines',
    'G4': 'Allegiant Air',
    'HA': 'Hawaiian Airlines'
}

# Major US airports
AIRPORTS = [
    'ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MCO', 'LAS',
    'PHX', 'MIA', 'SEA', 'IAH', 'JFK', 'EWR', 'SFO', 'MSP',
    'BOS', 'DTW', 'FLL', 'PHL', 'LGA', 'BWI', 'SLC', 'SAN',
    'IAD', 'DCA', 'MDW', 'TPA', 'PDX', 'HNL'
]

# Distance ranges between airport pairs (realistic miles)
DISTANCE_RANGES = {
    'short': (200, 600),
    'medium': (600, 1500),
    'long': (1500, 3000),
    'ultra': (3000, 5000)
}


def generate_flight_data(n: int = NUM_RECORDS) -> pd.DataFrame:
    """Generate n rows of realistic synthetic flight data."""

    # Base columns
    years = np.random.choice([2023, 2024], size=n, p=[0.4, 0.6])
    months = np.random.randint(1, 13, size=n)
    days = np.random.randint(1, 29, size=n)  # max 28 to avoid month issues
    day_of_week = np.random.randint(1, 8, size=n)  # 1=Mon, 7=Sun

    # Carrier selection (weighted by market share)
    carrier_codes = list(CARRIERS.keys())
    carrier_weights = [0.15, 0.18, 0.14, 0.20, 0.07, 0.05, 0.06, 0.05, 0.04, 0.06]
    carriers = np.random.choice(carrier_codes, size=n, p=carrier_weights)

    # Origin and Destination (ensure different)
    origins = np.random.choice(AIRPORTS, size=n)
    destinations = np.random.choice(AIRPORTS, size=n)
    # Fix same origin/dest
    same_mask = origins == destinations
    while same_mask.any():
        destinations[same_mask] = np.random.choice(AIRPORTS, size=same_mask.sum())
        same_mask = origins == destinations

    # Scheduled departure time (HHMM format, realistic distribution)
    # More flights during 6AM-10PM
    hours = np.random.choice(
        range(24), size=n,
        p=[0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.07, 0.08, 0.08, 0.07,
           0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.04,
           0.03, 0.02, 0.01, 0.01]
    )
    minutes = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], size=n)
    crs_dep_time = hours * 100 + minutes

    # Distance
    dist_category = np.random.choice(
        ['short', 'medium', 'long', 'ultra'], size=n,
        p=[0.30, 0.40, 0.25, 0.05]
    )
    distances = np.zeros(n, dtype=int)
    for cat, (lo, hi) in DISTANCE_RANGES.items():
        mask = dist_category == cat
        distances[mask] = np.random.randint(lo, hi, size=mask.sum())

    # --- Delay generation (realistic patterns) ---
    # Base delay probability affected by: carrier, time of day, month, day of week
    base_delay_prob = np.full(n, 0.20)  # 20% base delay rate

    # Carrier effects
    carrier_delay_modifier = {
        'AA': 0.02, 'DL': -0.03, 'UA': 0.01, 'WN': 0.03, 'B6': 0.04,
        'AS': -0.02, 'NK': 0.06, 'F9': 0.05, 'G4': 0.07, 'HA': -0.04
    }
    for code, mod in carrier_delay_modifier.items():
        base_delay_prob[carriers == code] += mod

    # Time of day effect (afternoon/evening more delays)
    afternoon_mask = (hours >= 14) & (hours <= 20)
    base_delay_prob[afternoon_mask] += 0.08
    morning_mask = hours < 9
    base_delay_prob[morning_mask] -= 0.05

    # Summer/holiday months more delays
    summer_mask = np.isin(months, [6, 7, 8, 12])
    base_delay_prob[summer_mask] += 0.05

    # Weekend effect
    weekend_mask = day_of_week >= 6
    base_delay_prob[weekend_mask] += 0.02

    # Clip probabilities
    base_delay_prob = np.clip(base_delay_prob, 0.05, 0.50)

    # Determine if delayed
    is_delayed = np.random.random(n) < base_delay_prob

    # Generate departure delay (minutes)
    dep_delay = np.zeros(n)
    dep_delay[is_delayed] = np.random.exponential(25, size=is_delayed.sum()) + 15
    dep_delay[~is_delayed] = np.random.normal(-2, 5, size=(~is_delayed).sum())
    dep_delay = np.round(dep_delay).astype(int)

    # Arrival delay (correlated with departure delay)
    arr_delay = dep_delay + np.random.normal(0, 8, size=n)
    arr_delay = np.round(arr_delay).astype(int)

    # Delay breakdown (only for delayed flights)
    carrier_delay = np.zeros(n)
    weather_delay = np.zeros(n)
    nas_delay = np.zeros(n)
    security_delay = np.zeros(n)
    late_aircraft_delay = np.zeros(n)

    delayed_idx = np.where(arr_delay > 15)[0]
    if len(delayed_idx) > 0:
        total_delay = arr_delay[delayed_idx].astype(float)
        # Distribute delay among causes
        proportions = np.random.dirichlet([3, 2, 2, 0.5, 2.5], size=len(delayed_idx))
        carrier_delay[delayed_idx] = np.round(total_delay * proportions[:, 0]).astype(int)
        weather_delay[delayed_idx] = np.round(total_delay * proportions[:, 1]).astype(int)
        nas_delay[delayed_idx] = np.round(total_delay * proportions[:, 2]).astype(int)
        security_delay[delayed_idx] = np.round(total_delay * proportions[:, 3]).astype(int)
        late_aircraft_delay[delayed_idx] = np.round(total_delay * proportions[:, 4]).astype(int)

    # Cancellations (rare ~2%)
    cancelled = (np.random.random(n) < 0.02).astype(int)
    diverted = (np.random.random(n) < 0.003).astype(int)

    # Flight number
    flight_nums = np.random.randint(100, 9999, size=n)

    # Taxi out/in times
    taxi_out = np.random.normal(16, 6, size=n).clip(3, 60).astype(int)
    taxi_in = np.random.normal(8, 4, size=n).clip(2, 30).astype(int)

    # Air time (based on distance)
    air_time = (distances / 8 + np.random.normal(0, 10, size=n)).clip(20, 700).astype(int)

    # Elapsed time
    actual_elapsed = (taxi_out + air_time + taxi_in).astype(int)
    crs_elapsed = (actual_elapsed + np.random.normal(0, 5, size=n)).clip(30, 800).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        'Year': years,
        'Quarter': ((months - 1) // 3 + 1),
        'Month': months,
        'DayofMonth': days,
        'DayOfWeek': day_of_week,
        'FlightDate': pd.to_datetime(
            [f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)],
            errors='coerce'
        ),
        'Reporting_Airline': carriers,
        'Flight_Number_Reporting_Airline': flight_nums,
        'Origin': origins,
        'Dest': destinations,
        'CRSDepTime': crs_dep_time,
        'DepDelay': dep_delay,
        'DepDelayMinutes': np.maximum(dep_delay, 0),
        'TaxiOut': taxi_out,
        'TaxiIn': taxi_in,
        'ArrDelay': arr_delay,
        'ArrDelayMinutes': np.maximum(arr_delay, 0),
        'Cancelled': cancelled,
        'Diverted': diverted,
        'CRSElapsedTime': crs_elapsed,
        'ActualElapsedTime': actual_elapsed,
        'AirTime': air_time,
        'Distance': distances,
        'CarrierDelay': carrier_delay.astype(int),
        'WeatherDelay': weather_delay.astype(int),
        'NASDelay': nas_delay.astype(int),
        'SecurityDelay': security_delay.astype(int),
        'LateAircraftDelay': late_aircraft_delay.astype(int),
    })

    # Drop rows with invalid dates
    df = df.dropna(subset=['FlightDate'])

    return df


def main():
    print("🛫 Generating synthetic BTS On-Time Performance data...")
    df = generate_flight_data(NUM_RECORDS)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Generated {len(df)} flight records → {OUTPUT_PATH}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Delay rate: {(df['ArrDelay'] > 15).mean():.1%}")
    print(f"   Carriers: {df['Reporting_Airline'].nunique()}")
    print(f"   Airports: {df['Origin'].nunique()} origins, {df['Dest'].nunique()} destinations")
    print(f"   Date range: {df['FlightDate'].min()} to {df['FlightDate'].max()}")


if __name__ == "__main__":
    main()

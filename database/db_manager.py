"""
DuckDB Database Manager — Store and query curated flight data.
"""

import duckdb
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_PATH = os.path.join(DB_DIR, "flight_db.duckdb")
PROCESSED_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "flights_processed.csv")


class DuckDBManager:
    """Manage flight data in DuckDB."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        logger.info(f"🦆 Connected to DuckDB: {self.db_path}")

    def create_table(self):
        """Create flights table if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS flights (
                Year INTEGER,
                FlightDate VARCHAR,
                Reporting_Airline VARCHAR,
                Flight_Number_Reporting_Airline INTEGER,
                Origin VARCHAR,
                Dest VARCHAR,
                Month INTEGER,
                DayofMonth INTEGER,
                DayOfWeek INTEGER,
                dep_hour INTEGER,
                dep_minute INTEGER,
                carrier_encoded INTEGER,
                origin_freq DOUBLE,
                dest_freq DOUBLE,
                Distance INTEGER,
                log_distance DOUBLE,
                is_weekend INTEGER,
                dep_delay_flag INTEGER,
                DepDelay DOUBLE,
                TaxiOut INTEGER,
                CRSElapsedTime INTEGER,
                ArrDelay DOUBLE,
                is_delayed INTEGER
            )
        """)
        logger.info("✅ Table 'flights' ready")

    def load_data(self, csv_path: str = None):
        """Load processed CSV data into DuckDB."""
        csv_path = csv_path or PROCESSED_DATA

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Processed data not found: {csv_path}")

        # Drop and recreate for fresh load
        self.conn.execute("DROP TABLE IF EXISTS flights")

        self.conn.execute(f"""
            CREATE TABLE flights AS
            SELECT * FROM read_csv_auto('{csv_path.replace(os.sep, "/")}')
        """)

        count = self.conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
        logger.info(f"✅ Loaded {count} rows into DuckDB")
        return count

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def get_stats(self) -> dict:
        """Get summary statistics from the flights table."""
        stats = {}
        stats['total_flights'] = self.conn.execute(
            "SELECT COUNT(*) FROM flights"
        ).fetchone()[0]
        stats['delay_rate'] = self.conn.execute(
            "SELECT AVG(is_delayed) FROM flights"
        ).fetchone()[0]
        stats['avg_delay'] = self.conn.execute(
            "SELECT AVG(ArrDelay) FROM flights WHERE is_delayed = 1"
        ).fetchone()[0]
        stats['carriers'] = self.conn.execute(
            "SELECT COUNT(DISTINCT Reporting_Airline) FROM flights"
        ).fetchone()[0]
        stats['airports'] = self.conn.execute(
            "SELECT COUNT(DISTINCT Origin) FROM flights"
        ).fetchone()[0]
        return stats

    def get_training_data(self) -> pd.DataFrame:
        """Get feature matrix for ML training."""
        return self.query("SELECT * FROM flights")

    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()
        logger.info("🦆 DuckDB connection closed")


if __name__ == "__main__":
    db = DuckDBManager()
    db.load_data()
    stats = db.get_stats()
    print("\n📊 Database Statistics:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    db.close()

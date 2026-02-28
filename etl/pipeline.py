"""
ETL Pipeline Orchestrator — Run full Extract → Transform → Load pipeline.
"""

import os
import sys
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from etl.extract import extract
from etl.transform import transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "flights_processed.csv")


def run_pipeline(input_path: str = None, output_path: str = None) -> str:
    """
    Run the full ETL pipeline.

    Args:
        input_path: Optional path to raw data file.
        output_path: Optional path for processed output.

    Returns:
        Path to the processed output file.
    """
    if output_path is None:
        output_path = OUTPUT_FILE

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("🚀 STARTING ETL PIPELINE")
    logger.info("=" * 60)

    # Step 1: Extract
    logger.info("\n📥 STEP 1: EXTRACT")
    raw_df = extract(input_path)
    logger.info(f"   Raw data shape: {raw_df.shape}")

    # Step 2: Transform
    logger.info("\n⚙️ STEP 2: TRANSFORM")
    processed_df = transform(raw_df)
    logger.info(f"   Processed data shape: {processed_df.shape}")

    # Step 3: Load (save to CSV)
    logger.info("\n💾 STEP 3: LOAD")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    logger.info(f"   Saved to: {output_path}")

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ ETL PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"   Input:  {raw_df.shape[0]} rows")
    logger.info(f"   Output: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
    logger.info(f"   Delay rate: {processed_df['is_delayed'].mean():.1%}")
    logger.info("=" * 60)

    return output_path


if __name__ == "__main__":
    run_pipeline()

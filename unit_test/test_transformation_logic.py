
import pandas as pd
import sys
import logging
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_transformation")

# Add parent to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.ingestor import DataIngestor
from src.transformer import DataTransformer
from src.config import ENV_VARS
from dotenv import load_dotenv, find_dotenv

def inspect_transformation():
    """
    Runs ingestion and transformation phases, then inspects the DataFrame
    BEFORE it gets written to JSONL.
    """
    print("Loading environment...")
    load_dotenv(find_dotenv(), override=True)
    
    # Check env
    conn_str = "" # Need logic to get from env safely if not loaded or mocked
    import os
    conn_str = os.getenv("AZURE_STORAGE_CONN_STR")
    container = os.getenv("AZURE_STORAGE_CONTAINER_WNLD")
    
    if not conn_str or not container:
        print("ERROR: Environment variables missing. Ensure .env is set.")
        return

    # 1. Ingest
    ingestor = DataIngestor(conn_str, container)
    try:
        latest, _ = ingestor.find_latest_csv_blob()
        # Ensure download exists (might be cached)
        local = base_dir / "downloads" / latest
        if not local.exists():
            print(f"Downloading {latest}...")
            local = ingestor.download_blob(latest)
    except Exception as e:
        print(f"Ingestion setup failed: {e}")
        return

    print(f"Reading {local.name}...")
    df = ingestor.read_csv(local)
    
    # 2. Transform
    print("Running Transformer...")
    transformer = DataTransformer()
    clean_df = transformer.run_pipeline(df)
    
    print("-" * 50)
    print("INSPECTION OF TRANSFORMED DATAFRAME")
    print(f"Shape: {clean_df.shape}")
    print("-" * 50)
    print(clean_df.sample(5))
    print("-" * 50)
    
    # Inspect specific list columns
    list_cols = ['po_numbers', 'booking_numbers', 'fcr_numbers', 'obl_nos']
    
    # Find a row where these have data
    sample_mask = clean_df['po_numbers'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    if sample_mask.any():
        sample_row = clean_df[sample_mask].iloc[0]
        print("\n[Sample Row with PO Numbers]")
        print(f"Row Index: {sample_row.name}")
        for c in list_cols:
            val = sample_row.get(c)
            print(f"{c} ({type(val).__name__}): {val}")
    else:
        print("\nNo rows found with populated PO numbers for inspection.")
        
    print("-" * 50)
    print("Column Types:")
    print(clean_df[list_cols].dtypes)

if __name__ == "__main__":
    inspect_transformation()

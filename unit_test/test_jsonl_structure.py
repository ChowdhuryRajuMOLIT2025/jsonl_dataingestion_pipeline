
import pandas as pd
import json
import io
import sys
import logging
from pathlib import Path

# Fix paths
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.ingestor import DataIngestor
from src.transformer import DataTransformer
from src.writer import JsonlWriter
from src.config import ENV_VARS
from dotenv import load_dotenv, find_dotenv

def inspect_structure():
    print("Loading environment...")
    load_dotenv(find_dotenv(), override=True)
    
    # 1. Ingest (Mock or Real)
    # We'll use the real one just to get one row
    conn_str = os.getenv("AZURE_STORAGE_CONN_STR")
    container = os.getenv("AZURE_STORAGE_CONTAINER_WNLD")
    
    if not conn_str:
        print("Skipping ingestion (no env), creating mock DataFrame row.")
        # MOCK ROW if env missing
        data = {
            "carr_eqp_uid": "MOCK_UID_123",
            "consignee_codes": ["C1", "C2"],
            "job_no": "JOB001",
            "container_number": "CONT123",
            "cargo_ready_date": "2025-01-01",
            "combined_content": "MOCK CONTENT BLOB",
            "source_group": "shipment_mock"
        }
        df = pd.DataFrame([data])
    else:
        # Use real ingestion (capped) if available to verify real data flow
        try:
             ingestor = DataIngestor(conn_str, container)
             latest, _ = ingestor.find_latest_csv_blob()
             local = base_dir / "downloads" / latest
             if not local.exists():
                 local = ingestor.download_blob(latest)
             # Manual read to speed up test
             df = pd.read_csv(local, dtype=str, keep_default_na=False, nrows=50, encoding="utf-8-sig")
             # Transform
             transformer = DataTransformer()
             df = transformer.run_pipeline(df)
        except Exception as e:
            print(f"Real data loading failed: {e}. Using mock.")
            # Mock fallback
            data = {
                "carr_eqp_uid": "MOCK_UID_123",
                "consignee_codes": ["C1", "C2"],
                "job_no": "JOB001",
                "container_number": "CONT123",
                "combined_content": "MOCK CONTENT",
                "source_group": "shipment_mock"
            }
            df = pd.DataFrame([data])

    print("-" * 50)
    print("Generating Sample JSONL Record...")
    
    # Simulate Writer Logic
    writer = JsonlWriter()
    record = df.iloc[0].to_dict()
    
    # 1. Sanitize
    clean_flat = writer._sanitize_record(record)
    
    # 2. Restructure (Copying logic from writer.py for verification)
    final_record = {
        "document_id": clean_flat.get("carr_eqp_uid"),
        "content": clean_flat.get("combined_content"),
        "consignee_code": clean_flat.get("consignee_codes"),
        "metadata": clean_flat
    }
    
    print(json.dumps(final_record, indent=2, ensure_ascii=False))
    
    print("-" * 50)
    print("Content Field Preview:\n")
    print(final_record["content"])

import os
if __name__ == "__main__":
    inspect_structure()

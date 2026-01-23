import os
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path to allow importing src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import COLUMN_MAPPING


def inspect_csv():
    """
    Reads the latest downloaded CSV and performs a quick sample check
    against the defined schema.
    """
    # Use path relative to this script file, not CWD
    base_dir = Path(__file__).resolve().parent.parent
    download_dir = base_dir / "downloads"

    if not download_dir.exists():
        print(f"Directory not found: {download_dir.resolve()}")
        # Check CWD for debugging
        print(f"Current Working Directory: {Path.cwd()}")
        return

    # Find latest csv
    csv_files = list(download_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in downloads/.")
        return

    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"Inspecting file: {latest_csv.name}")
    print("-" * 50)

    try:
        # Read a sample (first 100 rows) with strict string typing
        df = pd.read_csv(
            latest_csv,
            dtype=str,
            keep_default_na=False,
            nrows=100,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        print("UTF-8 strict failed, trying iso-8859-1...")
        df = pd.read_csv(
            latest_csv,
            dtype=str,
            keep_default_na=False,
            nrows=100,
            encoding="iso-8859-1",
        )

    print(f"Loaded {len(df)} sample rows.")

    # 1. Header Check
    source_cols = set(df.columns)
    mapped_source_cols = set(COLUMN_MAPPING.keys())

    # Cleaning headers as the pipeline does
    clean_headers = (
        pd.Index(df.columns)
        .astype(str)
        .str.replace(r"[\n\r\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df.columns = clean_headers
    source_cols = set(clean_headers)

    print(f"\nTotal Columns in CSV: {len(source_cols)}")

    missing_required = mapped_source_cols - source_cols
    # Filtering out columns that might be renamed or calculated?
    # Actually COLUMN_MAPPING keys ARE the expected source headers.

    if missing_required:
        print(f"\n[WARNING] The following mapped columns are MISSING in the CSV:")
        for c in missing_required:
            print(f" - {c}")
    else:
        print("\n[SUCCESS] All mapped columns found in CSV headers.")

    # 2. Sample Data
    print("\nSample Record (First Row):")
    first_row = df.iloc[0].to_dict()
    for k, v in first_row.items():
        # Only print keys that are in our mapping to reduce noise
        if k in mapped_source_cols:
            print(f"  {k} -> {v}")

    print("-" * 50)
    print("CSV Inspection Completed.")


if __name__ == "__main__":
    inspect_csv()

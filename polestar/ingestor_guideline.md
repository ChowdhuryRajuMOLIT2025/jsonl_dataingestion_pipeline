# ingestor.py Guideline

## Objective
`src/ingestor.py` is the ingestion layer of this pipeline. It is responsible for:
- connecting to Azure Blob Storage,
- identifying the latest CSV blob in the download container,
- downloading that CSV to local `downloads/`,
- loading the CSV into a pandas DataFrame with safe string handling.

This file does not transform or write JSONL. It only acquires and reads source data.

## Step-by-Step Execution
1. Module import phase:
- Imports logging, pathlib, datetime typing helpers, pandas, and Azure Blob SDK classes.
- Creates module logger: `logger = logging.getLogger("shipment_ingestion")`.

2. `DataIngestor.__init__(conn_str, container_name, download_dir="downloads")`:
- Stores connection string and container name.
- Creates local download directory if missing (`mkdir(parents=True, exist_ok=True)`).
- Initializes `BlobServiceClient` from connection string.
- Initializes container client for the given container.
- On any setup failure, logs with traceback and re-raises.

3. `find_latest_csv_blob(prefix=None) -> Tuple[str, datetime]`:
- Lists blobs in container, optionally filtered by prefix.
- Keeps only blobs ending in `.csv` (case-insensitive).
- Compares `last_modified` and picks the newest CSV.
- Returns `(latest_blob_name, latest_last_modified)`.
- Raises `FileNotFoundError` if no CSV is found.
- Logs scan and selected file details.

4. `download_blob(blob_name, overwrite=False) -> Path`:
- Computes local path as `downloads/<blob_filename_only>`.
- If the local file already exists and is non-empty, it skips the blob download (unless `overwrite=True`).
- Creates blob client for `blob_name`.
- Streams blob to local file in binary mode.
- Returns local file path after successful download.
- Logs start/end and re-raises on failure.

5. `read_csv(csv_path) -> pd.DataFrame`:
- Reads CSV with strict settings:
  - `dtype=str` to prevent ID corruption (leading zero loss),
  - `keep_default_na=False`,
  - `low_memory=False`,
  - primary encoding `utf-8-sig`.
- On `UnicodeDecodeError`, retries with `iso-8859-1`.
- Logs loaded DataFrame shape and re-raises if both reads fail.

## Runtime Usage in Pipeline
`src/pipeline.py` uses this sequence:
1. Instantiate `DataIngestor`.
2. Call `find_latest_csv_blob()`.
3. Call `download_blob(latest_csv_name)`.
4. Call `read_csv(local_csv_path)`.
5. Pass DataFrame to transformer stage.

Note: If a local CSV already exists in `downloads/`, the pipeline can skip Azure blob ingestion entirely and just read the local CSV.

## Operational Notes
1. The ingestor currently selects latest file by `last_modified`, not by filename date.
2. CSV filtering is extension-based; non-CSV blobs are ignored.
3. Download path uses blob basename only; nested blob folder structure is flattened locally.
4. Imported Azure exceptions are currently not explicitly handled per type in methods.

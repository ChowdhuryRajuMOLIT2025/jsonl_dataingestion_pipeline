# pipeline.py Guideline

## Objective
`src/pipeline.py` is the orchestrator for end-to-end execution. It coordinates:
- environment loading,
- ingestion from Azure Blob,
- transformation,
- parquet snapshot creation,
- JSONL generation,
- upload of generated JSONL files.

This file controls workflow order, runtime logging, and failure behavior.

## Step-by-Step Execution
1. Module import phase:
- Imports standard libs (`logging`, `os`, `sys`, `time`, `datetime`, `Path`).
- Imports dotenv helpers (`find_dotenv`, `load_dotenv`).
- Imports pipeline components:
  - `ENV_VARS`, `MAX_FILE_SIZE_BYTES` from `config.py`
  - `DataIngestor`
  - `DataTransformer`
  - `JsonlWriter`, `JsonlWriterConfig`

2. Logging setup (`setup_logging`):
- Initializes logger `shipment_ingestion` at `INFO`.
- Adds console handler once.
- Adds file handler at `logs/pipeline.log` once.
- Returns configured logger.

3. Pipeline init:
- `ShipmentDataPipeline.__init__` creates an empty config dict.

4. Configuration load (`load_configuration`):
- Loads `.env` using dotenv.
- Validates required env keys listed in `ENV_VARS`.
- Stores values in `self.config`.
- Raises `EnvironmentError` if any required key is missing.

5. Runtime entry (`run`):
- Starts timer and logs pipeline start.
- Calls `load_configuration`.

6. Step 1: Ingestion
- Creates `DataIngestor` using:
  - `AZURE_STORAGE_CONN_STR`
  - `AZURE_STORAGE_CONTAINER_WNLD`
- Finds latest CSV blob and downloads it.
- Reads CSV into DataFrame.
- Logs elapsed time for ingestion.

7. Step 2: Transformation
- Instantiates `DataTransformer`.
- Calls `transformer.run_pipeline(df)`.
- Logs elapsed time.

8. Step 2.5: Parquet snapshot
- Writes transformed full dataset to `output_parquet/master_ds.parquet`.
- If file exists, moves old file to `output_parquet/backup/master_ds_<ddmmmyy>.parquet`.
- Logs snapshot path.

9. Step 3: JSONL writing
- Creates `JsonlWriter` with `output_dir=\"output_jsonl\"`.
- Groups transformed data by `source_group`.
- Converts each group to records and writes one JSONL file per group.
- Logs number of generated files and elapsed time.

10. Step 4: Upload
- Determines upload target from `AZURE_STORAGE_CONTAINER_UPLD`.
- Falls back to `shipment-csv-data` if missing.
- Calls `writer.upload_files(...)`.
- Logs elapsed time and total duration.

11. Failure behavior
- Any uncaught exception logs traceback and exits process with `sys.exit(1)`.

12. Script mode
- If run as main module, creates `ShipmentDataPipeline` and executes `run()`.

## Runtime Relationship
- Upstream: `.env` + Azure Blob source CSV.
- Midstream: `DataIngestor` -> `DataTransformer` -> `JsonlWriter`.
- Downstream: Uploaded JSONL blobs in target container.

## Current Known Caveats
1. `MAX_FILE_SIZE_BYTES` is imported but not enforced in `pipeline.py`.
2. Grouping by `source_group` can skip rows where `source_group` is null.
3. Backup naming uses only date (not timestamp), so multiple runs in one day can overwrite backup file name.

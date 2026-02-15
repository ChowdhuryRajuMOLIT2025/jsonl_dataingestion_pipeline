# config.py Guideline

## Objective
`src/config.py` is the central contract file for this pipeline. It defines:
- required environment variables,
- hard constraints (like max file size),
- strict CSV-to-canonical column mapping,
- transformation control lists used by the transformer.

It does not execute pipeline logic by itself. It only publishes constants that other modules consume.

## Step-by-Step Execution (Import Time)
1. Import dependencies:
- `os` (currently unused here)
- typing helpers (`Dict`, `List`) for annotations.

2. Define required Azure environment variable names in `ENV_VARS`:
- `AZURE_STORAGE_ACCOUNT_URL`
- `AZURE_STORAGE_CONTAINER_UPLD`
- `AZURE_STORAGE_CONTAINER_WNLD`
- `AZURE_STORAGE_CONN_STR`

3. Define optional env variable names in `OPTIONAL_ENV_VARS`:
- `AZURE_BLOB_NAME`
- `AZURE_BLOB`

4. Define hard size cap in bytes:
- `MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024`
- This represents the strict 12 MB file limit.

5. Define `COLUMN_MAPPING` (source CSV header -> canonical internal field):
- This is the schema contract used during transformation.
- Headers not present here are treated as extra/unexpected.

6. Build `DATE_COLUMNS` from `COLUMN_MAPPING` values:
- Auto-select canonical fields ending with `_date`
- Also include `in-dc_date` explicitly.

7. Define special handling lists/sets:
- `MULTI_VALUED_DATE_COLS`
- `MULTI_VALUED_ID_COLS`
- `BOOLEAN_COLUMNS`
- `ID_COLUMNS`

8. Define cleanup drop list in `DROP_COLUMNS_AFTER_LOAD`:
- Technical/redundant columns to remove after enrichment.

## How Other Files Use This
- `src/pipeline.py`
  - reads `ENV_VARS` to validate required configuration.
  - imports config contracts for orchestration.

- `src/writer.py`
  - enforces `MAX_FILE_SIZE_BYTES` via `JsonlWriterConfig.max_file_size_bytes` (default hard cap).

- `src/transformer.py`
  - uses `COLUMN_MAPPING` for schema normalization and validation.
  - uses `DATE_COLUMNS`, `MULTI_VALUED_DATE_COLS`, `MULTI_VALUED_ID_COLS`, `BOOLEAN_COLUMNS`, `ID_COLUMNS` during cleaning/parsing.
  - uses `DROP_COLUMNS_AFTER_LOAD` before final output.

## Practical Maintenance Rules
1. Treat `COLUMN_MAPPING` as a versioned contract.
2. Add new source columns only with clear downstream usage.
3. If canonical names change, update transformer and downstream index mapping together.
4. Keep date fields consistently suffixed with `_date` when possible.
5. Keep `MAX_FILE_SIZE_BYTES` enforced in runtime writer logic; if changed, review downstream index/upload constraints.
6. Validate no duplicate semantic targets unless intentionally aliased.

## Current Known Caveat
- `COLUMN_MAPPING` currently maps both `Get_ISF_submission_dt` and `get_isf_submission_date` to the same canonical key `get_isf_submission_date`.
- This aliasing can be valid, but strict position-based schema validation must handle it carefully.

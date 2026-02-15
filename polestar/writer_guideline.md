# writer.py Guideline

## Objective
`src/writer.py` converts transformed rows into JSONL documents and uploads generated files to Azure Blob Storage.

Core responsibilities:
- normalize each input document into one stable schema,
- enforce strict file-size sharding (12MB cap by default),
- write UTF-8 JSONL safely,
- upload produced files.

## Step-by-Step Execution
1. Module setup:
- Imports JSON/logging/path utilities and Azure Blob SDK.
- Imports `MAX_FILE_SIZE_BYTES` from config.
- Initializes logger `shipment_ingestion`.

2. Helper utilities:
- `_mmmyy_from_dt(dt)`: default lower-case month tag.
- `_safe_tag(tag)`: sanitizes tag for filenames.
- `_next_counter(output_dir, tag)`: resolves next numeric suffix.
- `_coerce_consignee_codes(val)`: normalizes to `list[str]`.
- `_ensure_dict_metadata(m)`: guarantees metadata dict.
- `_sanitize_for_json(val)`: recursively converts NaN/NaT/pd.NA to `None`, datetime-like to ISO text.
- `_safe_text_or_empty(val)`: prevents `"nan"`-like text leakage into output fields.

3. Writer config:
- `JsonlWriterConfig`:
  - `output_dir`
  - `strict`
  - `max_file_size_bytes` (defaults to config 12MB hard limit)

4. `JsonlWriter.write(docs, mmmyy)`:
- Resolves output directory and filename tag.
- Enforces positive `max_file_size_bytes`.
- Iterates normalized docs and writes line-by-line.
- Rolls to next file when current file would exceed size cap.
- Raises hard error if a single record alone exceeds max size.
- Tracks all generated shard paths in `generated_files`.
- Returns the first file path created for that call.

5. Document normalization (`_normalize_doc`):
- Rejects non-dict input.
- Supports two inputs:
  - Case A: already `id/content/metadata` style.
  - Case B: raw transformed row.
- Both cases now emit same schema:
  - `document_id` (string)
  - `content` (string)
  - `consignee_code` (`list[str]`)
  - `metadata` (`dict`)
- `metadata["consignee_codes"]` is normalized to `list[str]`.

6. Validation (`_validate`):
- Requires keys: `document_id`, `content`, `metadata`.
- Ensures metadata is dict.
- In `strict=True`, enforces non-empty `document_id` and `content`.

7. Upload (`upload_files`):
- Connects by connection string.
- Creates target container if missing.
- Uploads each generated file with overwrite enabled.

8. Convenience API (`write_jsonl`):
- Creates writer instance and returns path string from `write(...)`.

## Runtime Relationship
- Upstream input: transformed row dicts from `DataTransformer`.
- Downstream output:
  - local shard files `shipment_<tag>_<n>.jsonl`
  - uploaded blobs in target container.

## Current Known Caveats
1. `write(...)` returns only the first path from a potentially multi-file shard write; full list is in `writer.generated_files`.
2. `strict=False` remains default, so empty content can still pass unless strict mode is enabled.

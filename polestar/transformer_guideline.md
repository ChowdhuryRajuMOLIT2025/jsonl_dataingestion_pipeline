# transformer.py Guideline

## Objective
`src/transformer.py` is the core transformation engine of the pipeline. It converts raw CSV rows into a canonical, enriched shipment dataset that is ready for JSONL generation and downstream indexing.

Primary outcomes:
- strict schema enforcement,
- type normalization,
- shipment-level derived fields,
- search-ready `combined_content`,
- partition tags for file grouping.

## Step-by-Step Execution
1. Module import phase:
- Imports pandas/numpy, regex, typing, logging.
- Imports config contracts:
  - `COLUMN_MAPPING`
  - `DATE_COLUMNS`
  - `MULTI_VALUED_DATE_COLS`
  - `MULTI_VALUED_ID_COLS`
  - `BOOLEAN_COLUMNS`
  - `ID_COLUMNS`
  - `DROP_COLUMNS_AFTER_LOAD`

2. Initialize transformer:
- `DataTransformer.__init__(logger_=None)` sets runtime logger.

3. Main entrypoint:
- `run_pipeline(raw_df)` executes transformation in strict sequence:
  1. `_normalize_and_validate_headers`
  2. `_rename_to_canonical`
  3. `_basic_clean`
  4. `_enforce_flp_date_consistency`
  5. `_derive_consignee_fields`
  6. `_derive_optimal_dates`
  7. `_derive_delay_flags`
  8. `_derive_shipment_status`
  9. `_derive_critical_dates_summary`
  10. `_derive_delay_reason_and_gaps`
  11. `_derive_milestones`
  12. `_derive_vessel_and_carrier`
  13. `_derive_port_route_summary`
  14. `add_partition_tags`
  15. `generate_search_content`
  16. Optional drop of technical columns via `DROP_COLUMNS_AFTER_LOAD`

4. Header normalization and schema enforcement:
- `_normalize_and_validate_headers(df)` performs:
  - whitespace normalization of incoming headers,
  - normalized-key matching across raw and canonical naming,
  - missing/extra detection,
  - canonical rename map construction,
  - strict position validation against expected canonical order.
- Raises `ValueError` if schema checks fail.

5. Basic cleanup phase:
- `_basic_clean(df)` runs:
  - `_normalize_object_strings`: trims object/string fields and cleans trivial noise.
  - `_ensure_str_ids`: forces ID columns and multi-ID columns to string.
  - `_coerce_booleans`: maps Y/N/TRUE/FALSE/1/0 into nullable boolean dtype.
  - `_parse_dates`: parses date columns into normalized `Timestamp` (or list of dates for multi-valued date columns).

6. Safety helper utilities (row-level robustness):
- `_first_scalar`, `_safe_text`, `_to_date_or_none`
- These prevent failures when `row.get(...)` returns `Series` due to duplicate column labels and normalize null-handling.

7. FLP consistency enforcement:
- `_enforce_flp_date_consistency(df)` applies business rule:
  - if `final_load_port` is empty/null, clear related FLP date values.
- Target FLP date fields:
  - `etd_flp_date`
  - `eta_flp_date` (if present)
  - `ata_flp_date`
  - `ara_flp_date` (if present)
  - `atd_flp_date`

8. Derivation phase (business logic):
- `_derive_consignee_fields`:
  - extracts `consignee_name` and `consignee_codes` from `consignee_raw`-like field.
- `_derive_optimal_dates`:
  - calculates `optimal_ata_dp_date` and `optimal_eta_fd_date`.
- `_derive_delay_flags`:
  - generates `delayed_dp`, `dp_delayed_dur`, `delayed_fd`, `fd_delayed_dur`.
- `_derive_shipment_status`:
  - computes shipment lifecycle status.
- `_derive_critical_dates_summary`:
  - builds concise date summary text.
- `_derive_delay_reason_and_gaps`:
  - creates `delay_reason_summary` and `workflow_gap_flags`.
- `_derive_milestones`:
  - composes timeline-like milestone text.
- `_derive_vessel_and_carrier`:
  - creates `vessel_summary` and `carrier_summary`.
- `_derive_port_route_summary`:
  - builds route narrative from origin/transshipment/discharge/delivery fields.

9. Partition tagging:
- `add_partition_tags(df)` derives:
  - `source_group` as `%Y-%m` from `etd_lp_date`,
  - `source_month_tag` as `%b_%y` from `etd_lp_date`.

10. Search content construction:
- `generate_search_content(df)` creates `combined_content` using `_build_combined_content`.
- `_build_combined_content` consolidates:
  - shipment identity,
  - route, dates, delay snapshots,
  - milestone and reasoning summaries,
  - operational events,
  - cargo/party/reference details.

11. Output contract to writer:
- Expected downstream-important fields include:
  - `carr_eqp_uid` (or fallback IDs),
  - `combined_content`,
  - `consignee_codes`,
  - partition tags (`source_group`, `source_month_tag`),
  - derived status/delay fields.

## Runtime Relationship
- Upstream input: DataFrame from `ingestor.py`.
- Downstream consumer: `writer.py` converts transformed rows to JSONL docs.
- Pipeline orchestration: `pipeline.py` calls `transformer.run_pipeline(df)` as step 2.

## Current Known Caveats (Important)
1. `_derive_optimal_ata_dp_date` references `derived` while assignment is commented, which can raise `NameError` on rows without `ata_dp_date`.
2. `_build_combined_content` uses truth check `if hot_container_flag:` which can fail for nullable boolean (`pd.NA`) with ambiguous truth-value error.
3. Date parsing uses `dayfirst=True`; ISO-style dates still parse but can trigger warnings and ambiguity for mixed formats.
4. Header validation enforces position strictly; alias-heavy or duplicate-target mappings in config can cause avoidable schema failures if not carefully managed.

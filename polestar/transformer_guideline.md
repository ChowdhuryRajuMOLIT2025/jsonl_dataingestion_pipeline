# transformer.py Guideline

## Objective
`src/transformer.py` is the core transformation engine. It converts raw CSV rows into canonical shipment records with derived status, delay, narrative fields, and search-ready text (`combined_content`).

## Execution Sequence (`run_pipeline`)
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
16. Optional drop via `DROP_COLUMNS_AFTER_LOAD`

## Key Behaviors (Current)
1. Header validation is strict:
- Validates names and order against canonical mapping.
- Raises `ValueError` on missing/extra or position mismatch.

2. FLP consistency rule:
- If `final_load_port` is blank/null token, clears FLP date columns when present:
  - `etd_flp_date`
  - `eta_flp_date`
  - `ata_flp_date`
  - `atd_flp_date`

3. Best ETA derivation:
- `best_eta_dp_date` currently derives from `derived_ata_dp_date` only.
- `best_eta_fd_date` priority:
  1. `predictive_eta_fd_date`
  2. `revised_eta_fd_date`
  3. `eta_fd_date`

4. DP delay logic (`_derive_dp_delay`):
- If `ata_dp_date` exists and `eta_dp_date` exists: delay is `ata_dp_date - eta_dp_date`.
- If ATA missing and both ETA DP and Best ETA DP are in past: delay is `best_eta_dp_date - eta_dp_date`.
- If ATA missing and ETA DP is past but Best ETA DP is future: delay is `today - eta_dp_date`.
- If ETA DP and Best ETA DP are future: `on_time`, `0`.

5. Shipment status logic (high level):
- Completed statuses checked first (`EMPTY_CONTAINER_RETURNED`, `DELIVERED`).
- DP detection supports ATA and fallback DP-end events:
  - `vehicle_arrival_date`
  - `carrier_vehicle_unload_date`
  - `out_gate_from_dp_date`
- Ocean transit detection includes:
  - `carrier_vehicle_load_date`
  - `vehicle_departure_date`
  - `atd_lp_date`
  - `atd_flp_date`

6. Critical timeline enrichment:
- `critical_dates_summary` now includes upstream **Cargo Processing** before Leg 1:
  - `cargo_ready_date`
  - `cargo_receiveds_date`
  - `in-dc_date`
  - `empty_container_dispatch_date` + `empty_container_dispatch_lcn`
  - `in_gate_date` + `in_gate_lcn`
- `milestones` includes matching Cargo Processing stage and updated leg-state narratives.

7. Search text quality:
- `combined_content` uses placeholder filtering to avoid null-like tokens (`NaN`, `None`, etc.) in text.
- `hot_container_flag` is handled safely for nullable booleans.

## Output Contract to Writer
Important fields produced for downstream JSONL:
- identity: `carr_eqp_uid`, `job_no`, `container_number`
- content: `combined_content`, `milestones`
- access/filtering: `consignee_codes`
- ETA/Delay/Status: `best_eta_dp_date`, `best_eta_fd_date`, `delayed_dp`, `dp_delayed_dur`, `delayed_fd`, `fd_delayed_dur`, `shipment_status`
- partitioning: `source_group`, `source_month_tag`

## Current Risks / Caveats
1. Date parsing uses `dayfirst=True`; mixed-format client dates can still parse unexpectedly.
2. Header validation is intentionally strict; any source schema drift hard-fails fast.
3. `best_eta_dp_date` is now driven by `derived_ata_dp_date`; if that upstream field is sparse/invalid, DP delay/status quality degrades.

"""
transformer.py

Production-oriented DataTransformer for shipment ingestion.

Key fixes vs your current file:
- No nested method definitions (everything is at class scope).
- All helper utilities are @staticmethod / @classmethod where appropriate.
- _derive_milestones(df) operates on a DataFrame and creates df["milestones"] safely.
- Robust handling for duplicate column names (row.get can return Series).
- No ambiguous truth-value checks on Series (fixes your ValueError root cause).
- No accidental "df is not defined" in row-level logic.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BOOLEAN_COLUMNS  # columns to coerce to booleans
from .config import COLUMN_MAPPING  # raw header -> canonical name
from .config import DATE_COLUMNS  # list of canonical *_date columns
from .config import \
    DROP_COLUMNS_AFTER_LOAD  # columns safe to drop after enrichment
from .config import ID_COLUMNS  # columns that must be string IDs
from .config import (  # date columns that can contain multiple dates in a cell; ID columns that can have multiple values in a cell
    MULTI_VALUED_DATE_COLS, MULTI_VALUED_ID_COLS)

logger = logging.getLogger("shipment_ingestion")


class DataTransformer:
    """
    End-to-end transformation pipeline for shipment data.

    Responsibilities:
    1. Header normalization + schema enforcement.
    2. Type normalization (strings, booleans, dates).
    3. Derivations:
       - consignee_name / consignee_codes
       - optimal_ata_dp_date / optimal_eta_fd_date
       - delay flags & durations (DP / FD)
       - shipment_status
       - critical_dates_summary
       - delay_reason_summary / workflow_gap_flags
       - milestones
       - vessel_summary / carrier_summary
       - port_route_summary
       - source_group / source_month_tag
       - combined_content (for Azure AI Search)
    """

    def __init__(self, logger_: Optional[logging.Logger] = None) -> None:
        self.logger = logger_ or logger

    # -------------------------------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------------------------------
    def run_pipeline(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting transformation pipeline. Raw shape=%s", raw_df.shape)

        df = self._normalize_and_validate_headers(raw_df)
        self.logger.info("Headers normalized & validated. Shape=%s", df.shape)

        df = self._rename_to_canonical(df)
        self.logger.info("Columns renamed to canonical schema. Shape=%s", df.shape)

        df = self._basic_clean(df)
        self.logger.info(
            "Basic cleaning done (strings, booleans, IDs, dates). Shape=%s", df.shape
        )

        df = self._derive_consignee_fields(df)
        self.logger.info("Consignee name/code derived. Shape=%s", df.shape)

        df = self._derive_optimal_dates(df)
        self.logger.info("Optimal DP / FD dates derived. Shape=%s", df.shape)

        df = self._derive_delay_flags(df)
        self.logger.info("Delay flags & durations (DP/FD) derived. Shape=%s", df.shape)

        df = self._derive_shipment_status(df)
        self.logger.info("Shipment status derived. Shape=%s", df.shape)

        df = self._derive_critical_dates_summary(df)
        self.logger.info("Critical dates summary derived. Shape=%s", df.shape)

        df = self._derive_delay_reason_and_gaps(df)
        self.logger.info("Delay reasons & workflow gaps derived. Shape=%s", df.shape)

        df = self._derive_milestones(df)
        self.logger.info("Milestones derived. Shape=%s", df.shape)

        df = self._derive_vessel_and_carrier(df)
        self.logger.info("Vessel & carrier summaries derived. Shape=%s", df.shape)

        df = self._derive_port_route_summary(df)
        self.logger.info("Port route summary derived. Shape=%s", df.shape)

        df = self.add_partition_tags(df)
        self.logger.info(
            "Partition tags (source_group / source_month_tag) derived. Shape=%s",
            df.shape,
        )

        df = self.generate_search_content(df)
        self.logger.info("combined_content generated. Final shape=%s", df.shape)

        if DROP_COLUMNS_AFTER_LOAD:
            before = df.shape[1]
            df = df.drop(
                columns=[c for c in DROP_COLUMNS_AFTER_LOAD if c in df.columns],
                errors="ignore",
            )
            after = df.shape[1]
            self.logger.info(
                "Dropped %d technical columns (DROP_COLUMNS_AFTER_LOAD). Final cols=%d",
                before - after,
                after,
            )

        self.logger.info("Transformation pipeline completed successfully.")
        return df

    # -------------------------------------------------------------------------
    # Header normalization & schema enforcement
    # -------------------------------------------------------------------------
    def _normalize_and_validate_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_cols = list(df.columns)
        self.logger.info("Raw columns in file: %d", len(raw_cols))

        normalized_cols = (
            pd.Index(raw_cols)
            .astype(str)
            .str.replace(r"[\r\n\t]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        out = df.copy()
        out.columns = normalized_cols

        # expected = list(COLUMN_MAPPING.keys().lower())
        # expected = [k.lower() for k in COLUMN_MAPPING.keys()]
        # expected = list(COLUMN_MAPPING.values())
        # actual = list(out.columns)
        # expected = [k.lower() for k in COLUMN_MAPPING.keys()]
        # actual = [c.lower() for c in out.columns]
        # missing = [c for c in expected if c not in actual]
        # extra = [c for c in actual if c not in expected]

        actual_raw = list(out.columns)

        def _norm_header_key(val: Any) -> str:
            text = str(val)
            text = re.sub(r"[\r\n\t]+", " ", text)
            text = re.sub(r"\s+", " ", text)
            text = text.strip().lower()
            text = re.sub(r"[^a-z0-9]+", "_", text)
            text = re.sub(r"_+", "_", text).strip("_")
            return text

        expected_raw_norm = [_norm_header_key(k) for k in COLUMN_MAPPING.keys()]
        expected_canon_norm = [_norm_header_key(v) for v in COLUMN_MAPPING.values()]
        actual_norm = [_norm_header_key(c) for c in actual_raw]

        schema_mode = None
        expected = expected_raw_norm
        actual = actual_norm

        if set(actual_norm) == set(expected_canon_norm):
            schema_mode = "canonical"
            expected = expected_canon_norm
        elif set(actual_norm) == set(expected_raw_norm):
            schema_mode = "raw_or_snake"
            expected = expected_raw_norm

        # missing = [c for c in expected if c not in actual]
        # extra = [c for c in actual if c not in expected]

        if schema_mode is None:
            missing_raw = [c for c in expected_raw_norm if c not in actual_norm]
            extra_raw = [c for c in actual_norm if c not in expected_raw_norm]
            missing_canon = [c for c in expected_canon_norm if c not in actual_norm]
            extra_canon = [c for c in actual_norm if c not in expected_canon_norm]
            if (len(missing_canon) + len(extra_canon)) < (
                len(missing_raw) + len(extra_raw)
            ):
                schema_mode = "canonical"
                expected = expected_canon_norm
                missing = missing_canon
                extra = extra_canon
            else:
                schema_mode = "raw_or_snake"
                expected = expected_raw_norm
                missing = missing_raw
                extra = extra_raw

        norm_to_canonical: Dict[str, str] = {}
        for raw_name, canonical_name in COLUMN_MAPPING.items():
            for key in (raw_name, canonical_name):
                norm_key = _norm_header_key(key)
                if norm_key not in norm_to_canonical:
                    norm_to_canonical[norm_key] = canonical_name
                elif norm_to_canonical[norm_key] != canonical_name:
                    self.logger.warning(
                        "Duplicate normalized header '%s' maps to both '%s' and '%s'.",
                        norm_key,
                        norm_to_canonical[norm_key],
                        canonical_name,
                    )

        expected_canon: List[str] = []
        _seen_expected: set = set()
        for v in COLUMN_MAPPING.values():
            if v in _seen_expected:
                continue
            expected_canon.append(v)
            _seen_expected.add(v)
        actual_canon_set = set()
        unmapped_cols: List[str] = []
        for col in out.columns:
            norm_col = _norm_header_key(col)
            if norm_col in norm_to_canonical:
                actual_canon_set.add(norm_to_canonical[norm_col])
            else:
                unmapped_cols.append(col)

        missing = [c for c in expected_canon if c not in actual_canon_set]
        extra = [_norm_header_key(c) for c in unmapped_cols]

        self.logger.info("Columns in COLUMN_MAPPING: %d", len(expected_canon))
        self.logger.info("Columns after normalization: %d", len(actual_norm))
        self.logger.info("Header schema mode detected: %s", schema_mode)
        self.logger.info("Missing (expected but not in raw): %d", len(missing))
        self.logger.info("Extra (in raw but not mapped): %d", len(extra))

        if missing:
            self.logger.error("Missing columns: %s", missing)
        if extra:
            self.logger.error("Extra columns: %s", extra)

        if missing or extra:
            raise ValueError(
                "HEADER VALIDATION FAILED: schema mismatch "
                f"(missing={len(missing)}, extra={len(extra)})."
            )

        rename_map: Dict[str, str] = {}
        for col in out.columns:
            norm_col = _norm_header_key(col)
            if norm_col in norm_to_canonical:
                rename_map[col] = norm_to_canonical[norm_col]

        if rename_map:
            out = out.rename(columns=rename_map)

        # Position check
        expected_pos = list(expected_canon)
        actual_pos = list(out.columns)
        mismatches: List[Tuple[int, str, str]] = []
        for i, (act, exp) in enumerate(zip(actual_pos, expected_pos)):
            if act != exp:
                mismatches.append((i, act, exp))

        if mismatches:
            self.logger.error(
                "HEADER VALIDATION FAILED: %d mismatch(es) found.", len(mismatches)
            )
            self.logger.error("First 20 mismatches (index | actual -> expected):")
            for i, act, exp in mismatches[:20]:
                self.logger.error("  [%d] '%s'  ->  '%s'", i, act, exp)
            first_idx = mismatches[0][0]
            start = max(0, first_idx - 3)
            end = min(len(actual), first_idx + 4)
            self.logger.error("Context around first mismatch:")
            for i in range(start, end):
                self.logger.error(
                    "  [%d] actual='%s' | expected='%s'",
                    i,
                    actual_pos[i],
                    expected_pos[i],
                )
            raise ValueError("HEADER VALIDATION FAILED: name/position mismatch.")

        self.logger.info(
            "Header validation PASSED: all names and positions align with schema."
        )
        return out

    def _rename_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=COLUMN_MAPPING).copy()

    # -------------------------------------------------------------------------
    # Basic cleaning
    # -------------------------------------------------------------------------
    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._normalize_object_strings(df)
        out = self._ensure_str_ids(out, ID_COLUMNS + MULTI_VALUED_ID_COLS)
        out = self._coerce_booleans(out, BOOLEAN_COLUMNS)
        out = self._parse_dates(out, DATE_COLUMNS, MULTI_VALUED_DATE_COLS)
        return out

    @staticmethod
    def _normalize_object_strings(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        obj_cols = out.select_dtypes(include="object").columns
        if len(obj_cols) == 0:
            return out

        out[obj_cols] = out[obj_cols].apply(lambda s: s.astype("string").str.strip())
        out[obj_cols] = out[obj_cols].replace("()", "", regex=False)
        out[obj_cols] = out[obj_cols].replace({"nan": pd.NA, "NaN": pd.NA})
        return out

    @staticmethod
    def _ensure_str_ids(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].astype("string").str.replace('"', "", regex=False)
        return out

    @staticmethod
    def _coerce_booleans(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        mapping: Dict[str, Optional[bool]] = {
            "Y": True,
            "YES": True,
            "TRUE": True,
            "1": True,
            "N": False,
            "NO": False,
            "FALSE": False,
            "0": False,
        }
        for c in cols:
            if c not in out.columns:
                continue
            s = out[c].astype("string").str.strip().str.upper()
            out[c] = s.map(mapping).astype("boolean")
        return out

    @staticmethod
    def _parse_dates(
        df: pd.DataFrame, date_cols: List[str], multi_date_cols: List[str]
    ) -> pd.DataFrame:
        out = df.copy()

        def _parse_single_date(val: Any) -> Optional[pd.Timestamp]:
            if val is None:
                return None
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass
            ts = pd.to_datetime(val, dayfirst=True, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.normalize()

        def _parse_multi_dates(val: Any) -> Optional[List[pd.Timestamp]]:
            if val is None:
                return None
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass

            if isinstance(val, (list, tuple)):
                parsed = [
                    d for d in (_parse_single_date(v) for v in val) if d is not None
                ]
                return parsed or None

            text = str(val).strip()
            if not text:
                return None

            parts = [p.strip() for p in text.split(",") if p.strip()]
            parsed = [
                d for d in (_parse_single_date(p) for p in parts) if d is not None
            ]
            return parsed or None

        for c in date_cols:
            if c not in out.columns:
                continue
            if c in multi_date_cols:
                out[c] = out[c].apply(_parse_multi_dates)
            else:
                out[c] = out[c].apply(_parse_single_date)

        return out

    # -------------------------------------------------------------------------
    # Utilities for safe row.get values (handles duplicate column names)
    # -------------------------------------------------------------------------
    @staticmethod
    def _first_scalar(v: Any) -> Any:
        """
        If v is a Series (common when df has duplicate column names), return the first
        meaningful scalar; else return v.
        """
        if isinstance(v, pd.Series):
            for x in v.tolist():
                if x is None:
                    continue
                try:
                    if pd.isna(x):
                        continue
                except Exception:
                    pass
                if isinstance(x, str) and not x.strip():
                    continue
                return x
            return v.iloc[0] if len(v) else None
        return v

    @staticmethod
    def _safe_text(v: Any, default: str = "") -> str:
        v = DataTransformer._first_scalar(v)
        if v is None:
            return default
        try:
            if pd.isna(v):
                return default
        except Exception:
            pass
        s = str(v).strip()
        return s if s else default

    @staticmethod
    def _to_date_or_none(v: Any) -> Optional[pd.Timestamp]:
        """
        Convert value to pandas Timestamp (normalized) or None.
        Accepts: Timestamp/date/datetime/string. Handles Series safely.
        """
        v = DataTransformer._first_scalar(v)
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        if isinstance(v, pd.Timestamp):
            return v.normalize()

        # datetime/date-like
        try:
            ts = pd.to_datetime(v, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.normalize()
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Consignee name / code derivation
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_consignee_code(txt: Any) -> Optional[str]:
        if not isinstance(txt, str) or not txt.strip():
            return None
        t = txt.strip()
        if "(" in t and ")" in t:
            try:
                inside = t.rsplit("(", 1)[1].split(")", 1)[0]
                inside = inside.strip()
                return inside[-9:].strip() or None
            except Exception:
                pass
        return t[-9:].strip() or None

    @staticmethod
    def _extract_consignee_name(txt: Any) -> Optional[str]:
        if not isinstance(txt, str) or not txt.strip():
            return None
        t = txt.strip()
        if "(" in t:
            base = t.rsplit("(", 1)[0]
            return base.strip() or None
        return t

    def _derive_consignee_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        source_col = None
        for candidate in [
            "consignee_raw",
            "consignee_codes",
            "Consignee Code (Multiple)",
        ]:
            if candidate in out.columns:
                source_col = candidate
                break

        if not source_col:
            self.logger.warning(
                "No consignee_raw / consignee_codes column found; skipping consignee derivation."
            )
            return out

        self.logger.info(
            "Deriving consignee_name / consignee_codes from '%s'.", source_col
        )

        out["consignee_name"] = out[source_col].apply(self._extract_consignee_name)
        out["consignee_codes"] = out[source_col].apply(self._extract_consignee_code)

        # If your downstream expects list for RLS (recommended), convert to list[str]
        # Keep it minimal: single code -> [code], none -> []
        out["consignee_codes"] = out["consignee_codes"].apply(
            lambda x: [x] if isinstance(x, str) and x else []
        )

        return out

    # -------------------------------------------------------------------------
    # Optimal dates (DP and FD)
    # -------------------------------------------------------------------------
    @staticmethod
    def _derive_optimal_ata_dp_date(row: pd.Series) -> Optional[pd.Timestamp]:
        today = pd.Timestamp("today").normalize()
        ata_dp = DataTransformer._to_date_or_none(row.get("ata_dp_date"))
        derived = DataTransformer._to_date_or_none(row.get("derived_ata_dp_date"))

        if isinstance(ata_dp, pd.Timestamp):
            return ata_dp
        if isinstance(derived, pd.Timestamp) and derived <= today:
            return derived
        return None

    @staticmethod
    def _derive_optimal_eta_fd_date(row: pd.Series) -> Optional[pd.Timestamp]:
        for c in ["predictive_eta_fd_date", "revised_eta_fd_date", "eta_fd_date"]:
            val = DataTransformer._to_date_or_none(row.get(c))
            if isinstance(val, pd.Timestamp):
                return val
        return None

    def _derive_optimal_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["optimal_ata_dp_date"] = out.apply(self._derive_optimal_ata_dp_date, axis=1)
        out["optimal_eta_fd_date"] = out.apply(self._derive_optimal_eta_fd_date, axis=1)
        return out

    # -------------------------------------------------------------------------
    # Delay flags & durations (DP / FD)
    # -------------------------------------------------------------------------
    @staticmethod
    def _derive_dp_delay(row: pd.Series) -> Tuple[str, float]:
        today = pd.Timestamp("today").normalize()

        optimal = DataTransformer._to_date_or_none(row.get("optimal_ata_dp_date"))
        eta = DataTransformer._to_date_or_none(row.get("eta_dp_date"))

        if optimal is None:
            if isinstance(eta, pd.Timestamp):
                delay_days = int((today - eta).days)
                if eta > today:
                    delay_days = 0
                    label = "on_time"
                else:
                    delay_days = max(0, delay_days)
                    label = "delay" if delay_days > 0 else "on_time"
            else:
                delay_days = 0
                label = "on_time"
        else:
            delay_days = int((optimal - eta).days)
            if delay_days > 0:
                label = "delay"
            elif delay_days < 0:
                label = "early"
            else:
                label = "on_time"
        return label, delay_days

    @staticmethod
    def _derive_fd_delay(row: pd.Series) -> Tuple[str, float]:
        today = pd.Timestamp("today").normalize()

        optimal = DataTransformer._to_date_or_none(row.get("optimal_eta_fd_date"))
        delivery = DataTransformer._to_date_or_none(
            row.get("delivery_to_consignee_date")
        )
        empty_ret = DataTransformer._to_date_or_none(
            row.get("empty_container_return_date")
        )

        if isinstance(delivery, pd.Timestamp):
            actual = delivery
            has_real_actual = True
        elif isinstance(empty_ret, pd.Timestamp):
            actual = empty_ret
            has_real_actual = True
        else:
            actual = today
            has_real_actual = False

        if isinstance(optimal, pd.Timestamp):
            delay_days = int((actual - optimal).days)
        else:
            delay_days = 0

        if has_real_actual:
            if delay_days > 0:
                label = "delay"
            elif delay_days < 0:
                label = "early"
            else:
                label = "on_time"
        else:
            if isinstance(optimal, pd.Timestamp):
                if optimal > today:
                    delay_days = 0
                    label = "on_time"
                else:
                    delay_days = max(0, delay_days)
                    label = "delay" if delay_days > 0 else "on_time"
            else:
                delay_days = 0
                label = "on_time"
        return label, delay_days

    def _derive_delay_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        dp = out.apply(self._derive_dp_delay, axis=1)
        fd = out.apply(self._derive_fd_delay, axis=1)
        out["delayed_dp"] = dp.apply(lambda x: x[0])
        out["dp_delayed_dur"] = dp.apply(lambda x: x[1])
        out["delayed_fd"] = fd.apply(lambda x: x[0])
        out["fd_delayed_dur"] = fd.apply(lambda x: x[1])
        return out

    # -------------------------------------------------------------------------
    # Shipment status
    # -------------------------------------------------------------------------
    @staticmethod
    def _derive_shipment_status_row(row: pd.Series) -> str:
        today = pd.Timestamp("today").normalize()

        empty_ret = DataTransformer._to_date_or_none(
            row.get("empty_container_return_date")
        )
        delivery = DataTransformer._to_date_or_none(
            row.get("delivery_to_consignee_date")
        )
        last_cy_arr = DataTransformer._to_date_or_none(
            row.get("equipment_arrived_at_last_cy_date")
        )
        last_cy_out = DataTransformer._to_date_or_none(
            row.get("out_gate_at_last_cy_date")
        )
        optimal_ata_dp = DataTransformer._to_date_or_none(
            row.get("optimal_ata_dp_date")
        )
        atd_flp = DataTransformer._to_date_or_none(row.get("atd_flp_date"))
        ata_flp = DataTransformer._to_date_or_none(row.get("ata_flp_date"))
        atd_lp = DataTransformer._to_date_or_none(row.get("atd_lp_date"))
        etd_lp = DataTransformer._to_date_or_none(row.get("etd_lp_date"))

        if isinstance(empty_ret, pd.Timestamp):
            return "EMPTY_CONTAINER_RETURNED"
        if isinstance(delivery, pd.Timestamp):
            return "DELIVERED"

        if isinstance(last_cy_out, pd.Timestamp):
            return "IN_INLAND_TRANSIT"
        if isinstance(last_cy_arr, pd.Timestamp):
            return "AT_LAST_CY"

        if isinstance(optimal_ata_dp, pd.Timestamp) and optimal_ata_dp <= today:
            return "AT_DP"

        # Ocean transit if departed TS (or load->ocean) but not yet at DP
        if isinstance(atd_flp, pd.Timestamp) and (
            not isinstance(optimal_ata_dp, pd.Timestamp) or optimal_ata_dp > today
        ):
            return "IN_OCEAN_TRANSIT"

        # At TS if arrived TS but not departed TS
        if isinstance(ata_flp, pd.Timestamp) and not isinstance(atd_flp, pd.Timestamp):
            return "AT_TS"

        # At origin if not left origin and ETD is today/future
        if (
            not isinstance(atd_lp, pd.Timestamp)
            and isinstance(etd_lp, pd.Timestamp)
            and etd_lp >= today
        ):
            return "AT_ORIGIN"

        return "UNKNOWN"

    def _derive_shipment_status(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["shipment_status"] = out.apply(self._derive_shipment_status_row, axis=1)
        return out

    # -------------------------------------------------------------------------
    # Critical dates summary
    # -------------------------------------------------------------------------
    @staticmethod
    def _fmt_date_val(val: Any) -> Optional[str]:
        if isinstance(val, pd.Timestamp):
            return val.strftime("%d-%b-%y")
        return None

    def _derive_critical_dates_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        def _summarise_row(row: pd.Series) -> str:
            parts: List[str] = []

            por = self._safe_text(row.get("place_of_receipt"))
            lp = self._safe_text(row.get("load_port"))
            flp = self._safe_text(row.get("final_load_port"))
            dp = self._safe_text(row.get("discharge_port"))
            last_cy = self._safe_text(row.get("equipment_arrived_at_last_cy_lcn"))
            final_dest = self._safe_text(row.get("final_destination"))

            etd_lp = self._to_date_or_none(row.get("etd_lp_date"))
            atd_lp = self._to_date_or_none(row.get("atd_lp_date"))
            ata_flp = self._to_date_or_none(row.get("ata_flp_date"))
            atd_flp = self._to_date_or_none(row.get("atd_flp_date"))
            optimal_ata_dp = self._to_date_or_none(row.get("optimal_ata_dp_date"))
            out_gate_from_dp = self._to_date_or_none(row.get("out_gate_from_dp_date"))
            equip_arr_last_cy = self._to_date_or_none(
                row.get("equipment_arrived_at_last_cy_date")
            )
            delivery = self._to_date_or_none(row.get("delivery_to_consignee_date"))
            empty_ret = self._to_date_or_none(row.get("empty_container_return_date"))

            # Leg 1
            leg1_desc = f"{por or 'POR'} → {lp or 'LOAD'}"
            dates1 = []
            if etd_lp:
                dates1.append(f"ETD {self._fmt_date_val(etd_lp)}")
            if atd_lp:
                dates1.append(f"ATD {self._fmt_date_val(atd_lp)}")
            if dates1:
                parts.append(
                    f"Leg 1 (POR → Load Port): {leg1_desc} | " + ", ".join(dates1)
                )

            # Leg 2: Load->TS
            if flp and flp != lp:
                leg2_desc = f"{lp or 'LOAD'} → {flp}"
                dates2 = []
                if ata_flp:
                    dates2.append(f"ATA {self._fmt_date_val(ata_flp)}")
                if atd_flp:
                    dates2.append(f"ATD {self._fmt_date_val(atd_flp)}")
                if dates2:
                    parts.append(
                        f"Leg 2 (Load → TS): {leg2_desc} | " + ", ".join(dates2)
                    )

            # Leg 3: Ocean to DP
            leg3_desc = f"{flp or lp or 'LOAD'} → {dp or 'DP'}"
            if optimal_ata_dp:
                parts.append(
                    f"Leg 3 (Ocean to DP): {leg3_desc} | ATA_DP {self._fmt_date_val(optimal_ata_dp)}"
                )

            # Leg 4: DP -> Last CY
            if last_cy:
                leg4_desc = f"{dp or 'DP'} → {last_cy}"
                dates4 = []
                if out_gate_from_dp:
                    dates4.append(f"OUT_GATE_DP {self._fmt_date_val(out_gate_from_dp)}")
                if equip_arr_last_cy:
                    dates4.append(
                        f"ARR_LAST_CY {self._fmt_date_val(equip_arr_last_cy)}"
                    )
                if dates4:
                    parts.append(
                        f"Leg 4 (DP → Last CY): {leg4_desc} | " + ", ".join(dates4)
                    )

            # Leg 5
            leg5_desc = f"{last_cy or dp or 'DP'} → {final_dest or 'FINAL'}"
            dates5 = []
            if delivery:
                dates5.append(f"DELIVERY {self._fmt_date_val(delivery)}")
            if empty_ret:
                dates5.append(f"EMPTY_RET {self._fmt_date_val(empty_ret)}")
            if dates5:
                parts.append(
                    f"Leg 5 (Last CY → Final): {leg5_desc} | " + ", ".join(dates5)
                )

            return " || ".join([p for p in parts if p]) if parts else ""

        out["critical_dates_summary"] = out.apply(_summarise_row, axis=1)
        return out

    # -------------------------------------------------------------------------
    # Delay reasons & workflow gaps
    # -------------------------------------------------------------------------
    @staticmethod
    def _derive_delay_reason_and_gap(row: pd.Series) -> Tuple[str, str]:
        reasons: List[str] = []
        gap_flag = "none_detected"

        dp_label = DataTransformer._safe_text(row.get("delayed_dp")).lower()
        fd_label = DataTransformer._safe_text(row.get("delayed_fd")).lower()

        dp_days = DataTransformer._first_scalar(row.get("dp_delayed_dur"))
        fd_days = DataTransformer._first_scalar(row.get("fd_delayed_dur"))

        last_cy_lcn = DataTransformer._safe_text(
            row.get("equipment_arrived_at_last_cy_lcn")
        )
        final_dest = DataTransformer._safe_text(row.get("final_destination"))

        if dp_label == "delay" and isinstance(dp_days, (float, int)) and dp_days > 0:
            reasons.append(
                f"Ocean / discharge-port leg is delayed versus ETA at DP by ~{int(dp_days)} day(s)."
            )

        if fd_label == "delay" and isinstance(fd_days, (float, int)) and fd_days > 0:
            reasons.append(
                f"Final delivery leg is delayed versus ETA at final destination by ~{int(fd_days)} day(s)."
            )

        if (
            fd_label == "delay"
            and last_cy_lcn
            and final_dest
            and last_cy_lcn != final_dest
        ):
            reasons.append(
                f"Last-mile from last CY {last_cy_lcn} to final destination {final_dest} appears delayed."
            )

        delivery = DataTransformer._to_date_or_none(
            row.get("delivery_to_consignee_date")
        )
        empty_ret = DataTransformer._to_date_or_none(
            row.get("empty_container_return_date")
        )
        if delivery is None and isinstance(empty_ret, pd.Timestamp):
            gap_flag = "missing_delivery_event_with_empty_return"

        if not reasons:
            reasons.append(
                "Shipment is on time against configured DP / final destination ETAs."
            )

        return "; ".join(reasons), gap_flag

    def _derive_delay_reason_and_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rr = out.apply(self._derive_delay_reason_and_gap, axis=1)
        out["delay_reason_summary"] = rr.apply(lambda x: x[0])
        out["workflow_gap_flags"] = rr.apply(lambda x: x[1])
        return out

    # -------------------------------------------------------------------------
    # Milestones (row-wise story)  **FIXED**
    # -------------------------------------------------------------------------
    @staticmethod
    def _derive_milestones_row(row: pd.Series) -> str:
        today = pd.Timestamp("today").normalize().date()

        status = DataTransformer._safe_text(
            row.get("shipment_status"), default="UNKNOWN"
        ).upper()
        display_status = status
        if status in {"EMPTY_RETURNED", "EMPTY_CONTAINER_RETURNED"}:
            display_status = "EMPTY CONTAINER RETURNED"

        # Locations
        por = DataTransformer._safe_text(row.get("place_of_receipt")) or "POR"
        lp = DataTransformer._safe_text(row.get("load_port")) or "Load Port"
        ts = DataTransformer._safe_text(row.get("final_load_port"))  # optional TS
        dp = DataTransformer._safe_text(row.get("discharge_port")) or "Discharge Port"

        dp_out_lcn = DataTransformer._safe_text(row.get("out_gate_from_dp_lcn")) or dp
        last_cy_lcn = (
            DataTransformer._safe_text(row.get("equipment_arrived_at_last_cy_lcn"))
            or DataTransformer._safe_text(row.get("last_cy_location"))
            or "Last CY"
        )
        cy_out_lcn = (
            DataTransformer._safe_text(row.get("out_gate_at_last_cy_lcn"))
            or last_cy_lcn
        )

        pod = DataTransformer._safe_text(row.get("place_of_delivery"))
        final_dest = (
            DataTransformer._safe_text(row.get("final_destination"))
            or pod
            or "Final Destination"
        )

        delivery_lcn = (
            DataTransformer._safe_text(row.get("delivery_to_consignee_lcn"))
            or final_dest
        )
        empty_rt_lcn = (
            DataTransformer._safe_text(row.get("empty_container_return_lcn"))
            or delivery_lcn
        )

        # Dates (as date objects)
        def d(v: Any) -> Optional[pd.Timestamp]:
            return DataTransformer._to_date_or_none(v)

        etd_lp = d(row.get("etd_lp_date"))
        atd_lp = d(row.get("atd_lp_date"))

        ata_flp = d(row.get("ata_flp_date"))
        atd_flp = d(row.get("atd_flp_date"))

        eta_dp = d(row.get("eta_dp_date"))
        ata_dp = d(row.get("optimal_ata_dp_date"))

        out_dp = d(row.get("out_gate_from_dp_date"))
        equip_arr_cy = d(row.get("equipment_arrived_at_last_cy_date"))
        out_cy = d(row.get("out_gate_at_last_cy_date"))

        eta_fd = d(row.get("optimal_eta_fd_date"))
        delivery = d(row.get("delivery_to_consignee_date"))
        empty_rt = d(row.get("empty_container_return_date"))

        # Rail signals
        rail_load_dt = d(row.get("rail_load_dp_date"))
        rail_load_lc = DataTransformer._safe_text(row.get("rail_load_dp_lcn"))
        rail_dep_dt = d(row.get("rail_departure_dp_date"))
        rail_dep_lc = DataTransformer._safe_text(row.get("rail_departure_dp_lcn"))
        rail_arr_dt = d(row.get("rail_arrival_destination_date"))
        rail_arr_lc = DataTransformer._safe_text(
            row.get("rail_arrival_destination_lcn")
        )

        delayed_dp = DataTransformer._safe_text(
            row.get("delayed_dp"), default="unknown"
        ).lower()
        dp_dur = DataTransformer._first_scalar(row.get("dp_delayed_dur"))
        dp_days = None
        if isinstance(dp_dur, (int, float)) and not (
            isinstance(dp_dur, float) and np.isnan(dp_dur)
        ):
            dp_days = int(dp_dur)

        delayed_fd = DataTransformer._safe_text(
            row.get("delayed_fd"), default="unknown"
        ).lower()
        fd_dur = DataTransformer._first_scalar(row.get("fd_delayed_dur"))
        fd_days = None
        if isinstance(fd_dur, (int, float)) and not (
            isinstance(fd_dur, float) and np.isnan(fd_dur)
        ):
            fd_days = int(fd_dur)

        leg_msgs: List[str] = []

        # Leg 1
        if atd_lp is not None:
            leg1_state = "COMPLETED"
            leg1_desc = f"departed {lp} on {atd_lp.date().isoformat()}."
        elif etd_lp is not None:
            if etd_lp.date() < today:
                leg1_state = "OVERDUE"
                leg1_desc = f"ETD {etd_lp.date().isoformat()} passed; actual departure not recorded."
            else:
                leg1_state = "PLANNED"
                leg1_desc = f"planned ETD {etd_lp.date().isoformat()} from {lp}."
        else:
            leg1_state = "UNKNOWN"
            leg1_desc = "no ETD/ATD recorded for origin leg."
        leg_msgs.append(
            f"Leg 1 (POR → Load Port) [{leg1_state}]: {por} → {lp}; {leg1_desc}"
        )

        # Leg 2
        if ts:
            if atd_flp is not None:
                leg2_state = "COMPLETED"
                parts = []
                if ata_flp is not None:
                    parts.append(f"arrived TS {ts} on {ata_flp.date().isoformat()}")
                parts.append(f"departed TS on {atd_flp.date().isoformat()}")
                leg2_desc = "; ".join(parts)
            elif ata_flp is not None:
                leg2_state = "PARTIAL"
                leg2_desc = f"arrived TS {ts} on {ata_flp.date().isoformat()}; TS departure not recorded."
            else:
                leg2_state = "PLANNED"
                leg2_desc = (
                    f"TS at {ts} configured, but no arrival/departure recorded yet."
                )
            leg_msgs.append(
                f"Leg 2 (Load Port → TS) [{leg2_state}]: {lp} → {ts}; {leg2_desc}"
            )
        else:
            leg_msgs.append(
                f"Leg 2 (Load Port → TS) [NOT_APPLICABLE]: no transshipment; direct ocean leg from {lp} to {dp}."
            )

        # Leg 3
        dp_arrived = ata_dp is not None and ata_dp.date() <= today
        if dp_arrived:
            leg3_state = "COMPLETED"
            if delayed_dp == "delay" and dp_days is not None and dp_days > 0:
                delay_txt = f"delayed by ~{dp_days} day(s) vs ETA DP."
            elif delayed_dp == "early" and dp_days is not None and dp_days < 0:
                delay_txt = f"early by ~{abs(dp_days)} day(s) vs ETA DP."
            elif delayed_dp == "on_time":
                delay_txt = "on time vs ETA DP."
            else:
                delay_txt = "arrival vs ETA DP not clearly classified."
            leg3_desc = f"arrived {dp} on {ata_dp.date().isoformat()} ({delay_txt})"
        else:
            if eta_dp is not None:
                if eta_dp.date() < today:
                    leg3_state = "OVERDUE"
                    leg3_desc = f"ETA DP {eta_dp.date().isoformat()} passed; actual arrival not recorded."
                else:
                    leg3_state = "IN_PROGRESS"
                    leg3_desc = f"in ocean transit towards {dp} (ETA {eta_dp.date().isoformat()})."
            else:
                leg3_state = "UNKNOWN"
                leg3_desc = "no ETA/ATA DP recorded."
        leg_msgs.append(
            f"Leg 3 (TS/Load → DP) [{leg3_state}]: {(ts or lp)} → {dp}; {leg3_desc}"
        )

        # Leg 4 (DP -> CY, include rail)
        rail_bits: List[str] = []
        if rail_load_dt is not None or rail_load_lc:
            rail_bits.append(
                f"rail load {rail_load_lc or dp_out_lcn} on {rail_load_dt.date().isoformat() if rail_load_dt else 'date not recorded'}"
            )
        if rail_dep_dt is not None or rail_dep_lc:
            rail_bits.append(
                f"rail depart {rail_dep_lc or dp_out_lcn} on {rail_dep_dt.date().isoformat() if rail_dep_dt else 'date not recorded'}"
            )
        if rail_arr_dt is not None or rail_arr_lc:
            rail_bits.append(
                f"rail arrive {rail_arr_lc or last_cy_lcn} on {rail_arr_dt.date().isoformat() if rail_arr_dt else 'date not recorded'}"
            )

        if equip_arr_cy is not None or rail_arr_dt is not None:
            leg4_state = "COMPLETED"
            arr_dt = equip_arr_cy or rail_arr_dt
            arr_lc = (
                last_cy_lcn
                if equip_arr_cy is not None
                else (rail_arr_lc or last_cy_lcn)
            )
            if out_dp is not None:
                leg4_desc = f"departed DP area ({dp_out_lcn}) on {out_dp.date().isoformat()}, arrived {arr_lc} on {arr_dt.date().isoformat()}."
            else:
                leg4_desc = f"arrived {arr_lc} on {arr_dt.date().isoformat()}; DP out-gate not recorded."
            if rail_bits:
                leg4_desc += " Rail events: " + "; ".join(rail_bits)
        elif out_dp is not None or rail_dep_dt is not None or rail_load_dt is not None:
            leg4_state = "IN_PROGRESS"
            parts = []
            if out_dp is not None:
                parts.append(
                    f"out-gated from DP area ({dp_out_lcn}) on {out_dp.date().isoformat()}"
                )
            if rail_bits:
                parts.append("Rail events: " + "; ".join(rail_bits))
            parts.append("arrival at last CY not recorded.")
            leg4_desc = "; ".join(parts)
        else:
            if status in {
                "IN_INLAND_TRANSIT",
                "AT_LAST_CY",
                "DELIVERED",
                "EMPTY_RETURNED",
                "EMPTY_CONTAINER_RETURNED",
            }:
                leg4_state = "UNKNOWN"
                leg4_desc = "status suggests DP→CY movement happened, but DP/rail/CY events are missing."
            else:
                leg4_state = "PLANNED"
                leg4_desc = "no DP→CY movement recorded yet."
        leg_msgs.append(
            f"Leg 4 (DP → Last CY) [{leg4_state}]: {dp} → {last_cy_lcn}; {leg4_desc}"
        )

        # Leg 5 (CY -> Final + Empty return)
        has_delivery = delivery is not None
        has_empty = empty_rt is not None

        if has_empty:
            leg5_state = "COMPLETED"
        elif has_delivery:
            leg5_state = "PARTIAL"
        else:
            if eta_fd is not None:
                leg5_state = "OVERDUE" if eta_fd.date() < today else "PLANNED"
            else:
                leg5_state = "UNKNOWN"

        if delayed_fd == "delay" and fd_days is not None and fd_days > 0:
            delay_txt = f"delayed by ~{fd_days} day(s) vs ETA FD."
        elif delayed_fd == "early" and fd_days is not None and fd_days < 0:
            delay_txt = f"early by ~{abs(fd_days)} day(s) vs ETA FD."
        elif delayed_fd == "on_time":
            delay_txt = "on time vs ETA FD."
        else:
            delay_txt = "completion vs ETA FD not clearly classified."

        details = []
        if out_cy is not None:
            details.append(
                f"departed last CY ({cy_out_lcn}) on {out_cy.date().isoformat()}"
            )
        if has_delivery:
            details.append(
                f"delivered at {delivery_lcn} on {delivery.date().isoformat()}"
            )
        elif not has_empty:
            if eta_fd is not None:
                details.append(f"delivery pending; ETA FD {eta_fd.date().isoformat()}")
            else:
                details.append("delivery pending; ETA FD not available")

        if has_empty:
            details.append(
                f"empty container returned at {empty_rt_lcn} on {empty_rt.date().isoformat()}"
            )
        else:
            details.append("empty container return pending")

        leg5_desc = "; ".join(details) + f" ({delay_txt})"
        leg_msgs.append(
            f"Leg 5 (Last CY → Final) [{leg5_state}]: {last_cy_lcn} → {final_dest}; {leg5_desc}"
        )

        header = f"Overall shipment status: {display_status}"
        return " || ".join([header] + leg_msgs)

    def _derive_milestones(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if out.columns.duplicated().any():
            dup = out.columns[out.columns.duplicated()].tolist()
            # Do not hard-fail; log (you can choose to raise)
            self.logger.warning(
                "Duplicate columns detected before milestone derivation: %s", dup
            )
        out["milestones"] = out.apply(self._derive_milestones_row, axis=1)
        return out

    # -------------------------------------------------------------------------
    # Vessel & carrier summaries
    # -------------------------------------------------------------------------
    @staticmethod
    def _derive_vessel_summary_row(row: pd.Series) -> str:
        vessels: List[str] = []
        first_vessel = DataTransformer._safe_text(row.get("first_vessel_name"))
        final_vessel = DataTransformer._safe_text(row.get("final_vessel_name"))

        if first_vessel:
            vessels.append(f"First vessel: {first_vessel}")
        if final_vessel and final_vessel != first_vessel:
            vessels.append(f"Final vessel: {final_vessel}")
        return " | ".join(vessels)

    @staticmethod
    def _derive_carrier_summary_row(row: pd.Series) -> str:
        parts: List[str] = []
        final_carrier_name = DataTransformer._safe_text(row.get("final_carrier_name"))
        true_carrier_scac_name = DataTransformer._safe_text(
            row.get("true_carrier_scac_name")
        )

        if final_carrier_name:
            parts.append(f"Operated by {final_carrier_name}")
        if true_carrier_scac_name and true_carrier_scac_name != final_carrier_name:
            parts.append(f"True operating carrier: {true_carrier_scac_name}")
        return " | ".join(parts)

    def _derive_vessel_and_carrier(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["vessel_summary"] = out.apply(self._derive_vessel_summary_row, axis=1)
        out["carrier_summary"] = out.apply(self._derive_carrier_summary_row, axis=1)
        return out

    # -------------------------------------------------------------------------
    # Port route summary
    # -------------------------------------------------------------------------
    @staticmethod
    def _derive_port_route_summary_row(row: pd.Series) -> str:
        route_parts: List[str] = []

        por = DataTransformer._safe_text(row.get("place_of_receipt"))
        load_port = DataTransformer._safe_text(row.get("load_port"))
        flp = DataTransformer._safe_text(row.get("final_load_port"))
        dp = DataTransformer._safe_text(row.get("discharge_port"))
        pod = DataTransformer._safe_text(row.get("place_of_delivery"))
        final_dest = DataTransformer._safe_text(row.get("final_destination"))

        if por and load_port and por != load_port:
            route_parts.append(f"Origin: {por}")
            route_parts.append(f"Load Port: {load_port}")
        else:
            if load_port:
                route_parts.append(f"Origin/Load: {load_port}")
            elif por:
                route_parts.append(f"Origin: {por}")

        if flp and flp != load_port:
            route_parts.append(f"Transshipment: {flp}")

        if dp:
            route_parts.append(f"Discharge: {dp}")

        if pod and pod != dp:
            route_parts.append(f"Delivery at: {pod}")

        if final_dest and final_dest != pod:
            route_parts.append(f"Final: {final_dest}")

        return " \u2192 ".join(route_parts)

    def _derive_port_route_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["port_route_summary"] = out.apply(
            self._derive_port_route_summary_row, axis=1
        )
        return out

    # -------------------------------------------------------------------------
    # Partition tags
    # -------------------------------------------------------------------------
    @staticmethod
    def add_partition_tags(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        def _source_group(row: pd.Series) -> Optional[str]:
            etd = DataTransformer._to_date_or_none(row.get("etd_lp_date"))
            return etd.strftime("%Y-%m") if isinstance(etd, pd.Timestamp) else None

        def _source_month_tag(row: pd.Series) -> Optional[str]:
            etd = DataTransformer._to_date_or_none(row.get("etd_lp_date"))
            return (
                etd.strftime("%b_%y").lower() if isinstance(etd, pd.Timestamp) else None
            )

        out["source_group"] = out.apply(_source_group, axis=1)
        out["source_month_tag"] = out.apply(_source_month_tag, axis=1)
        return out

    # -------------------------------------------------------------------------
    # combined_content builder for Azure AI Search
    # -------------------------------------------------------------------------
    @staticmethod
    def _fmt_value_for_text(val: Any) -> Optional[str]:
        if val is None:
            return None

        if isinstance(val, (list, tuple, set)):
            cleaned: List[str] = []
            for v in val:
                if v is None:
                    continue
                try:
                    if pd.isna(v):
                        continue
                except TypeError:
                    pass
                if hasattr(v, "strftime"):
                    cleaned.append(v.strftime("%d-%b-%y"))
                else:
                    s = str(v).strip()
                    if s:
                        cleaned.append(s)
            return ", ".join(cleaned) if cleaned else None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass

        if hasattr(val, "strftime"):
            try:
                return val.strftime("%d-%b-%y")
            except Exception:
                pass

        s = str(val).strip()
        return s or None

    @staticmethod
    def _fmt_event(
        label: str, date_val: Optional[str], lcn_val: Optional[str]
    ) -> Optional[str]:
        if not date_val and not lcn_val:
            return None
        if date_val and lcn_val:
            return f"{label}: {date_val} @ {lcn_val}"
        if date_val:
            return f"{label}: {date_val}"
        return f"{label} location: {lcn_val}"

    @classmethod
    def _build_combined_content(cls, row: pd.Series) -> str:
        parts: List[str] = []

        consignee_name = cls._fmt_value_for_text(row.get("consignee_name"))
        container_number = cls._fmt_value_for_text(row.get("container_number"))
        container_type = cls._fmt_value_for_text(row.get("container_type"))
        destination_service = cls._fmt_value_for_text(row.get("destination_service"))
        shipment_status = cls._fmt_value_for_text(row.get("shipment_status"))
        display_status = shipment_status
        if shipment_status and shipment_status.upper() in {
            "EMPTY_RETURNED",
            "EMPTY_CONTAINER_RETURNED",
        }:
            display_status = "EMPTY CONTAINER RETURNED"
        hot_container_flag = row.get("hot_container_flag")  # boolean
        seal_number = cls._fmt_value_for_text(row.get("seal_number"))

        load_port = cls._fmt_value_for_text(row.get("load_port"))
        discharge_port = cls._fmt_value_for_text(row.get("discharge_port"))
        final_destination = cls._fmt_value_for_text(row.get("final_destination"))
        last_cy_lcn = cls._fmt_value_for_text(
            row.get("equipment_arrived_at_last_cy_lcn")
        )

        header_fragments: List[str] = []

        if hot_container_flag:
            header_fragments.append("HOT CONTAINER")

        container_label = container_number
        if container_number and container_type:
            container_label = f"{container_number} ({container_type})"

        if container_label and consignee_name:
            header_fragments.append(
                f"Container: {container_label} | Consignee: {consignee_name}"
            )
        elif container_label:
            header_fragments.append(f"Container: {container_label}")
        elif consignee_name:
            header_fragments.append(f"Consignee: {consignee_name}")

        if container_type and not container_number:
            header_fragments.append(f"Container Type: {container_type}")

        if seal_number:
            header_fragments.append(f"Seal Number: {seal_number}")

        route_bits: List[str] = []
        if load_port:
            route_bits.append(f"Origin/Load: {load_port}")
        if discharge_port and last_cy_lcn:
            route_bits.append(
                f"Discharge: {discharge_port} \u2192 Last CY location: {last_cy_lcn}"
            )
        else:
            if discharge_port:
                route_bits.append(f"Discharge: {discharge_port}")
            if last_cy_lcn:
                route_bits.append(f"Last CY location: {last_cy_lcn}")
        if final_destination:
            route_bits.append(f"Final Destination: {final_destination}")
        if route_bits:
            header_fragments.append(" | ".join(route_bits))

        if destination_service:
            header_fragments.append(f"Destination service: {destination_service}")
        if display_status:
            header_fragments.append(f"Current shipment status: {display_status}")

        if header_fragments:
            parts.append(" | ".join(header_fragments))

        # Dates & delay snapshot
        etd_lp_date = cls._fmt_value_for_text(row.get("etd_lp_date"))
        eta_dp_date = cls._fmt_value_for_text(row.get("eta_dp_date"))
        optimal_ata_dp_date = cls._fmt_value_for_text(row.get("optimal_ata_dp_date"))
        optimal_eta_fd_date = cls._fmt_value_for_text(row.get("optimal_eta_fd_date"))

        delayed_dp = cls._fmt_value_for_text(row.get("delayed_dp"))
        dp_delayed_dur = cls._fmt_value_for_text(row.get("dp_delayed_dur"))
        delayed_fd = cls._fmt_value_for_text(row.get("delayed_fd"))
        fd_delayed_dur = cls._fmt_value_for_text(row.get("fd_delayed_dur"))

        key_dates: List[str] = []
        if etd_lp_date:
            key_dates.append(f"ETD LP: {etd_lp_date}")
        if eta_dp_date:
            key_dates.append(f"ETA DP: {eta_dp_date}")
        if optimal_ata_dp_date:
            key_dates.append(f"ATA DP (optimal): {optimal_ata_dp_date}")
        if optimal_eta_fd_date:
            key_dates.append(f"ETA FD (optimal): {optimal_eta_fd_date}")

        delay_bits: List[str] = []
        if delayed_dp and dp_delayed_dur is not None:
            delay_bits.append(f"DP delay: {delayed_dp}, Δ≈{dp_delayed_dur} day(s)")
        if delayed_fd and fd_delayed_dur is not None:
            delay_bits.append(f"FD delay: {delayed_fd}, Δ≈{fd_delayed_dur} day(s)")

        if key_dates or delay_bits:
            lines = []
            if key_dates:
                lines.append("Key dates: " + " | ".join(key_dates))
            if delay_bits:
                lines.append("Delay snapshot: " + " | ".join(delay_bits))
            parts.append(" ".join(lines))

        # Route summaries
        vessel_summary = cls._fmt_value_for_text(row.get("vessel_summary"))
        carrier_summary = cls._fmt_value_for_text(row.get("carrier_summary"))
        port_route_summary = cls._fmt_value_for_text(row.get("port_route_summary"))

        route_lines: List[str] = []
        if port_route_summary:
            route_lines.append(f"Route summary: {port_route_summary}")
        if last_cy_lcn:
            route_lines.append(f"Last CY location: {last_cy_lcn}")
        if vessel_summary:
            route_lines.append(f"Vessel summary: {vessel_summary}")
        if carrier_summary:
            route_lines.append(f"Carrier summary: {carrier_summary}")
        if route_lines:
            parts.append(" ".join(route_lines))

        # Timeline
        milestones = cls._fmt_value_for_text(row.get("milestones"))
        critical_dates_summary = cls._fmt_value_for_text(
            row.get("critical_dates_summary")
        )
        delay_reason_summary = cls._fmt_value_for_text(row.get("delay_reason_summary"))
        workflow_gap_flags = cls._fmt_value_for_text(row.get("workflow_gap_flags"))

        timeline_lines: List[str] = []
        if milestones:
            timeline_lines.append(f"Milestones: {milestones}")
        if critical_dates_summary:
            timeline_lines.append(f"Critical dates: {critical_dates_summary}")
        if delay_reason_summary:
            timeline_lines.append(f"Delay analysis: {delay_reason_summary}")
        if workflow_gap_flags and workflow_gap_flags.lower() != "none_detected":
            timeline_lines.append(f"Workflow gaps: {workflow_gap_flags}")
        if timeline_lines:
            parts.append(" ".join(timeline_lines))

        # Operational status & compliance
        current_departure_status = cls._fmt_value_for_text(
            row.get("current_departure_status")
        )
        current_arrival_status = cls._fmt_value_for_text(
            row.get("current_arrival_status")
        )
        filing_856_status = cls._fmt_value_for_text(row.get("856_filing_status"))
        get_isf_submission_date = cls._fmt_value_for_text(
            row.get("get_isf_submission_date")
        )
        delivery_to_consignee_date = cls._fmt_value_for_text(
            row.get("delivery_to_consignee_date")
        )
        empty_container_return_date = cls._fmt_value_for_text(
            row.get("empty_container_return_date")
        )

        def _is_no(val: Optional[str]) -> bool:
            return isinstance(val, str) and val.strip().lower() == "no"

        if delivery_to_consignee_date or empty_container_return_date:
            if _is_no(current_departure_status):
                current_departure_status = None
            if _is_no(current_arrival_status):
                current_arrival_status = None

        ops_status_bits: List[str] = []
        if current_departure_status:
            ops_status_bits.append(
                f"Current departure status: {current_departure_status}"
            )
        if current_arrival_status:
            ops_status_bits.append(f"Current arrival status: {current_arrival_status}")
        if ops_status_bits:
            parts.append("Operational status: " + " | ".join(ops_status_bits))

        compliance_bits: List[str] = []
        if filing_856_status:
            compliance_bits.append(f"856 filing status: {filing_856_status}")
        if get_isf_submission_date:
            compliance_bits.append(f"ISF submission: {get_isf_submission_date}")
        if compliance_bits:
            parts.append("Compliance: " + " | ".join(compliance_bits))

        # Additional operational events
        in_dc_date = cls._fmt_value_for_text(row.get("in-dc_date"))
        cargo_ready_date = cls._fmt_value_for_text(row.get("cargo_ready_date"))
        cargo_receiveds_date = cls._fmt_value_for_text(row.get("cargo_receiveds_date"))

        empty_container_dispatch_date = cls._fmt_value_for_text(
            row.get("empty_container_dispatch_date")
        )
        empty_container_dispatch_lcn = cls._fmt_value_for_text(
            row.get("empty_container_dispatch_lcn")
        )
        in_gate_date = cls._fmt_value_for_text(row.get("in_gate_date"))
        in_gate_lcn = cls._fmt_value_for_text(row.get("in_gate_lcn"))

        carrier_vehicle_load_date = cls._fmt_value_for_text(
            row.get("carrier_vehicle_load_date")
        )
        carrier_vehicle_load_lcn = cls._fmt_value_for_text(
            row.get("carrier_vehicle_load_lcn")
        )
        carrier_vehicle_unload_date = cls._fmt_value_for_text(
            row.get("carrier_vehicle_unload_date")
        )
        carrier_vehicle_unload_lcn = cls._fmt_value_for_text(
            row.get("carrier_vehicle_unload_lcn")
        )

        vehicle_departure_date = cls._fmt_value_for_text(
            row.get("vehicle_departure_date")
        )
        vehicle_departure_lcn = cls._fmt_value_for_text(
            row.get("vehicle_departure_lcn")
        )
        vehicle_arrival_date = cls._fmt_value_for_text(row.get("vehicle_arrival_date"))
        vehicle_arrival_lcn = cls._fmt_value_for_text(row.get("vehicle_arrival_lcn"))

        out_gate_from_dp_date = cls._fmt_value_for_text(
            row.get("out_gate_from_dp_date")
        )
        out_gate_from_dp_lcn = cls._fmt_value_for_text(row.get("out_gate_from_dp_lcn"))
        equipment_arrived_at_last_cy_date = cls._fmt_value_for_text(
            row.get("equipment_arrived_at_last_cy_date")
        )
        equipment_arrived_at_last_cy_lcn = cls._fmt_value_for_text(
            row.get("equipment_arrived_at_last_cy_lcn")
        )
        out_gate_at_last_cy_date = cls._fmt_value_for_text(
            row.get("out_gate_at_last_cy_date")
        )
        out_gate_at_last_cy_lcn = cls._fmt_value_for_text(
            row.get("out_gate_at_last_cy_lcn")
        )

        rail_load_dp_date = cls._fmt_value_for_text(row.get("rail_load_dp_date"))
        rail_load_dp_lcn = cls._fmt_value_for_text(row.get("rail_load_dp_lcn"))
        rail_departure_dp_date = cls._fmt_value_for_text(
            row.get("rail_departure_dp_date")
        )
        rail_departure_dp_lcn = cls._fmt_value_for_text(
            row.get("rail_departure_dp_lcn")
        )
        rail_arrival_destination_date = cls._fmt_value_for_text(
            row.get("rail_arrival_destination_date")
        )
        rail_arrival_destination_lcn = cls._fmt_value_for_text(
            row.get("rail_arrival_destination_lcn")
        )

        event_lines: List[str] = []
        for line in [
            cls._fmt_event("In-DC", in_dc_date, None),
            cls._fmt_event("Cargo ready", cargo_ready_date, None),
            cls._fmt_event("Cargo received", cargo_receiveds_date, None),
            cls._fmt_event(
                "Empty container dispatch",
                empty_container_dispatch_date,
                empty_container_dispatch_lcn,
            ),
            cls._fmt_event("In-gate", in_gate_date, in_gate_lcn),
            cls._fmt_event(
                "Carrier vehicle load",
                carrier_vehicle_load_date,
                carrier_vehicle_load_lcn,
            ),
            cls._fmt_event(
                "Carrier vehicle unload",
                carrier_vehicle_unload_date,
                carrier_vehicle_unload_lcn,
            ),
            cls._fmt_event(
                "Vehicle departure", vehicle_departure_date, vehicle_departure_lcn
            ),
            cls._fmt_event(
                "Vehicle arrival", vehicle_arrival_date, vehicle_arrival_lcn
            ),
            cls._fmt_event(
                "Out-gate from DP", out_gate_from_dp_date, out_gate_from_dp_lcn
            ),
            cls._fmt_event(
                "Equipment arrived at last CY",
                equipment_arrived_at_last_cy_date,
                equipment_arrived_at_last_cy_lcn,
            ),
            cls._fmt_event(
                "Out-gate at last CY", out_gate_at_last_cy_date, out_gate_at_last_cy_lcn
            ),
            cls._fmt_event("Rail load", rail_load_dp_date, rail_load_dp_lcn),
            cls._fmt_event(
                "Rail departure", rail_departure_dp_date, rail_departure_dp_lcn
            ),
            cls._fmt_event(
                "Rail arrival",
                rail_arrival_destination_date,
                rail_arrival_destination_lcn,
            ),
        ]:
            if line:
                event_lines.append(line)
        if event_lines:
            parts.append("Operational events: " + " | ".join(event_lines))

        # Cargo & parties / identifiers (kept lean)
        cargo_weight_kg = cls._fmt_value_for_text(row.get("cargo_weight_kg"))
        cargo_measure_cubic_meter = cls._fmt_value_for_text(
            row.get("cargo_measure_cubic_meter")
        )
        cargo_count = cls._fmt_value_for_text(row.get("cargo_count"))
        cargo_um = cls._fmt_value_for_text(row.get("cargo_um"))
        cargo_detail_count = cls._fmt_value_for_text(row.get("cargo_detail_count"))
        detail_cargo_um = cls._fmt_value_for_text(row.get("detail_cargo_um"))
        co2_tank_on_wheel = cls._fmt_value_for_text(row.get("co2_tank_on_wheel"))
        co2_well_to_wheel = cls._fmt_value_for_text(row.get("co2_well_to_wheel"))
        demurrage_free_days = cls._fmt_value_for_text(row.get("demurrage_free_days"))
        detention_free_days = cls._fmt_value_for_text(row.get("detention_free_days"))

        cargo_bits: List[str] = []
        metrics: List[str] = []
        if cargo_weight_kg:
            metrics.append(f"weight {cargo_weight_kg} kg")
        if cargo_measure_cubic_meter:
            metrics.append(f"volume {cargo_measure_cubic_meter} m3")
        if cargo_count and cargo_um:
            metrics.append(f"Cargo count: {cargo_count} {cargo_um}")
        elif cargo_count:
            metrics.append(f"Cargo count: {cargo_count}")
        if cargo_detail_count and detail_cargo_um:
            metrics.append(f"Detail cargo: {cargo_detail_count} {detail_cargo_um}")
        if metrics:
            cargo_bits.append("Cargo metrics: " + " ".join(metrics))
        co2_bits: List[str] = []
        if co2_tank_on_wheel:
            co2_bits.append(f"tank-to-wheel {co2_tank_on_wheel}")
        if co2_well_to_wheel:
            co2_bits.append(f"well-to-wheel {co2_well_to_wheel}")
        if co2_bits:
            cargo_bits.append("CO2: " + " | ".join(co2_bits))
        if cargo_bits:
            parts.append(" ".join(cargo_bits))

        free_days_bits: List[str] = []
        if demurrage_free_days:
            free_days_bits.append(f"Demurrage free days: {demurrage_free_days}")
        if detention_free_days:
            free_days_bits.append(f"Detention free days: {detention_free_days}")
        if free_days_bits:
            parts.append("Free days: " + " | ".join(free_days_bits))

        po_numbers = cls._fmt_value_for_text(row.get("po_numbers"))
        booking_numbers = cls._fmt_value_for_text(row.get("booking_numbers"))
        fcr_numbers = cls._fmt_value_for_text(row.get("fcr_numbers"))
        obl_nos = cls._fmt_value_for_text(row.get("obl_nos"))
        job_no = cls._fmt_value_for_text(row.get("job_no"))
        mcs_hbl = cls._fmt_value_for_text(row.get("mcs_hbl"))
        service_contract_number = cls._fmt_value_for_text(
            row.get("service_contract_number")
        )

        supplier_vendor_name = cls._fmt_value_for_text(row.get("supplier_vendor_name"))
        manufacturer_name = cls._fmt_value_for_text(row.get("manufacturer_name"))
        ship_to_party_name = cls._fmt_value_for_text(row.get("ship_to_party_name"))
        job_type = cls._fmt_value_for_text(row.get("job_type"))
        transport_mode = cls._fmt_value_for_text(row.get("transport_mode"))
        booking_approval_status = cls._fmt_value_for_text(
            row.get("booking_approval_status")
        )

        id_line: List[str] = []
        if po_numbers:
            id_line.append(f"PO Numbers: {po_numbers}")
        if booking_numbers:
            id_line.append(f"Booking Numbers: {booking_numbers}")
        if fcr_numbers:
            id_line.append(f"FCR Numbers: {fcr_numbers}")
        if obl_nos:
            id_line.append(f"Ocean BL Numbers: {obl_nos}")
        if job_no:
            id_line.append(f"Job No: {job_no}")
        if mcs_hbl:
            id_line.append(f"MCS HBL: {mcs_hbl}")
        if service_contract_number:
            id_line.append(f"Service Contract: {service_contract_number}")
        if id_line:
            parts.append("References: " + " | ".join(id_line))

        party_line: List[str] = []
        if consignee_name:
            party_line.append(f"Consignee: {consignee_name}")
        if ship_to_party_name:
            party_line.append(f"Ship-to party: {ship_to_party_name}")
        if supplier_vendor_name:
            party_line.append(f"Supplier/Vendor: {supplier_vendor_name}")
        if manufacturer_name:
            party_line.append(f"Manufacturer: {manufacturer_name}")
        if job_type:
            party_line.append(f"Job type: {job_type}")
        if transport_mode:
            party_line.append(f"Transport mode: {transport_mode}")
        if hot_container_flag is not None:
            party_line.append(f"Hot container flag: {hot_container_flag}")
        if booking_approval_status:
            party_line.append(f"Booking approval status: {booking_approval_status}")
        if party_line:
            parts.append("Parties & service: " + " | ".join(party_line))

        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            container_display = container_number or "UNKNOWN"
            return f"Shipment record for container {container_display}."

        return "\n\n".join(parts)

    def generate_search_content(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["combined_content"] = out.apply(self._build_combined_content, axis=1)
        return out

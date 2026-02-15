"""
writer.py
Fixes implemented:
1) `metadata` is ALWAYS a dict.
2) Output filename format: `shipment_<mmmyy>_<counter>.jsonl`
   - mmmyy like: jan26, feb26, mar26 ...
   - counter auto-increments based on existing files in the output directory.
3) Robust JSONL writing with UTF-8, one JSON object per line.
4) Hard max-size enforcement with automatic file roll-over (12MB by default).
5) Consistent normalized output schema across both input paths.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient

from .config import MAX_FILE_SIZE_BYTES

logger = logging.getLogger("shipment_ingestion")


def _mmmyy_from_dt(dt: datetime) -> str:
    # jan26, feb26, mar26 ...
    return dt.strftime("%b%y").lower()


def _ensure_dict_metadata(m: Any) -> Dict[str, Any]:
    """
    Enforce metadata dict.
    If someone accidentally passes a string (e.g., container number), wrap it safely.
    """
    if m is None:
        return {}
    if isinstance(m, dict):
        return m
    # Fallback: wrap non-dict metadata to avoid breaking downstream consumers
    return {"_raw_metadata": m}


def _is_missing_scalar(val: Any) -> bool:
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except Exception:
        return False


def _sanitize_for_json(val: Any) -> Any:
    """
    Recursively sanitize values for strict JSON serialization:
    - pd.NA/NaN/NaT -> None
    - datetime-like -> ISO string
    - set/tuple -> list
    """
    if isinstance(val, dict):
        return {str(k): _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_for_json(v) for v in val]
    if isinstance(val, tuple):
        return [_sanitize_for_json(v) for v in val]
    if isinstance(val, set):
        return [_sanitize_for_json(v) for v in val]

    if _is_missing_scalar(val):
        return None

    try:
        if hasattr(val, "isoformat"):
            return val.isoformat()
    except Exception:
        pass

    return val


def _safe_text_or_empty(val: Any) -> str:
    if _is_missing_scalar(val):
        return ""
    s = str(val).strip()
    if s.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return ""
    return s


def _safe_tag(tag: Optional[str]) -> str:
    raw = (tag or _mmmyy_from_dt(datetime.now())).strip().lower()
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^a-z0-9_-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or _mmmyy_from_dt(datetime.now())


def _json_default(o: Any) -> Any:
    """
    JSON serializer fallback for non-serializable types (e.g., pandas Timestamp).
    """
    try:
        # pandas Timestamp / datetime-like
        if hasattr(o, "isoformat"):
            return o.isoformat()
    except Exception:
        pass
    return str(o)


def _next_counter(output_dir: Path, mmmyy: str) -> int:
    """
    Scan output_dir for shipment_<mmmyy>_<n>.jsonl and return next counter.
    """
    pattern = re.compile(rf"^shipment_{re.escape(mmmyy)}_(\d+)\.jsonl$", re.IGNORECASE)
    max_n = 0
    if output_dir.exists():
        for p in output_dir.iterdir():
            if not p.is_file():
                continue
            m = pattern.match(p.name)
            if not m:
                continue
            try:
                n = int(m.group(1))
                max_n = max(max_n, n)
            except Exception:
                continue
    return max_n + 1


def _coerce_consignee_codes(val: Any) -> List[str]:
    """
    Normalize consignee_codes into list[str] for RLS.
    Accepts: list/tuple/set, comma-separated str, single str, None.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        out = []
        for x in val:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # handle comma-separated
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts if parts else [s]
    # fallback
    s = str(val).strip()
    return [s] if s else []


@dataclass(frozen=True)
class JsonlWriterConfig:
    output_dir: str = "output"
    # If you want strict enforcement of required keys, set True.
    strict: bool = False
    # Hard file cap (12 MB by default from config).
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES


class JsonlWriter:
    """
    Writes shipment documents into JSONL for indexing.

    Expected input doc format (recommended):
    {
      "id": "<unique id>",
      "content": "<text to embed/index>",
      "metadata": { ... },
      # optional additional top-level keys:
      # "consignee_codes": ["000123456", ...]   # keep at top-level
    }

    If docs arrive as raw transformed rows, we map the common fields:
      - id: carr_eqp_uid (fallback job_no, fallback container_number, fallback auto id)
      - content: combined_content (fallback milestones)
      - metadata: dict of core attributes (container_number, ports, status, tags...)
    """

    def __init__(
        self,
        config: Optional[JsonlWriterConfig] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or JsonlWriterConfig()
        self.logger = logger_ or logger
        self.generated_files: List[Path] = []

    def write(
        self, docs: Iterable[Dict[str, Any]], mmmyy: Optional[str] = None
    ) -> Path:
        """
        Write docs to: shipment_<mmmyy>_<counter>.jsonl

        Parameters
        ----------
        docs : Iterable[dict]
            Documents to write.
        mmmyy : Optional[str]
            If not provided, uses current month of runtime (e.g., jan26).

        Returns
        -------
        Path
            Path to the written JSONL file.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mmmyy_tag = _safe_tag(mmmyy)
        counter = _next_counter(output_dir, mmmyy_tag)
        max_file_size = int(self.config.max_file_size_bytes)
        if max_file_size <= 0:
            raise ValueError("max_file_size_bytes must be > 0.")

        docs_list = list(docs)  # allows logging counts and single-pass iteration
        self.logger.info(
            "Writing %d docs with cap=%d bytes per file and tag=%s",
            len(docs_list),
            max_file_size,
            mmmyy_tag,
        )

        file_counter = counter
        current_path: Optional[Path] = None
        first_path: Optional[Path] = None
        current_size = 0
        current_lines = 0
        total_lines = 0
        files_written_this_call: List[Path] = []
        f = None

        def _open_new_file() -> None:
            nonlocal current_path, first_path, current_size, current_lines, f
            current_path = output_dir / f"shipment_{mmmyy_tag}_{file_counter}.jsonl"
            if first_path is None:
                first_path = current_path
            current_size = 0
            current_lines = 0
            f = current_path.open("w", encoding="utf-8")

        try:
            for idx, doc in enumerate(docs_list, start=1):
                normalized = self._normalize_doc(doc, fallback_index=idx)
                line = (
                    json.dumps(normalized, ensure_ascii=False, default=_json_default)
                    + "\n"
                )
                line_bytes = len(line.encode("utf-8"))

                if line_bytes > max_file_size:
                    raise ValueError(
                        f"Record {idx} is {line_bytes} bytes, exceeds max file size {max_file_size}."
                    )

                if f is None:
                    _open_new_file()

                if current_lines > 0 and (current_size + line_bytes) > max_file_size:
                    f.close()
                    if current_path is not None:
                        files_written_this_call.append(current_path)
                    self.logger.info(
                        "Rolled file at %d bytes, %d line(s): %s",
                        current_size,
                        current_lines,
                        current_path,
                    )
                    file_counter += 1
                    _open_new_file()

                f.write(line)
                current_size += line_bytes
                current_lines += 1
                total_lines += 1

            # Preserve historical behavior: write an empty file if docs list is empty.
            if f is None:
                _open_new_file()
        finally:
            if f is not None:
                f.close()
                if current_path is not None:
                    files_written_this_call.append(current_path)

        self.generated_files.extend(files_written_this_call)
        self.logger.info(
            "JSONL write complete. Files=%d, lines=%d, first=%s",
            len(files_written_this_call),
            total_lines,
            first_path,
        )

        if first_path is None:
            raise RuntimeError("Writer failed to create output path.")
        return first_path

    def _normalize_doc(
        self, doc: Dict[str, Any], fallback_index: int
    ) -> Dict[str, Any]:
        """
        Ensure:
        - metadata is dict (never a string / container number)
        - id exists and is string
        - content exists and is string
        - consignee_codes is a list[str] in metadata (recommended for RLS)
        """
        if not isinstance(doc, dict):
            raise TypeError(f"Each doc must be a dict. Got: {type(doc)}")

        # Case A: already in expected format
        if "content" in doc or "metadata" in doc or "id" in doc:
            doc_id = doc.get("id")
            content = doc.get("content")
            metadata = _ensure_dict_metadata(doc.get("metadata"))

            # Handle common bug: metadata accidentally equals container_number or other scalar.
            # _ensure_dict_metadata already wraps it, but we also try to populate proper metadata.
            if "_raw_metadata" in metadata and isinstance(
                metadata["_raw_metadata"], str
            ):
                # If the raw metadata looks like a container number, store it properly.
                metadata.setdefault("container_number", metadata["_raw_metadata"])

            # If content missing but combined_content exists (your transformer output)
            if content is None and "combined_content" in doc:
                content = doc.get("combined_content")

            # If still missing, fallback to milestones or empty
            if content is None:
                content = doc.get("milestones") or ""

            # Build id fallback chain
            if not doc_id:
                doc_id = (
                    doc.get("carr_eqp_uid")
                    or doc.get("job_no")
                    or doc.get("container_number")
                    or f"doc_{fallback_index}"
                )

            # RLS consignee codes: prefer at top-level or inside metadata
            if "consignee_codes" in doc:
                consignee_codes = _coerce_consignee_codes(doc.get("consignee_codes"))
            else:
                consignee_codes = _coerce_consignee_codes(
                    metadata.get("consignee_codes")
                )
            metadata["consignee_codes"] = consignee_codes

            out = {
                "document_id": _safe_text_or_empty(doc_id) or f"doc_{fallback_index}",
                "content": _safe_text_or_empty(content),
                "consignee_code": consignee_codes,
                "metadata": _sanitize_for_json(metadata),
            }

            self._validate(out)
            return out

        # Case B: raw row-like dict from dataframe (no id/content keys)
        doc_id = (
            doc.get("carr_eqp_uid")
            or doc.get("job_no")
            or doc.get("container_number")
            or f"doc_{fallback_index}"
        )
        content = doc.get("combined_content") or doc.get("milestones") or ""

        metadata = dict(doc)
        consignee_codes = _coerce_consignee_codes(doc.get("consignee_codes"))
        metadata["consignee_codes"] = consignee_codes
        if "carr_eqp_uid" in metadata:
            del metadata["carr_eqp_uid"]
        if "consignee_raw" in metadata:
            del metadata["consignee_raw"]
        if "combined_content" in metadata:
            del metadata["combined_content"]

        out = {
            "document_id": _safe_text_or_empty(doc_id) or f"doc_{fallback_index}",
            "content": _safe_text_or_empty(content),
            "consignee_code": consignee_codes,
            "metadata": _sanitize_for_json(metadata),
        }
        self._validate(out)
        return out

    def _validate(self, normalized: Dict[str, Any]) -> None:
        """
        Lightweight validation. In strict mode, raise on missing/empty content.
        """
        if "document_id" not in normalized:
            raise ValueError("Normalized doc missing 'document_id'.")
        if "content" not in normalized:
            raise ValueError("Normalized doc missing 'content'.")
        if "metadata" not in normalized:
            raise ValueError("Normalized doc missing 'metadata'.")

        if not isinstance(normalized["metadata"], dict):
            raise ValueError("Normalized doc 'metadata' must be a dict.")

        if self.config.strict:
            if not str(normalized["document_id"]).strip():
                raise ValueError("Doc 'document_id' is empty in strict mode.")
            if not str(normalized["content"]).strip():
                raise ValueError("Doc 'content' is empty in strict mode.")

    def upload_files(self, conn_str: str, container_name: str) -> None:
        """
        Uploads all generated files to the specified Azure Blob container.
        """
        self.logger.info(
            f"Uploading {len(self.generated_files)} file(s) to container '{container_name}'..."
        )
        try:
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            container_client = blob_service_client.get_container_client(container_name)

            # Ensure container exists
            if not container_client.exists():
                self.logger.warning(
                    f"Container '{container_name}' does not exist. Creating it..."
                )
                container_client.create_container()

            for file_path in self.generated_files:
                blob_name = file_path.name
                blob_client = container_client.get_blob_client(blob_name)

                self.logger.info(f"Uploading {blob_name}...")
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

            self.logger.info("All files uploaded successfully.")

        except Exception as e:
            self.logger.error(f"Upload failed: {e}", exc_info=True)
            raise


# Convenience function if your pipeline uses functional style
def write_jsonl(
    docs: Iterable[Dict[str, Any]],
    output_dir: str = "output",
    mmmyy: Optional[str] = None,
    strict: bool = False,
) -> str:
    writer = JsonlWriter(JsonlWriterConfig(output_dir=output_dir, strict=strict))
    path = writer.write(docs, mmmyy=mmmyy)
    return str(path)

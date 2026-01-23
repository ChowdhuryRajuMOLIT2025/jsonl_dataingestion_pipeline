import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from azure.storage.blob import BlobServiceClient

from .config import ENV_VARS, MAX_FILE_SIZE_BYTES

logger = logging.getLogger("shipment_ingestion")


class JsonlWriter:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_files = []

    def write_partitioned_jsonl(self, df: pd.DataFrame):
        """
        Splits DataFrame by 'source_group' and writes to size-limited JSONL files.
        """
        logger.info("Starting JSONL writing process...")

        # Group by the source tag
        grouped = df.groupby("source_group")

        total_groups = len(grouped)
        logger.info(f"Found {total_groups} partition groups.")

        for group_name, group_df in grouped:
            self._write_group(str(group_name), group_df)

        logger.info(f"Writing completed. Generated {len(self.generated_files)} files.")
        return self.generated_files

    def _write_group(self, group_name: str, df: pd.DataFrame):
        # Sort by some logical order if needed (e.g. date)
        # df = df.sort_values("etd_lp_date") # Optional

        records = df.to_dict(orient="records")

        file_counter = 1
        current_file_path = self.output_dir / f"{group_name}_{file_counter}.jsonl"
        current_file_size = 0
        current_handle = open(current_file_path, "w", encoding="utf-8", newline="")
        self.generated_files.append(current_file_path)

        logger.info(f"Writing group '{group_name}' (Rows: {len(records)})")

        try:
            for record in records:
                # 1. Sanitize flat record first (handle NaNs, dates, lists)
                clean_flat = self._sanitize_record(record)

                # 2. Restructure for Azure Search
                # User Requirement:
                # document_id -> carr_eqp_uid
                # content -> combined_content
                # consignee_code -> consignee_codes (List of strings)
                # metadata -> clean_flat (all fields)

                final_record = {
                    "document_id": clean_flat.get("carr_eqp_uid"),
                    "content": clean_flat.get("combined_content"),
                    "consignee_code": clean_flat.get("consignee_codes"),
                    "metadata": clean_flat,
                }

                # Check critical keys (optional validation)
                if not final_record["document_id"]:
                    logger.warning(
                        "Skipping record without 'carr_eqp_uid' (document_id)."
                    )
                    continue

                line = json.dumps(final_record, ensure_ascii=False) + "\n"
                line_bytes = len(line.encode("utf-8"))

                # Check size
                if current_file_size + line_bytes > MAX_FILE_SIZE_BYTES:
                    # Roll file
                    current_handle.close()
                    file_counter += 1
                    current_file_path = (
                        self.output_dir / f"{group_name}_{file_counter}.jsonl"
                    )
                    current_file_size = 0
                    current_handle = open(
                        current_file_path, "w", encoding="utf-8", newline=""
                    )
                    self.generated_files.append(current_file_path)

                current_handle.write(line)
                current_file_size += line_bytes

        finally:
            if current_handle:
                current_handle.close()

    def _sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts NaT/NaN to None so JSON serializer handles them as null.
        """
        new_record = {}
        for k, v in record.items():
            # SAFETY: Check for list first to avoid "truth value of an array is ambiguous" error
            if isinstance(v, list):
                new_record[k] = v
                continue

            if pd.isna(v):
                new_record[k] = None
            elif isinstance(v, (pd.Timestamp, datetime.date, datetime.datetime)):
                new_record[k] = str(v)
            else:
                new_record[k] = v
        return new_record

    def upload_files(self, conn_str: str, container_name: str):
        """
        Uploads all generated files to the specified Azure Blob container.
        """
        logger.info(
            f"Uploading {len(self.generated_files)} files to container '{container_name}'..."
        )
        try:
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            container_client = blob_service_client.get_container_client(container_name)

            # Ensure container exists
            if not container_client.exists():
                logger.warning(
                    f"Container '{container_name}' does not exist. Creating it..."
                )
                container_client.create_container()

            for file_path in self.generated_files:
                blob_name = file_path.name
                blob_client = container_client.get_blob_client(blob_name)

                logger.info(f"Uploading {blob_name}...")
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

            logger.info("All files uploaded successfully.")

        except Exception as e:
            logger.error(f"Upload failed: {e}", exc_info=True)
            # We don't raise here to avoid crashing after successful processing, but we log strictly.
            raise


import datetime

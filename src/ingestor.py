
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, HttpResponseError

logger = logging.getLogger("shipment_ingestion")

class DataIngestor:
    def __init__(self, conn_str: str, container_name: str, download_dir: str = "downloads"):
        self.conn_str = conn_str
        self.container_name = container_name
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.conn_str)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)
        except Exception as e:
            logger.error("Failed to initialize Blob Service Client.", exc_info=True)
            raise

    def find_latest_csv_blob(self, prefix: Optional[str] = None) -> Tuple[str, datetime]:
        """
        Find latest CSV in container by last_modified.
        """
        latest_name = None
        latest_lm = None

        try:
            logger.info(f"Scanning container '{self.container_name}' for CSV files...")
            for blob in self.container_client.list_blobs(name_starts_with=prefix):
                name = blob.name
                if not name or not name.lower().endswith(".csv"):
                    continue
                lm = blob.last_modified
                if latest_lm is None or lm > latest_lm:
                    latest_name = name
                    latest_lm = lm
            
            if not latest_name:
                raise FileNotFoundError(f"No .csv blobs found in container '{self.container_name}'")
                
            logger.info(f"Found latest CSV: {latest_name} (Last Modified: {latest_lm})")
            return latest_name, latest_lm

        except Exception as e:
            logger.error("Error finding latest CSV blob.", exc_info=True)
            raise

    def download_blob(self, blob_name: str) -> Path:
        """
        Downloads a specific blob to the local download directory.
        """
        local_path = self.download_dir / Path(blob_name).name
        try:
            logger.info(f"Downloading blob '{blob_name}' to '{local_path}'...")
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as f:
                stream = blob_client.download_blob(max_concurrency=4)
                stream.readinto(f)
            logger.info("Download completed successfully.")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download blob '{blob_name}'.", exc_info=True)
            raise

    def read_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Reads CSV with strict string types to preserve IDs and formatting.
        """
        logger.info(f"Reading CSV file: {csv_path}")
        try:
            # dtype=str is crucial to prevent leading zero loss in IDs
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, low_memory=False, encoding="utf-8-sig")
            logger.info(f"CSV loaded. Shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            logger.warning("UTF-8 strict decoding failed. Retrying with 'iso-8859-1'...")
            try:
                df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, low_memory=False, encoding="iso-8859-1")
                logger.info(f"CSV loaded with fallback encoding. Shape: {df.shape}")
                return df
            except Exception as e:
                logger.error("Failed to read CSV with fallback encoding.", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"Failed to read CSV file: {csv_path}", exc_info=True)
            raise

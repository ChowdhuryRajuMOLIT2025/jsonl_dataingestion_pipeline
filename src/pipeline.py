import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .config import ENV_VARS, MAX_FILE_SIZE_BYTES
from .ingestor import DataIngestor
from .transformer import DataTransformer
from .writer import JsonlWriter, JsonlWriterConfig


# Setup Logger
def setup_logging():
    logger = logging.getLogger("shipment_ingestion")
    logger.setLevel(logging.INFO)

    # Ensure handlers are only added once
    if not logger.handlers:
        # 1. Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_format = logging.Formatter("[%(levelname)s] %(asctime)s | %(message)s")
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # 2. File Handler
        try:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            f_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
            f_format = logging.Formatter(
                "[%(levelname)s] %(asctime)s | %(module)s | %(message)s"
            )
            f_handler.setFormatter(f_format)
            logger.addHandler(f_handler)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")

    return logger


logger = setup_logging()


class ShipmentDataPipeline:
    def __init__(self):
        self.config = {}

    def load_configuration(self):
        logger.info("Loading configuration...")
        load_dotenv(find_dotenv(), override=True)

        missing = []
        for var in ENV_VARS:
            val = os.getenv(var)
            if not val:
                missing.append(var)
            self.config[var] = val

        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            raise EnvironmentError(f"Missing required ENV variables: {missing}")

        logger.info("Configuration loaded successfully.")

    def run(self):
        try:
            total_start = time.time()
            logger.info("Starting Data Pipeline Execution...")
            self.load_configuration()

            # 1. Ingestion
            t0 = time.time()
            # Using 'WNLD' container for download as per original script logic
            ingestor = DataIngestor(
                conn_str=self.config["AZURE_STORAGE_CONN_STR"],
                container_name=self.config["AZURE_STORAGE_CONTAINER_WNLD"],
            )

            # Find latest
            try:
                latest_csv, _ = ingestor.find_latest_csv_blob()
                local_csv_path = ingestor.download_blob(latest_csv)
            except FileNotFoundError:
                logger.error("No CSV found to process. Terminating pipeline.")
                return
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
                raise

            # Read into DataFrame
            df = ingestor.read_csv(local_csv_path)
            t1 = time.time()
            logger.info(f"Step 1: Ingestion completed in {t1 - t0:.2f} seconds.")

            # 2. Transformation
            logger.info("Starting Transformation Step...")
            transformer = DataTransformer()
            processed_df = transformer.run_pipeline(df)
            t2 = time.time()
            logger.info(f"Step 2: Transformation completed in {t2 - t1:.2f} seconds.")

            # 2.5 Parquet snapshot (full transformed dataset)
            parquet_dir = Path("output_parquet")
            parquet_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = parquet_dir / "master_ds.parquet"
            if parquet_path.exists():
                backup_dir = parquet_dir / "backup"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_name = f"master_ds_{datetime.now().strftime('%d%b%y').lower()}.parquet"
                backup_path = backup_dir / backup_name
                parquet_path.replace(backup_path)
            processed_df.to_parquet(parquet_path, index=False)
            logger.info("Parquet snapshot saved to %s", parquet_path)

            # 3. Writing
            logger.info("Starting Writing Step (JSONL Generation)...")
            writer = JsonlWriter(config=JsonlWriterConfig(output_dir="output_jsonl"))
            grouped = processed_df.groupby("source_group")
            generated_files = []
            for group_name, group_df in grouped:
                docs = group_df.to_dict(orient="records")
                path = writer.write(docs, mmmyy=str(group_name))
                generated_files.append(path)
            logger.info(f"Writing completed. Generated {len(generated_files)} file(s).")
            t3 = time.time()
            logger.info(f"Step 3: Writing completed in {t3 - t2:.2f} seconds.")

            # 4. Uploading
            logger.info("Starting Upload Step...")
            target_container = self.config.get("AZURE_STORAGE_CONTAINER_UPLD")
            if not target_container:
                logger.warning(
                    "AZURE_STORAGE_CONTAINER_UPLD not set. Defaulting to 'shipment-csv-data'."
                )
                target_container = "shipment-csv-data"

            writer.upload_files(
                conn_str=self.config["AZURE_STORAGE_CONN_STR"],
                container_name=target_container,
            )
            t4 = time.time()
            logger.info(f"Step 4: Upload completed in {t4 - t3:.2f} seconds.")

            total_duration = t4 - total_start
            logger.info(
                f"Pipeline executed successfully in {total_duration:.2f} seconds."
            )

        except Exception as e:
            logger.error("Pipeline execution failed.", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    pipeline = ShipmentDataPipeline()
    pipeline.run()

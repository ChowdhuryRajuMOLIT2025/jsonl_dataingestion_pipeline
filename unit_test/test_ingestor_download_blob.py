from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.ingestor import DataIngestor


def _make_ingestor(download_dir: Path, container_client: Mock) -> DataIngestor:
    ingestor = DataIngestor.__new__(DataIngestor)
    ingestor.download_dir = download_dir
    ingestor.container_client = container_client
    ingestor.container_name = "test-container"
    return ingestor


def test_download_blob_skips_when_local_file_exists() -> None:
    tmp_root = Path(__file__).resolve().parent.parent / "pytest_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    test_root = tmp_root / f"run_{uuid.uuid4().hex}"
    download_dir = test_root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    existing = download_dir / "sample.csv"
    existing.write_text("already here", encoding="utf-8")

    container_client = Mock()
    container_client.get_blob_client.side_effect = AssertionError(
        "Blob download should be skipped when file exists."
    )
    ingestor = _make_ingestor(download_dir, container_client)

    out = ingestor.download_blob("some/folder/sample.csv")
    assert out == existing
    container_client.get_blob_client.assert_not_called()


def test_download_blob_overwrite_true_downloads() -> None:
    tmp_root = Path(__file__).resolve().parent.parent / "pytest_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    test_root = tmp_root / f"run_{uuid.uuid4().hex}"
    download_dir = test_root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    existing = download_dir / "sample.csv"
    existing.write_text("old", encoding="utf-8")

    blob_client = Mock()
    blob_client.download_blob.return_value = SimpleNamespace(
        readinto=lambda f: f.write(b"new")
    )

    container_client = Mock()
    container_client.get_blob_client.return_value = blob_client

    ingestor = _make_ingestor(download_dir, container_client)

    out = ingestor.download_blob("sample.csv", overwrite=True)
    assert out == existing
    assert existing.read_bytes() == b"new"
    container_client.get_blob_client.assert_called_once()

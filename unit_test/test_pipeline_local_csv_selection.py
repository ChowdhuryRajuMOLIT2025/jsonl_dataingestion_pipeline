from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from src.pipeline import _find_latest_local_csv


def test_find_latest_local_csv_none_when_missing() -> None:
    tmp_root = Path(__file__).resolve().parent.parent / "pytest_tmp"
    missing = tmp_root / f"run_{uuid.uuid4().hex}" / "downloads"
    assert _find_latest_local_csv(missing) is None


def test_find_latest_local_csv_picks_newest() -> None:
    tmp_root = Path(__file__).resolve().parent.parent / "pytest_tmp"
    d = tmp_root / f"run_{uuid.uuid4().hex}" / "downloads"
    d.mkdir(parents=True, exist_ok=True)

    older = d / "older.csv"
    newer = d / "newer.csv"

    older.write_text("a", encoding="utf-8")
    newer.write_text("b", encoding="utf-8")

    now = time.time()
    os.utime(older, (now - 10, now - 10))
    os.utime(newer, (now, now))

    assert _find_latest_local_csv(d) == newer

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


def read_last_run_date(state_file: Path) -> date | None:
    """Read the last successful run date from file.

    Returns None if the file doesn't exist (first run = full backfill).
    """
    if not state_file.exists():
        logger.info("No state file found at %s — first run", state_file)
        return None

    text = state_file.read_text().strip()
    if not text:
        return None

    parsed = date.fromisoformat(text)
    logger.info("Last run date: %s", parsed)
    return parsed


def write_last_run_date(state_file: Path, run_date: date) -> None:
    """Persist run date atomically (write to temp, then rename)."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=state_file.parent, suffix=".tmp"
    )
    os.close(tmp_fd)
    tmp = Path(tmp_path)
    try:
        tmp.write_text(run_date.isoformat())
        tmp.rename(state_file)
        logger.info("Wrote last run date: %s", run_date)
    finally:
        if tmp.exists():
            tmp.unlink()

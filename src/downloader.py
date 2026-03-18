from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import DOWNLOAD_TIMEOUT_SECONDS
from .db.interface import TranscriptRef

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadResult:
    ref: TranscriptRef
    file_path: Path | None
    error: str | None


def download_transcripts(
    refs: list[TranscriptRef],
    temp_dir: Path,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
) -> list[DownloadResult]:
    """Download transcript JSONs from URLs into temp_dir.

    Each file is saved as {event_id}.json. Failed downloads are
    logged and included in results with error set.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []

    for ref in refs:
        dest = temp_dir / f"{ref.event_id}.json"
        try:
            resp = requests.get(ref.url, timeout=timeout)
            resp.raise_for_status()

            # Validate it's parseable JSON
            json.loads(resp.text)

            dest.write_text(resp.text, encoding="utf-8")
            results.append(DownloadResult(ref=ref, file_path=dest, error=None))
            logger.info("Downloaded event_id=%s", ref.event_id)

        except requests.RequestException as exc:
            msg = f"Download failed for event_id={ref.event_id}: {exc}"
            logger.warning(msg)
            results.append(DownloadResult(ref=ref, file_path=None, error=msg))

        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON for event_id={ref.event_id}: {exc}"
            logger.warning(msg)
            results.append(DownloadResult(ref=ref, file_path=None, error=msg))

    return results


def load_transcript_json(file_path: Path) -> dict[str, Any] | None:
    """Load and validate a transcript JSON file.

    Returns None (with logged warning) if the file is invalid.
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", file_path, exc)
        return None

    required = {"event_id", "company_id", "transcript"}
    missing = required - set(data.keys())
    if missing:
        logger.warning("Missing keys %s in %s", missing, file_path)
        return None

    if "paragraphs" not in data.get("transcript", {}):
        logger.warning("Missing transcript.paragraphs in %s", file_path)
        return None

    return data


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove all contents of the temp directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temp dir: %s", temp_dir)

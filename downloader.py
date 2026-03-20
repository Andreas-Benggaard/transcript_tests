import json
import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm

# Internal session library
from Utilities import CommonLibrary as CL  # type: ignore

from config import DOWNLOAD_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class DownloadResult:
    def __init__(self, ref, file_path, error):
        self.ref = ref
        self.file_path = file_path
        self.error = error


def download_transcripts(refs, temp_dir, timeout=DOWNLOAD_TIMEOUT_SECONDS):
    """Download transcript JSONs from URLs into temp_dir.

    Each file is saved as {event_id}.json. Failed downloads are
    logged and included in results with error set.
    """
    if isinstance(temp_dir, str):
        temp_dir = Path(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)
    results = []

    s = CL.getSession(external=1, SSL=1)
    s.headers.update({"User-Agent": "andreas.benggaard@nordea.com"})

    for ref in tqdm(refs, desc="Downloading JSON transcripts"):
        dest = temp_dir / f"{ref.event_id}.json"
        try:
            resp = s.get(ref.url, timeout=timeout)
            resp.raise_for_status()

            # Save the file
            data = resp.json()
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            logger.info("Downloaded event_id=%s", ref.event_id)
            results.append(DownloadResult(ref=ref, file_path=dest, error=None))

        except Exception as exc:
            msg = f"Download failed for event_id={ref.event_id}: {exc}"
            logger.warning(msg)
            results.append(DownloadResult(ref=ref, file_path=None, error=msg))

    return results


def download_transcript_jsons(df, temp_dir):
    """Original function — downloads each URL in df to temp/<id>.json.
    One-by-one, validates JSON. Returns list of saved file paths.
    """
    s = CL.getSession(external=1, SSL=1)
    s.headers.update({"User-Agent": "andreas.benggaard@nordea.com"})

    os.makedirs(temp_dir, exist_ok=True)
    saved_files = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading JSON transcripts"):
        doc_id = int(row["id"])
        url = row["fileUrl"]

        if not isinstance(url, str) or not url.startswith("http"):
            print(f"[SKIP] Invalid URL for id={doc_id}")
            continue

        out_path = os.path.join(temp_dir, f"{doc_id}.json")
        try:
            r = s.get(url, timeout=60)
            data = r.json()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            saved_files.append(out_path)
        except Exception as e:
            print(f"[ERROR] Failed id={doc_id} url={url}\n{e}")

    return saved_files


def load_transcript_json(file_path):
    """Load and validate a transcript JSON file.

    Returns None (with logged warning) if the file is invalid.
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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


def cleanup_temp_dir(temp_dir):
    """Remove all contents of the temp directory."""
    if isinstance(temp_dir, str):
        temp_dir = Path(temp_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temp dir: %s", temp_dir)

"""Weekly entry point for the sentiment analysis pipeline.

Usage:
    uv run python run_weekly.py
    uv run python run_weekly.py --full-backfill
    uv run python run_weekly.py --upload-sentences

Set up as a cron job or scheduled task to run every weekend.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.batch_runner import run_batch_pipeline
from src.config import STATE_FILE

# ── Replace this import with your real DB implementation ──
from src.db.stub import MSSQLTranscriptDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_run.log"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run weekly earnings call sentiment analysis"
    )
    parser.add_argument(
        "--full-backfill",
        action="store_true",
        help="Ignore last_run_date and process all transcripts",
    )
    parser.add_argument(
        "--upload-sentences",
        action="store_true",
        help="Also upload sentence-level results to SQL",
    )
    parser.add_argument(
        "--connection-string",
        default="",
        help="MSSQL connection string",
    )
    args = parser.parse_args()

    state_file = None if args.full_backfill else STATE_FILE

    # ── Replace with your real DB ──
    db = MSSQLTranscriptDB(connection_string=args.connection_string)

    report = run_batch_pipeline(
        db=db,
        state_file=state_file,
        upload_sentences=args.upload_sentences,
    )

    logger.info("=== Run Summary ===")
    logger.info("Processed: %d", report.total_processed)
    logger.info("Failed:    %d", report.total_failed)
    if report.failed_event_ids:
        logger.warning("Failed event_ids: %s", report.failed_event_ids)
    logger.info("Duration:  %.1f seconds", report.duration_seconds)

    sys.exit(1 if report.total_failed > 0 else 0)


if __name__ == "__main__":
    main()

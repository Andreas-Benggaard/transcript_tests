from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import BATCH_SIZE, FINBERT_BATCH_SIZE, TEMP_DIR
from .db.interface import TranscriptDB, TranscriptRef
from .downloader import cleanup_temp_dir, download_transcripts, load_transcript_json
from .pipeline import PipelineResult, run_sentiment_pipeline
from .run_state import read_last_run_date, write_last_run_date
from .sentiment.finbert import FinBERTAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchReport:
    total_processed: int
    total_failed: int
    failed_event_ids: tuple[int, ...]
    duration_seconds: float


def _chunks(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_batch_pipeline(
    db: TranscriptDB,
    state_file: Path | None = None,
    batch_size: int = BATCH_SIZE,
    upload_sentences: bool = False,
) -> BatchReport:
    """Run the full batch pipeline: fetch → download → analyze → upload.

    Parameters
    ----------
    db : Your TranscriptDB implementation (see src/db/stub.py).
    state_file : Path to last_run_date file. None = process everything.
    batch_size : Number of transcripts per batch (default 10).
    upload_sentences : If True, also upload sentence-level data to SQL.
    """
    start = time.monotonic()
    total_processed = 0
    total_failed = 0
    failed_ids: list[int] = []

    # 1. Load FinBERT once
    logger.info("Loading FinBERT model...")
    analyzer = FinBERTAnalyzer()

    # 2. Determine since_date
    since_date = read_last_run_date(state_file) if state_file else None

    # 3. Fetch transcript URLs from SQL
    refs = db.get_transcript_urls(since_date)
    if not refs:
        logger.info("No new transcripts to process")
        return BatchReport(0, 0, (), time.monotonic() - start)

    logger.info("Found %d transcripts to process", len(refs))

    # 4. Fetch prior quarter sentiments for QoQ delta
    company_ids = list({r.company_id for r in refs})
    try:
        prior_df = db.get_prior_event_sentiments(company_ids)
    except NotImplementedError:
        logger.info("get_prior_event_sentiments not implemented — skipping QoQ delta")
        prior_df = pd.DataFrame()

    # 5. Process in batches
    for batch_num, batch_refs in enumerate(_chunks(refs, batch_size), start=1):
        batch_event_ids = [r.event_id for r in batch_refs]
        logger.info(
            "Batch %d: processing event_ids %s", batch_num, batch_event_ids
        )

        temp_dir = TEMP_DIR / f"batch_{batch_num}"
        try:
            # Download
            dl_results = download_transcripts(batch_refs, temp_dir)

            # Load successful downloads
            transcripts: list[dict] = []
            for dl in dl_results:
                if dl.error is not None:
                    total_failed += 1
                    failed_ids.append(dl.ref.event_id)
                    continue

                data = load_transcript_json(dl.file_path)
                if data is None:
                    total_failed += 1
                    failed_ids.append(dl.ref.event_id)
                    continue

                data["date"] = dl.ref.event_date
                transcripts.append(data)

            if not transcripts:
                logger.warning("Batch %d: no valid transcripts — skipping", batch_num)
                continue

            # Run pipeline
            result = run_sentiment_pipeline(
                transcripts,
                batch_size=FINBERT_BATCH_SIZE,
                analyzer=analyzer,
                enrich=True,
                prior_events_df=prior_df if not prior_df.empty else None,
            )

            # Upload
            _upload_results(db, result, upload_sentences)
            total_processed += len(transcripts)

            logger.info(
                "Batch %d: processed %d, failed %d",
                batch_num, len(transcripts),
                len(batch_refs) - len(transcripts),
            )

        finally:
            cleanup_temp_dir(temp_dir)

    # 6. Update state only on success
    if state_file and total_processed > 0:
        write_last_run_date(state_file, date.today())

    elapsed = time.monotonic() - start
    report = BatchReport(
        total_processed=total_processed,
        total_failed=total_failed,
        failed_event_ids=tuple(failed_ids),
        duration_seconds=elapsed,
    )
    logger.info(
        "Pipeline complete — processed=%d, failed=%d, duration=%.1fs",
        report.total_processed, report.total_failed, report.duration_seconds,
    )
    return report


def _upload_results(
    db: TranscriptDB,
    result: PipelineResult,
    upload_sentences: bool,
) -> None:
    """Upload pipeline results to SQL."""
    if not result.events_by_section_df.empty:
        db.upload_event_sentiments(result.events_by_section_df)

    if upload_sentences and not result.sentences_df.empty:
        db.upload_sentence_sentiments(result.sentences_df)

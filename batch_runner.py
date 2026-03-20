import logging
import time
from dataclasses import dataclass

from config import BATCH_SIZE, FINBERT_BATCH_SIZE, MODEL_PATH, TEMP_DIR
from downloader import cleanup_temp_dir, download_transcripts, load_transcript_json
from pipeline import run_sentiment_pipeline
from sentiment import FinBERTAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchReport:
    total_processed: int
    total_failed: int
    failed_event_ids: tuple
    duration_seconds: float


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_batch_pipeline(
    db,
    since_date=None,
    batch_size=BATCH_SIZE,
    upload_sentences=False,
    limit=None,
):
    """Run the weekly batch pipeline: fetch → download → analyse → upload.

    Parameters
    ----------
    db             : MSSQLTranscriptDB instance.
    since_date     : Only process transcripts created after this date.
                     None = process everything not yet in sentiment_events.
    batch_size     : Number of transcripts per download/processing batch.
    upload_sentences : If True, also upload sentence-level data.
    limit          : Cap total transcripts fetched (useful for testing).
    """
    start = time.monotonic()
    total_processed = 0
    total_failed = 0
    failed_ids = []

    # 1. Load FinBERT once
    logger.info("Loading FinBERT model...")
    analyzer = FinBERTAnalyzer(MODEL_PATH)

    # 2. Fetch transcript URLs from SQL
    refs = db.get_transcript_urls(since_date, limit=limit)
    if not refs:
        logger.info("No new transcripts to process")
        return BatchReport(0, 0, (), time.monotonic() - start)

    logger.info("Found %d transcripts to process", len(refs))

    # 3. Fetch prior quarter sentiments for QoQ delta
    company_ids = list({r.company_id for r in refs})
    try:
        prior_df = db.get_prior_event_sentiments(company_ids)
        logger.info("Found %d prior event sentiments", len(prior_df))
    except NotImplementedError:
        logger.info("get_prior_event_sentiments not implemented — skipping QoQ delta")
        import pandas as pd
        prior_df = pd.DataFrame()

    # 4. Process in batches
    for batch_num, batch_refs in enumerate(_chunks(refs, batch_size), start=1):
        batch_event_ids = [r.event_id for r in batch_refs]
        logger.info("Batch %d: processing event_ids %s", batch_num, batch_event_ids)

        temp_dir = TEMP_DIR / f"batch_{batch_num}"
        try:
            dl_results = download_transcripts(batch_refs, temp_dir)

            transcripts = []
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

            result = run_sentiment_pipeline(
                transcripts,
                batch_size=FINBERT_BATCH_SIZE,
                analyzer=analyzer,
                enrich=True,
                prior_events_df=prior_df if not prior_df.empty else None,
            )

            _upload_results(db, result, upload_sentences)
            total_processed += len(transcripts)

            logger.info(
                "Batch %d: processed %d, failed %d",
                batch_num,
                len(transcripts),
                len(batch_refs) - len(transcripts),
            )

        finally:
            cleanup_temp_dir(temp_dir)

    elapsed = time.monotonic() - start
    report = BatchReport(
        total_processed=total_processed,
        total_failed=total_failed,
        failed_event_ids=tuple(failed_ids),
        duration_seconds=elapsed,
    )
    logger.info(
        "Pipeline complete — processed=%d, failed=%d, duration=%.1fs",
        report.total_processed,
        report.total_failed,
        report.duration_seconds,
    )
    return report


def run_sentence_backfill_pipeline(db, batch_size=BATCH_SIZE, limit=None):
    """Run the full sentence-level backfill: fetch all → download → analyse → upload.

    Differences from run_batch_pipeline():
    - Fetches ALL transcript refs (no since_date filter, no dedup check)
    - Always uploads sentence-level data (upload_sentences=True)
    - Skips QoQ delta enrichment (prior_events_df=None)
    - Uses 'backfill_batch_N' temp dirs to avoid collisions with weekly runs
    """
    start = time.monotonic()
    total_processed = 0
    total_failed = 0
    failed_ids = []

    # 1. Load FinBERT once
    logger.info("Loading FinBERT model...")
    analyzer = FinBERTAnalyzer(MODEL_PATH)

    # 2. Fetch all refs — no date filter, ASC order mandatory for QoQ correctness
    refs = db.get_all_transcript_urls(limit=limit)
    if not refs:
        logger.info("No transcripts found")
        return BatchReport(0, 0, (), time.monotonic() - start)

    logger.info("Found %d transcripts to process", len(refs))

    # 3. Process in batches
    for batch_num, batch_refs in enumerate(_chunks(refs, batch_size), start=1):
        batch_event_ids = [r.event_id for r in batch_refs]
        logger.info(
            "Backfill batch %d: processing event_ids %s", batch_num, batch_event_ids
        )

        temp_dir = TEMP_DIR / f"backfill_batch_{batch_num}"
        try:
            dl_results = download_transcripts(batch_refs, temp_dir)

            transcripts = []
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
                logger.warning(
                    "Backfill batch %d: no valid transcripts — skipping", batch_num
                )
                continue

            result = run_sentiment_pipeline(
                transcripts,
                batch_size=FINBERT_BATCH_SIZE,
                analyzer=analyzer,
                enrich=True,
                prior_events_df=None,  # No QoQ delta on backfill
            )

            _upload_results(db, result, upload_sentences=True)
            total_processed += len(transcripts)

            logger.info(
                "Backfill batch %d: processed %d, failed %d",
                batch_num,
                len(transcripts),
                len(batch_refs) - len(transcripts),
            )

        finally:
            cleanup_temp_dir(temp_dir)

    elapsed = time.monotonic() - start
    report = BatchReport(
        total_processed=total_processed,
        total_failed=total_failed,
        failed_event_ids=tuple(failed_ids),
        duration_seconds=elapsed,
    )
    logger.info(
        "Backfill complete — processed=%d, failed=%d, duration=%.1fs",
        report.total_processed,
        report.total_failed,
        report.duration_seconds,
    )
    if report.failed_event_ids:
        logger.warning("Failed event_ids: %s", report.failed_event_ids)

    return report


def _upload_results(db, result, upload_sentences):
    """Upload pipeline results to SQL."""
    if not result.events_by_section_df.empty:
        db.upload_event_sentiments(result.events_by_section_df)

    if upload_sentences and not result.sentences_df.empty:
        db.upload_sentence_sentiments(result.sentences_df)

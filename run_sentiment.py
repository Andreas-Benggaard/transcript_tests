import logging

from interface import MSSQLTranscriptDB
from batch_runner import run_sentence_backfill_pipeline

logger = logging.getLogger(__name__)


def run_sentiment_flow():
    """
    Entry point for the sentence-level sentiment backfill.
    Registered in the SQL job store — called via runApp().
    """
    logger.info('Starting sentence-level sentiment flow')

    db = MSSQLTranscriptDB()
    report = run_sentence_backfill_pipeline(db=db)

    logger.info(
        'Sentiment flow complete — processed: %d, failed: %d, duration: %.1fs',
        report.total_processed,
        report.total_failed,
        report.duration_seconds,
    )

    if report.failed_event_ids:
        logger.warning('Failed event_ids: %s', report.failed_event_ids)


def runApp():
    run_sentiment_flow()

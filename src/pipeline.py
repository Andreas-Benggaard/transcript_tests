"""
Earnings call sentiment analysis pipeline.

Usage
-----
from src.pipeline import run_sentiment_pipeline

transcripts = get_transcripts()
result = run_sentiment_pipeline(transcripts)
# result.sentences_df, result.paragraphs_df, result.events_df, result.events_by_section_df
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .enrichment.delta import compute_qoq_delta
from .enrichment.sections import detect_sections
from .enrichment.speakers import classify_speaker
from .parsers.detect import parse_transcript
from .sentiment.aggregator import (
    aggregate_events,
    aggregate_events_by_section,
    aggregate_paragraphs,
)
from .sentiment.finbert import FinBERTAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    sentences_df: pd.DataFrame
    paragraphs_df: pd.DataFrame
    events_df: pd.DataFrame
    events_by_section_df: pd.DataFrame


def run_sentiment_pipeline(
    transcripts: list[dict[str, Any]],
    batch_size: int = 32,
    analyzer: FinBERTAnalyzer | None = None,
    enrich: bool = True,
    prior_events_df: pd.DataFrame | None = None,
) -> PipelineResult:
    """Run FinBERT sentiment analysis on a list of transcript dicts.

    Parameters
    ----------
    transcripts:
        Each dict must have ``event_id``, ``company_id``, ``date``, and the
        standard ``transcript`` sub-object with ``paragraphs``.
    batch_size:
        Number of sentences fed to FinBERT at once. Reduce if you hit OOM.
    analyzer:
        Optional pre-loaded FinBERTAnalyzer (reuse across batches).
    enrich:
        If True, classify speakers and detect sections.
    prior_events_df:
        DataFrame with [company_id, net_sentiment] for QoQ delta.
    """
    empty = PipelineResult(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    if not transcripts:
        return empty

    if analyzer is None:
        analyzer = FinBERTAnalyzer()

    all_records: list[dict] = []

    for transcript in transcripts:
        event_id = transcript["event_id"]
        transcript_date = transcript.get("date")

        logger.info("Parsing event_id=%s", event_id)
        sentence_records = parse_transcript(transcript)

        if not sentence_records:
            logger.warning("event_id=%s yielded no sentences — skipping", event_id)
            continue

        # --- Enrichment: speaker classification ---
        speaker_types = [
            classify_speaker(rec.speaker_role, rec.speaker_company)
            if enrich else "unknown"
            for rec in sentence_records
        ]

        # --- Enrichment: section detection ---
        if enrich:
            paragraph_indices = [rec.paragraph_index for rec in sentence_records]
            sections = detect_sections(speaker_types, paragraph_indices)
        else:
            sections = ["unknown"] * len(sentence_records)

        for rec, stype, section in zip(sentence_records, speaker_types, sections):
            all_records.append(
                {
                    "event_id": rec.event_id,
                    "company_id": rec.company_id,
                    "date": transcript_date,
                    "paragraph_index": rec.paragraph_index,
                    "sentence_index": rec.sentence_index,
                    "speaker_id": rec.speaker_id,
                    "speaker_name": rec.speaker_name,
                    "speaker_role": rec.speaker_role,
                    "speaker_company": rec.speaker_company,
                    "speaker_type": stype,
                    "section": section,
                    "text": rec.text,
                    "start_time": rec.start_time,
                    "end_time": rec.end_time,
                }
            )

    if not all_records:
        return empty

    df = pd.DataFrame(all_records)

    # --- FinBERT inference ---
    logger.info("Running FinBERT on %d sentences", len(df))
    texts = df["text"].tolist()
    results = analyzer.predict_batch(texts, batch_size=batch_size)

    sentiment_df = pd.DataFrame(
        {
            "sentiment_label": [r.label for r in results],
            "sentiment_positive": [r.positive for r in results],
            "sentiment_negative": [r.negative for r in results],
            "sentiment_neutral": [r.neutral for r in results],
        }
    )

    sentences_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)

    # --- Aggregation ---
    paragraphs_df = aggregate_paragraphs(sentences_df)
    events_df = aggregate_events(sentences_df)
    events_by_section_df = aggregate_events_by_section(sentences_df)

    # --- QoQ delta ---
    if prior_events_df is not None:
        events_df = compute_qoq_delta(events_df, prior_events_df)
        events_by_section_df = compute_qoq_delta(
            events_by_section_df, prior_events_df
        )

    logger.info(
        "Done — %d events, %d paragraphs, %d sentences",
        len(events_df),
        len(paragraphs_df),
        len(sentences_df),
    )

    return PipelineResult(
        sentences_df=sentences_df,
        paragraphs_df=paragraphs_df,
        events_df=events_df,
        events_by_section_df=events_by_section_df,
    )

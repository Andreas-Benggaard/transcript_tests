from __future__ import annotations

import pandas as pd

from ..enrichment.threshold import threshold_labels


def _mode_label(labels: pd.Series) -> str:
    return labels.mode().iloc[0]


def _speaker_net_sentiment(
    sentence_df: pd.DataFrame,
    speaker_type: str,
    groupby_cols: list[str],
) -> pd.Series:
    """Compute net_sentiment for a specific speaker_type within each group."""
    filtered = sentence_df[sentence_df["speaker_type"] == speaker_type]
    if filtered.empty:
        return pd.Series(dtype=float)
    grouped = filtered.groupby(groupby_cols, dropna=False)
    return grouped.apply(
        lambda g: g["sentiment_positive"].mean() - g["sentiment_negative"].mean()
    )


def aggregate_paragraphs(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level sentiment to paragraph level.

    Returns a new DataFrame — input is never mutated.
    """
    group_cols = [
        "event_id", "company_id", "date", "paragraph_index", "speaker_id",
        "speaker_name", "speaker_role", "speaker_company", "speaker_type", "section",
    ]
    # Only group by columns that exist
    group_cols = [c for c in group_cols if c in sentence_df.columns]

    grouped = sentence_df.groupby(group_cols, dropna=False)
    agg = (
        grouped
        .agg(
            sentence_count=("text", "count"),
            sentiment_label=("sentiment_label", _mode_label),
            sentiment_positive=("sentiment_positive", "mean"),
            sentiment_negative=("sentiment_negative", "mean"),
            sentiment_neutral=("sentiment_neutral", "mean"),
        )
        .reset_index()
    )
    agg = agg.assign(
        net_sentiment=agg["sentiment_positive"] - agg["sentiment_negative"]
    )
    agg = agg.assign(sentiment_label=threshold_labels(agg["net_sentiment"]))
    return agg


def aggregate_events(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level sentiment to event (full transcript) level.

    Returns a new DataFrame — input is never mutated.
    """
    grouped = sentence_df.groupby(["event_id", "company_id", "date"], dropna=False)
    agg = (
        grouped
        .agg(
            sentence_count=("text", "count"),
            sentiment_positive=("sentiment_positive", "mean"),
            sentiment_negative=("sentiment_negative", "mean"),
            sentiment_neutral=("sentiment_neutral", "mean"),
        )
        .reset_index()
    )
    agg = agg.assign(
        net_sentiment=agg["sentiment_positive"] - agg["sentiment_negative"],
        section="all",
    )
    agg = agg.assign(sentiment_label=threshold_labels(agg["net_sentiment"]))

    # Add management / analyst net sentiment
    mgmt_net = _speaker_net_sentiment(
        sentence_df, "management", ["event_id", "company_id", "date"]
    )
    analyst_net = _speaker_net_sentiment(
        sentence_df, "analyst", ["event_id", "company_id", "date"]
    )

    if not mgmt_net.empty:
        mgmt_df = mgmt_net.reset_index(name="management_net_sentiment")
        agg = agg.merge(mgmt_df, on=["event_id", "company_id", "date"], how="left")
    else:
        agg = agg.assign(management_net_sentiment=float("nan"))

    if not analyst_net.empty:
        analyst_df = analyst_net.reset_index(name="analyst_net_sentiment")
        agg = agg.merge(analyst_df, on=["event_id", "company_id", "date"], how="left")
    else:
        agg = agg.assign(analyst_net_sentiment=float("nan"))

    return agg


def aggregate_events_by_section(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment per event per section (prepared_remarks / q_and_a).

    Returns rows for each section plus the 'all' rollup from aggregate_events.
    Input is never mutated.
    """
    # Per-section aggregation
    if "section" not in sentence_df.columns:
        return aggregate_events(sentence_df)

    grouped = sentence_df.groupby(
        ["event_id", "company_id", "date", "section"], dropna=False
    )
    by_section = (
        grouped
        .agg(
            sentence_count=("text", "count"),
            sentiment_positive=("sentiment_positive", "mean"),
            sentiment_negative=("sentiment_negative", "mean"),
            sentiment_neutral=("sentiment_neutral", "mean"),
        )
        .reset_index()
    )
    by_section = by_section.assign(
        net_sentiment=by_section["sentiment_positive"] - by_section["sentiment_negative"]
    )
    by_section = by_section.assign(
        sentiment_label=threshold_labels(by_section["net_sentiment"])
    )

    # Combine with the 'all' rollup
    all_events = aggregate_events(sentence_df)
    return pd.concat([all_events, by_section], ignore_index=True)

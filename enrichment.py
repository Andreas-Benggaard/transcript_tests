import pandas as pd

from config import (
    ANALYST_KEYWORDS,
    MANAGEMENT_KEYWORDS,
    MODERATOR_KEYWORDS,
    NET_SENTIMENT_NEGATIVE_THRESHOLD,
    NET_SENTIMENT_POSITIVE_THRESHOLD,
)


def compute_qoq_delta(current_events_df, prior_events_df):
    """Add QoQ sentiment delta columns to current events.

    Parameters
    ----------
    current_events_df : must have columns [company_id, net_sentiment]
    prior_events_df   : must have columns [company_id, net_sentiment]
                        (one row per company — the most recent prior quarter)

    Returns a new DataFrame with two added columns:
        net_sentiment_prior_qtr, net_sentiment_delta_qoq
    Input is never mutated.
    """
    if prior_events_df.empty:
        return current_events_df.assign(
            net_sentiment_prior_qtr=float("nan"),
            net_sentiment_delta_qoq=float("nan"),
        )

    prior = prior_events_df[["company_id", "net_sentiment"]].rename(
        columns={"net_sentiment": "net_sentiment_prior_qtr"}
    )
    merged = current_events_df.merge(prior, on="company_id", how="left")
    merged = merged.assign(
        net_sentiment_delta_qoq=(
            merged["net_sentiment"] - merged["net_sentiment_prior_qtr"]
        )
    )
    return merged


def detect_sections(speaker_types, paragraph_indices):
    """Label each row as 'prepared_remarks' or 'q_and_a'.

    The boundary is the first paragraph where speaker_type == 'analyst'.
    Everything before that paragraph is prepared remarks.
    If no analyst is found, all rows are labelled 'unknown'.

    Parameters
    ----------
    speaker_types     : per-row speaker type ('management', 'analyst', etc.)
    paragraph_indices : per-row paragraph index from the transcript

    Returns
    -------
    List of section labels, same length as inputs.
    """
    if not speaker_types:
        return []

    # Find the paragraph index where the first analyst appears
    boundary_para = None
    for stype, pidx in zip(speaker_types, paragraph_indices):
        if stype == "analyst":
            boundary_para = pidx
            break

    if boundary_para is None:
        return ["unknown"] * len(speaker_types)

    return [
        "prepared_remarks" if pidx < boundary_para else "q_and_a"
        for pidx in paragraph_indices
    ]


def classify_speaker(speaker_role, speaker_company=None):
    """Classify a speaker as management / analyst / moderator / unknown.

    Uses keyword matching on the role string. Falls back to 'unknown'
    when no role is available.
    """
    if speaker_role is None:
        return "unknown"

    role_lower = speaker_role.lower()

    if _matches_any(role_lower, MODERATOR_KEYWORDS):
        return "moderator"

    # Check analyst BEFORE management — bank analysts often have titles
    # like "VP - Equity Research" where "vp" would false-match management.
    if _matches_any(role_lower, ANALYST_KEYWORDS):
        return "analyst"

    if _matches_any(role_lower, MANAGEMENT_KEYWORDS):
        return "management"

    return "unknown"


def _matches_any(text, keywords):
    return any(kw in text for kw in keywords)


def threshold_label(
    net_sentiment,
    pos_threshold=NET_SENTIMENT_POSITIVE_THRESHOLD,
    neg_threshold=NET_SENTIMENT_NEGATIVE_THRESHOLD,
):
    """Classify a single net_sentiment value."""
    if net_sentiment > pos_threshold:
        return "positive"
    if net_sentiment < neg_threshold:
        return "negative"
    return "neutral"


def threshold_labels(
    series,
    pos_threshold=NET_SENTIMENT_POSITIVE_THRESHOLD,
    neg_threshold=NET_SENTIMENT_NEGATIVE_THRESHOLD,
):
    """Vectorized threshold labelling for a DataFrame column."""
    return series.apply(lambda x: threshold_label(x, pos_threshold, neg_threshold))

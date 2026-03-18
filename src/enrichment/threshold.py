from __future__ import annotations

import pandas as pd

from ..config import (
    NET_SENTIMENT_NEGATIVE_THRESHOLD,
    NET_SENTIMENT_POSITIVE_THRESHOLD,
)


def threshold_label(
    net_sentiment: float,
    pos_threshold: float = NET_SENTIMENT_POSITIVE_THRESHOLD,
    neg_threshold: float = NET_SENTIMENT_NEGATIVE_THRESHOLD,
) -> str:
    """Classify a single net_sentiment value."""
    if net_sentiment > pos_threshold:
        return "positive"
    if net_sentiment < neg_threshold:
        return "negative"
    return "neutral"


def threshold_labels(
    series: pd.Series,
    pos_threshold: float = NET_SENTIMENT_POSITIVE_THRESHOLD,
    neg_threshold: float = NET_SENTIMENT_NEGATIVE_THRESHOLD,
) -> pd.Series:
    """Vectorized threshold labeling for a DataFrame column."""
    return series.apply(
        lambda x: threshold_label(x, pos_threshold, neg_threshold)
    )

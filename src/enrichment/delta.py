from __future__ import annotations

import pandas as pd


def compute_qoq_delta(
    current_events_df: pd.DataFrame,
    prior_events_df: pd.DataFrame,
) -> pd.DataFrame:
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

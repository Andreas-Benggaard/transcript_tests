import logging
from dataclasses import dataclass

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from enrichment import threshold_labels

logger = logging.getLogger(__name__)

MAX_TOKENS = 512
LABEL_ORDER = ["positive", "negative", "neutral"]


@dataclass(frozen=True)
class SentimentResult:
    label: str
    positive: float
    negative: float
    neutral: float


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_sentiment_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


class FinBERTAnalyzer:
    def __init__(self, model_path):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading FinBERT on %s", self._device)
        self._tokenizer, self._model = load_sentiment_model(model_path)
        self._model.to(self._device)
        self._model.eval()

        # FinBERT label order from config: positive=0, negative=1, neutral=2
        id2label = self._model.config.id2label
        self._idx_positive = next(k for k, v in id2label.items() if v == "positive")
        self._idx_negative = next(k for k, v in id2label.items() if v == "negative")
        self._idx_neutral = next(k for k, v in id2label.items() if v == "neutral")

    def predict_batch(self, texts, batch_size=32):
        """Run FinBERT on a list of texts. Returns one SentimentResult per text."""
        results = []

        for batch in _batches(texts, batch_size):
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )

            # Log any text that was likely truncated
            for text in batch:
                tokens = self._tokenizer.encode(text, add_special_tokens=True)
                if len(tokens) >= MAX_TOKENS:
                    logger.warning(
                        "Text truncated to %d tokens: %s...", MAX_TOKENS, text[:80]
                    )

            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

            for row in probs:
                pos = row[self._idx_positive]
                neg = row[self._idx_negative]
                neu = row[self._idx_neutral]
                label = LABEL_ORDER[[pos, neg, neu].index(max(pos, neg, neu))]
                results.append(
                    SentimentResult(label=label, positive=pos, negative=neg, neutral=neu)
                )

        return results


def _mode_label(labels):
    return labels.mode().iloc[0]


def _speaker_net_sentiment(sentence_df, speaker_type, groupby_cols):
    """Compute net_sentiment for a specific speaker type within each group."""
    filtered = sentence_df[sentence_df["speaker_type"] == speaker_type]
    if filtered.empty:
        return pd.Series(dtype=float)
    return filtered.groupby(groupby_cols, dropna=False).apply(
        lambda g: g["sentiment_positive"].mean() - g["sentiment_negative"].mean()
    )


def aggregate_paragraphs(sentence_df):
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


def aggregate_events(sentence_df):
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
        sentence_df, "management", groupby_cols=["event_id", "company_id", "date"]
    )
    analyst_net = _speaker_net_sentiment(
        sentence_df, "analyst", groupby_cols=["event_id", "company_id", "date"]
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


def aggregate_events_by_section(sentence_df):
    """Aggregate sentiment per event per section (prepared_remarks / q_and_a).

    Returns rows for each section plus the 'all' rollup from aggregate_events.
    Input is never mutated.
    """
    if "section" not in sentence_df.columns:
        return sentence_df

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

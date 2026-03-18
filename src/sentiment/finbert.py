from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512
LABEL_ORDER = ["positive", "negative", "neutral"]


@dataclass(frozen=True)
class SentimentResult:
    label: str
    positive: float
    negative: float
    neutral: float


def _batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class FinBERTAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading FinBERT on %s", self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        # FinBERT label order from config: positive=0, negative=1, neutral=2
        id2label: dict = self._model.config.id2label
        self._idx_positive = next(k for k, v in id2label.items() if v == "positive")
        self._idx_negative = next(k for k, v in id2label.items() if v == "negative")
        self._idx_neutral = next(k for k, v in id2label.items() if v == "neutral")

    def predict_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[SentimentResult]:
        """Run FinBERT on a list of texts. Returns one SentimentResult per text."""
        results: list[SentimentResult] = []

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
                label = LABEL_ORDER[
                    [pos, neg, neu].index(max(pos, neg, neu))
                ]
                results.append(
                    SentimentResult(
                        label=label, positive=pos, negative=neg, neutral=neu
                    )
                )

        return results

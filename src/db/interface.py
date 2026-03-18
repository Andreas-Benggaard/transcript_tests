from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class TranscriptRef:
    """A reference to a transcript in the source database."""

    event_id: int
    company_id: int
    event_date: date
    url: str


class TranscriptDB(Protocol):
    """Protocol your MSSQL implementation must satisfy.

    Implement all four methods in src/db/stub.py.
    """

    def get_transcript_urls(self, since_date: date | None) -> list[TranscriptRef]:
        """Fetch transcript URLs added after since_date.

        If since_date is None, return all transcripts (full backfill).
        """
        ...

    def upload_event_sentiments(self, df: pd.DataFrame) -> None:
        """Upsert event-level sentiment rows to SQL.

        Use MERGE / INSERT ON CONFLICT so re-runs are idempotent.
        Composite key: (event_id, section).
        """
        ...

    def upload_sentence_sentiments(self, df: pd.DataFrame) -> None:
        """Upsert sentence-level sentiment rows to SQL.

        Composite key: (event_id, paragraph_index, sentence_index).
        """
        ...

    def get_prior_event_sentiments(
        self, company_ids: list[int]
    ) -> pd.DataFrame:
        """Fetch the most recent prior net_sentiment per company.

        Returns DataFrame with columns: company_id, net_sentiment.
        Only the latest row per company_id (section='all').
        """
        ...

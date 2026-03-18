"""MSSQL implementation using the internal Database lib.

Import path — adjust to wherever your lib lives:
    from local_libs.database import Database as DB   # <-- change this line
"""

from __future__ import annotations

import secrets
from datetime import date

import pandas as pd

# TODO: update this import to match your local path
from local_libs.database import Database as DB  # type: ignore

from .interface import TranscriptDB, TranscriptRef

# The MSSQL database name passed to every DB call
_TARGET = "your_database_name"  # TODO: set this


class MSSQLTranscriptDB:
    """Reads and writes transcript/sentiment data via the internal DB lib."""

    def get_transcript_urls(self, since_date: date | None) -> list[TranscriptRef]:
        if since_date is not None:
            query = f"""
                SELECT event_id, company_id, event_date, url
                FROM transcripts
                WHERE createdOn > '{since_date}'
                ORDER BY createdOn ASC
            """
        else:
            query = """
                SELECT event_id, company_id, event_date, url
                FROM transcripts
                ORDER BY createdOn ASC
            """

        df = DB.getGenericMssql(query, _TARGET)

        # Convert each row into a TranscriptRef
        records = df[["event_id", "company_id", "event_date", "url"]].to_dict("records")
        return [TranscriptRef(**r) for r in records]

    def upload_event_sentiments(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["id"] = [secrets.token_hex(8) for _ in range(len(df))]
        DB.upload(df, _TARGET)

    def upload_sentence_sentiments(self, df: pd.DataFrame) -> None:
        # DB.upload requires a unique ID column — add one using a random hex string
        df = df.copy()
        df["id"] = [secrets.token_hex(8) for _ in range(len(df))]
        DB.upload(df, _TARGET)

    def get_prior_event_sentiments(self, company_ids: list[int]) -> pd.DataFrame:
        ids = ", ".join(str(i) for i in company_ids)
        query = f"""
            SELECT company_id, net_sentiment
            FROM (
                SELECT company_id, net_sentiment,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_id
                           ORDER BY event_date DESC
                       ) AS rn
                FROM sentiment_events
                WHERE section = 'all'
                  AND company_id IN ({ids})
            ) sub
            WHERE rn = 1
        """
        return DB.getGenericMssql(query, _TARGET)

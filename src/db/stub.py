"""Stub implementation of TranscriptDB.

Replace the NotImplementedError bodies with your MSSQL code.
Example connection pattern:

    import pyodbc
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=your_server;"
        "DATABASE=your_db;"
        "UID=your_user;"
        "PWD=your_password;"
    )
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .interface import TranscriptDB, TranscriptRef


class MSSQLTranscriptDB:
    """Your MSSQL implementation — fill in the four methods below."""

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        # Example:
        # import pyodbc
        # self._conn = pyodbc.connect(connection_string)

    def get_transcript_urls(self, since_date: date | None) -> list[TranscriptRef]:
        """
        YOUR CODE HERE.

        Example SQL:
            SELECT event_id, company_id, event_date, url
            FROM transcripts
            WHERE createdOn > ?
            ORDER BY createdOn ASC

        Return:
            [TranscriptRef(event_id=..., company_id=..., event_date=..., url=...), ...]
        """
        raise NotImplementedError("Implement with your MSSQL query")

    def upload_event_sentiments(self, df: pd.DataFrame) -> None:
        """
        YOUR CODE HERE.

        Example SQL (MERGE for upsert):
            MERGE INTO sentiment_events AS target
            USING (VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)) AS source
                  (event_id, company_id, event_date, section, sentence_count,
                   sentiment_positive, sentiment_negative, sentiment_neutral,
                   net_sentiment, sentiment_label,
                   management_net_sentiment, analyst_net_sentiment,
                   net_sentiment_prior_qtr, net_sentiment_delta_qoq,
                   processed_at)
            ON target.event_id = source.event_id AND target.section = source.section
            WHEN MATCHED THEN UPDATE SET ...
            WHEN NOT MATCHED THEN INSERT ...;
        """
        raise NotImplementedError("Implement with your MSSQL upsert")

    def upload_sentence_sentiments(self, df: pd.DataFrame) -> None:
        """
        YOUR CODE HERE.

        Upsert to sentiment_sentences table.
        Key: (event_id, paragraph_index, sentence_index).
        """
        raise NotImplementedError("Implement with your MSSQL upsert")

    def get_prior_event_sentiments(
        self, company_ids: list[int]
    ) -> pd.DataFrame:
        """
        YOUR CODE HERE.

        Example SQL:
            SELECT company_id, net_sentiment
            FROM (
                SELECT company_id, net_sentiment,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_id
                           ORDER BY event_date DESC
                       ) AS rn
                FROM sentiment_events
                WHERE section = 'all'
                  AND company_id IN (?, ?, ...)
            ) sub
            WHERE rn = 1

        Return: DataFrame with columns [company_id, net_sentiment]
        """
        raise NotImplementedError("Implement with your MSSQL query")

import logging

import pandas as pd

from Utilities import Database as DB  # type: ignore

_FACTSET_TARGET = "FactSet"
_EDI_TARGET = "EDI"

logger = logging.getLogger(__name__)


class TranscriptRef:
    """A reference to a transcript in the source database."""

    def __init__(self, event_id, company_id, event_date, url, doc_id):
        self.event_id = event_id
        self.company_id = company_id
        self.event_date = event_date
        self.url = url
        self.doc_id = doc_id  # The 'id' from quartz_documents — base for docid


class MSSQLTranscriptDB:
    """Reads and writes transcript/sentiment data via the internal DB lib."""

    def get_transcript_urls(self, since_date=None, limit=None):
        """Fetch transcript URLs that haven't been processed yet.

        Returns transcripts from FactSet that aren't already in sentiment_events.
        """
        base_query = """
            SELECT id,
                   companyId,
                   eventId,
                   fileUrl,
                   createdAt
            FROM [FACTSET_2].[dbo].[quartz_documents]
            WHERE fileType = 'Transcript'
            AND [documentDescription] = 'In-house transcript'
            AND fileUrl IS NOT NULL
        """

        if since_date is not None:
            base_query += f" AND createdAt > '{since_date}'"
            logger.info("Fetching transcript URLs for %s", base_query)

        if limit is not None:
            base_query = (
                f"SELECT TOP {limit} * FROM ({base_query}) sub ORDER BY createdAt ASC"
            )
            logger.info("Limit checking %s", base_query)

        scripts = DB.getGenericMSQL(base_query, target=_FACTSET_TARGET)

        # Get already-processed documents from the events table
        completed_query = """
            SELECT DISTINCT LEFT(docid, CHARINDEX('_', docid + '_') - 1) AS base_docid
            FROM [EDI].[raw].[sentiment_events]
        """
        completed = DB.getGenericMSQL(completed_query, target=_EDI_TARGET)

        # Normalise types to strings + strip
        scripts = scripts.copy()
        completed = completed.copy()
        scripts["id"] = scripts["id"].astype(str).str.strip()
        completed["base_docid"] = completed["base_docid"].astype(str).str.strip()

        # Build set for faster membership check
        completed_ids = set(completed["base_docid"].dropna())

        # Filter out already-processed documents
        scripts_filtered = scripts[~scripts["id"].isin(completed_ids)]

        records = []
        for _, row in scripts_filtered.iterrows():
            records.append(
                TranscriptRef(
                    event_id=row["eventId"],
                    company_id=row["companyId"],
                    event_date=row["createdAt"],
                    url=row["fileUrl"],
                    doc_id=row["id"],
                )
            )

        return records

    def get_all_transcript_urls(self, limit=None):
        """Fetch ALL transcript refs with no date filter.

        Identical to get_transcript_urls() but without the since_date WHERE
        clause and without the already-processed deduplication check.
        ORDER BY createdAt ASC is intentional — the backfill must process
        oldest events first so that compute_qoq_delta() finds prior-quarter
        rows already uploaded when it reaches event N.
        """
        base_query = """
            SELECT id,
                   companyId,
                   eventId,
                   fileUrl,
                   createdAt
            FROM [FACTSET_2].[dbo].[quartz_documents]
            WHERE fileType = 'Transcript'
            AND [documentDescription] = 'In-house transcript'
            AND fileUrl IS NOT NULL
            ORDER BY createdAt ASC
        """

        if limit is not None:
            base_query = (
                f"SELECT TOP {limit} * FROM ({base_query}) sub ORDER BY createdAt ASC"
            )
            logger.info("Limit checking %s", base_query)

        scripts = DB.getGenericMSQL(base_query, target=_FACTSET_TARGET)

        records = []
        for _, row in scripts.iterrows():
            records.append(
                TranscriptRef(
                    event_id=row["eventId"],
                    company_id=row["companyId"],
                    event_date=row["createdAt"],
                    url=row["fileUrl"],
                    doc_id=row["id"],
                )
            )

        return records

    def upload_event_sentiments(self, df):
        """Upload event-level sentiment rows to sentiment_events table."""
        df = df.copy()

        # Create composite docid: event_id + '_' + section
        # Ensures uniqueness while maintaining traceability
        df["docid"] = df["event_id"].astype(str) + "_" + df["section"]

        upload_df = pd.DataFrame({
            "docid": df["docid"],
            "event_id": df["event_id"],
            "company_id": df["company_id"],
            "event_date": df["date"],
            "section": df["section"],
            "sentence_count": df["sentence_count"],
            "sentiment_label": df["sentiment_label"],
            "sentiment_positive": df["sentiment_positive"],
            "sentiment_negative": df["sentiment_negative"],
            "sentiment_neutral": df["sentiment_neutral"],
            "net_sentiment": df["net_sentiment"],
            "management_net_sentiment": df.get("management_net_sentiment"),
            "analyst_net_sentiment": df.get("analyst_net_sentiment"),
            "net_sentiment_prior_qtr": df.get("net_sentiment_prior_qtr"),
            "net_sentiment_delta_qoq": df.get("net_sentiment_delta_qoq"),
        })

        # Remove duplicates on docid
        upload_df = upload_df.drop_duplicates(subset=["docid"], keep="last")

        DB.upload_msql(
            df=upload_df,
            schema="raw",
            table="sentiment_events",
            db="EDI",
            replace=0,
            id_col_name="docid",
        )

    def upload_sentence_sentiments(self, df):
        """Upload sentence-level sentiment rows to sentiment_sentences table."""
        df = df.copy()

        # Composite unique key for sentences
        df["sentence_id"] = (
            df["event_id"].astype(str)
            + "_"
            + df["paragraph_index"].astype(str)
            + "_"
            + df["sentence_index"].astype(str)
        )

        # Link back to the 'all' section row in sentiment_events
        df["docid"] = df["event_id"].astype(str) + "_all"

        upload_df = pd.DataFrame({
            "sentence_id": df["sentence_id"],
            "docid": df["docid"],
            "event_id": df["event_id"],
            "company_id": df["company_id"],
            "paragraph_index": df["paragraph_index"],
            "sentence_index": df["sentence_index"],
            "section": df["section"],
            "speaker_name": df["speaker_name"],
            "speaker_type": df["speaker_type"],
            "text": df["text"],
            "sentiment_label": df["sentiment_label"],
            "sentiment_positive": df["sentiment_positive"],
            "sentiment_negative": df["sentiment_negative"],
            "sentiment_neutral": df["sentiment_neutral"],
            "net_sentiment": df["sentiment_positive"] - df["sentiment_negative"],
        })

        # Remove duplicates
        upload_df = upload_df.drop_duplicates(subset=["sentence_id"], keep="last")

        DB.upload_msql(
            df=upload_df,
            schema="raw",
            table="sentiment_sentences",
            db="EDI",
            replace=0,
            id_col_name="sentence_id",
        )

    def get_prior_event_sentiments(self, company_ids):
        """Fetch the most recent prior net_sentiment per company."""
        ids = ", ".join(str(i) for i in company_ids)
        query = f"""
            SELECT company_id, net_sentiment
            FROM (
                SELECT company_id, net_sentiment,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_id
                           ORDER BY event_date DESC
                       ) AS rn
                FROM [EDI].[raw].[sentiment_events]
                WHERE section = 'all'
                AND company_id IN ({ids})
            ) sub
            WHERE rn = 1
        """
        return DB.getGenericMSQL(query, target=_EDI_TARGET)


# ─────────────────────────────────────────────────────────────────────────────
# DDL (run once in SSMS against EDI before first pipeline run)
# ─────────────────────────────────────────────────────────────────────────────
#
# CREATE TABLE [EDI].[raw].[sentiment_events] (
#     docid                    VARCHAR(50)   NOT NULL PRIMARY KEY,
#     event_id                 INT           NOT NULL,
#     company_id               INT,
#     event_date               DATE          NOT NULL,
#     section                  VARCHAR(20),  -- all / prepared_remarks / q_and_a / unknown
#     sentence_count           INT,
#     sentiment_label          VARCHAR(10),  -- positive / negative / neutral
#     sentiment_positive       FLOAT,
#     sentiment_negative       FLOAT,
#     sentiment_neutral        FLOAT,
#     net_sentiment            FLOAT,
#     management_net_sentiment FLOAT,
#     analyst_net_sentiment    FLOAT,
#     net_sentiment_prior_qtr  FLOAT,
#     net_sentiment_delta_qoq  FLOAT,
#     processed_at             DATETIME      DEFAULT GETDATE()
# );
#
# CREATE TABLE [EDI].[raw].[sentiment_sentences] (
#     sentence_id        VARCHAR(100)  NOT NULL PRIMARY KEY,
#     docid              VARCHAR(50)   NOT NULL,
#     event_id           INT           NOT NULL,
#     company_id         INT,
#     paragraph_index    INT           NOT NULL,
#     sentence_index     INT           NOT NULL,
#     section            VARCHAR(20),
#     speaker_name       NVARCHAR(200),
#     speaker_type       VARCHAR(20),
#     text               NVARCHAR(MAX),
#     sentiment_label    VARCHAR(10),
#     sentiment_positive FLOAT,
#     sentiment_negative FLOAT,
#     sentiment_neutral  FLOAT,
#     net_sentiment      FLOAT,
#     processed_at       DATETIME      DEFAULT GETDATE()
# );

# Earnings Call Sentiment Pipeline — Setup Guide

## What this does

Runs FinBERT sentiment analysis on earnings call transcripts from Nordic companies.
Produces sentiment scores at sentence, paragraph, and event level — including a
prepared remarks vs Q&A split and management vs analyst breakdown.

Runs weekly. Only processes transcripts added since the last run.

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- ODBC Driver 18 for SQL Server installed on the machine running the pipeline
- Access to your Microsoft SQL Server database

---

## 1. Install dependencies

```bash
cd transcripts
uv sync
```

For MSSQL connectivity, also install pyodbc:

```bash
uv add pyodbc
```

---

## 2. Implement your SQL code

Open `src/db/stub.py`. There are 4 methods to implement. Each has the expected
SQL and return type in the docstring.

### `get_transcript_urls(since_date)`

Fetches transcript URLs added after `since_date`. Called at the start of every run.

```python
# Example SQL
SELECT event_id, company_id, event_date, url
FROM transcripts
WHERE createdOn > ?
ORDER BY createdOn ASC
```

Return a list of `TranscriptRef` objects:

```python
return [
    TranscriptRef(
        event_id=row.event_id,
        company_id=row.company_id,
        event_date=row.event_date,
        url=row.url,
    )
    for row in cursor.fetchall()
]
```

### `upload_event_sentiments(df)`

Upserts the event-level results to SQL. Must be an upsert (not insert) so
re-runs don't create duplicates. Composite key: `(event_id, section)`.

The DataFrame has these columns:

| Column | Type | Notes |
|--------|------|-------|
| `event_id` | int | |
| `company_id` | int | |
| `date` | date | |
| `section` | str | `all`, `prepared_remarks`, `q_and_a`, or `unknown` |
| `sentiment_label` | str | `positive`, `negative`, `neutral` |
| `net_sentiment` | float | positive − negative (main indicator) |
| `management_net_sentiment` | float | nullable |
| `analyst_net_sentiment` | float | nullable |
| `net_sentiment_delta_qoq` | float | nullable — requires prior data |
| `net_sentiment_prior_qtr` | float | nullable |
| `sentiment_positive` | float | |
| `sentiment_negative` | float | |
| `sentiment_neutral` | float | |
| `sentence_count` | int | |

```sql
-- Example MERGE (MSSQL upsert)
MERGE INTO sentiment_events AS target
USING (VALUES (?, ?, ...)) AS source (event_id, section, ...)
ON target.event_id = source.event_id AND target.section = source.section
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...;
```

### `upload_sentence_sentiments(df)` *(optional)*

Only called if you pass `--upload-sentences`. Upserts sentence-level data.
Key: `(event_id, paragraph_index, sentence_index)`. Skip if you don't need
drill-down data in SQL.

### `get_prior_event_sentiments(company_ids)`

Fetches the most recent `net_sentiment` per company for QoQ delta calculation.
Returns a DataFrame with columns `[company_id, net_sentiment]`.

```sql
-- One row per company, most recent call only
SELECT company_id, net_sentiment
FROM (
    SELECT company_id, net_sentiment,
           ROW_NUMBER() OVER (
               PARTITION BY company_id ORDER BY event_date DESC
           ) AS rn
    FROM sentiment_events
    WHERE section = 'all'
      AND company_id IN (?, ?, ...)
) sub
WHERE rn = 1
```

If you haven't run the pipeline before, return an empty DataFrame:

```python
return pd.DataFrame(columns=["company_id", "net_sentiment"])
```

---

## 3. Create the output table in SQL Server

Run this once in your database before the first pipeline run:

```sql
CREATE TABLE sentiment_events (
    event_id                  INT            NOT NULL,
    company_id                INT            NOT NULL,
    event_date                DATE           NOT NULL,
    section                   VARCHAR(20)    NOT NULL,  -- all / prepared_remarks / q_and_a / unknown
    sentence_count            INT,
    sentiment_label           VARCHAR(10),              -- positive / negative / neutral
    sentiment_positive        FLOAT,
    sentiment_negative        FLOAT,
    sentiment_neutral         FLOAT,
    net_sentiment             FLOAT,
    management_net_sentiment  FLOAT,
    analyst_net_sentiment     FLOAT,
    net_sentiment_prior_qtr   FLOAT,
    net_sentiment_delta_qoq   FLOAT,
    processed_at              DATETIME       DEFAULT GETDATE(),
    CONSTRAINT PK_sentiment_events PRIMARY KEY (event_id, section)
);

-- Optional: sentence-level table (only if using --upload-sentences)
CREATE TABLE sentiment_sentences (
    event_id          INT             NOT NULL,
    company_id        INT,
    paragraph_index   INT             NOT NULL,
    sentence_index    INT             NOT NULL,
    section           VARCHAR(20),
    speaker_name      NVARCHAR(200),
    speaker_type      VARCHAR(20),
    text              NVARCHAR(MAX),
    sentiment_label   VARCHAR(10),
    sentiment_positive FLOAT,
    sentiment_negative FLOAT,
    sentiment_neutral  FLOAT,
    net_sentiment      FLOAT,
    processed_at       DATETIME        DEFAULT GETDATE(),
    CONSTRAINT PK_sentiment_sentences PRIMARY KEY (event_id, paragraph_index, sentence_index)
);
```

---

## 4. Wire up your DB class in run_weekly.py

Open `run_weekly.py`. Find line 22:

```python
from src.db.stub import MSSQLTranscriptDB
```

If you create your own class name, update this import. Then find line 43:

```python
db = MSSQLTranscriptDB(connection_string=args.connection_string)
```

Update to match however your class initialises the connection.

---

## 5. Test with the example transcripts

Before running against real data, verify the pipeline works end to end:

```bash
uv run python test_examples.py
```

This runs the two example JSONs locally (no SQL needed) and saves:
- `sentiment_events.csv` — event-level results
- `sentiment_events_by_section.csv` — prepared remarks vs Q&A split

---

## 6. Run the pipeline

**Incremental run** (normal weekly use — only processes new transcripts):

```bash
uv run python run_weekly.py --connection-string "DRIVER={ODBC Driver 18 for SQL Server};SERVER=...;DATABASE=...;UID=...;PWD=..."
```

**Full backfill** (first run, or to reprocess everything):

```bash
uv run python run_weekly.py --full-backfill --connection-string "..."
```

**Include sentence-level data in SQL:**

```bash
uv run python run_weekly.py --upload-sentences --connection-string "..."
```

---

## 7. Schedule as a weekly job

Add to Windows Task Scheduler or a cron job to run every Saturday morning.

**cron example** (Linux/macOS):

```
0 6 * * 6 cd /path/to/transcripts && uv run python run_weekly.py --connection-string "..." >> pipeline.log 2>&1
```

**Windows Task Scheduler**: point the action to `uv.exe` with arguments:
```
run python run_weekly.py --connection-string "..."
```

The pipeline tracks the last run date in `~/.transcripts_pipeline_last_run`.
Each run only processes transcripts added since that date.

---

## Tuning

All thresholds and settings are in `src/config.py`:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `BATCH_SIZE` | 10 | Transcripts downloaded at a time |
| `FINBERT_BATCH_SIZE` | 32 | Sentences fed to FinBERT at once — reduce if you hit memory errors |
| `NET_SENTIMENT_POSITIVE_THRESHOLD` | 0.05 | net_sentiment above this → `positive` label |
| `NET_SENTIMENT_NEGATIVE_THRESHOLD` | -0.05 | net_sentiment below this → `negative` label |
| `DOWNLOAD_TIMEOUT_SECONDS` | 30 | Per-URL timeout |

---

## Interpreting the output

**`net_sentiment`** = `sentiment_positive − sentiment_negative`
- Range roughly -1 to +1
- Positive = more positive language than negative
- Neutral language (most of earnings calls) scores near 0

**`sentiment_label`** = threshold-based from net_sentiment
- `positive` if net_sentiment > 0.05
- `negative` if net_sentiment < -0.05
- `neutral` otherwise

**`section`**
- `prepared_remarks` — scripted management monologue before Q&A
- `q_and_a` — everything from the first analyst question onward
- `unknown` — transcripts without speaker names (no section split possible)

**Key indicators for report writing**
- `net_sentiment_delta_qoq` — direction of change quarter over quarter
- `analyst_net_sentiment` vs `management_net_sentiment` gap — tension in the call
- Q&A section more negative than prepared remarks — management under pressure

---

## Troubleshooting

**FinBERT OOM error** — reduce `FINBERT_BATCH_SIZE` in `src/config.py`

**Dead URLs / download failures** — logged to `pipeline_run.log`, skipped.
Failed `event_id`s are in the `BatchReport`. Re-running will retry them.

**Wrong speaker classifications** — add keywords to `MANAGEMENT_KEYWORDS`,
`ANALYST_KEYWORDS`, or `MODERATOR_KEYWORDS` in `src/config.py`

**Section always `unknown`** — transcript is in timestamped format (no
`speaker_mapping`). Section detection requires named speakers.

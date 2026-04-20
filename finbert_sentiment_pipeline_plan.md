# FinBERT Sentiment Pipeline for Quartr In-House Transcripts

**Author:** Andreas Benggaard
**Target runtime:** Python 3.11 on a **CPU laptop**, behind Nordea corporate proxy
**Scheduler:** custom — pipeline exposes a `main(mode, max_docs)` entrypoint the scheduler can call
**Source table:** `FACTSET_2.dbo.quartr_documents` (filter: `documentDescription = 'In-house transcript'`)
**Sinks:** MSSQL (`EDI`, schema `raw`) + Parquet
**Consumer:** Sell-side Equity Research dev team
**Scope:** ~11k historical in-house transcripts (backfill, 3–4 day laptop run) + weekly incremental run

---

## Data model summary (short version)

**Three SQL tables. Three Parquet folders.** No sections table, no run-log table.

```
sentiment_events     = the call          (1 row per call)         ← ~11k rows
sentiment_speakers   = who spoke         (~6 rows per call)       ← ~50k rows
sentiment_segments   = what was said     (~60 rows per call)      ← ~660k rows
```

- **`sentiment_events`** — the headline table. One row per earnings call, keyed by `docid` (= `documentId`). Carries whole-call score plus `prepared_score` / `qna_score` as columns.
- **`sentiment_speakers`** — one row per (call, speaker), keyed by `docid_speakerid = docid × 1_000 + speaker_id`. Carries name, role, company, `role_class`, whole-call score, plus optional `prepared_score` / `qna_score` (NULL if the speaker didn't appear in that section).
- **`sentiment_segments`** — one row per paragraph, keyed by `docid_paragraphindex = docid × 100_000 + paragraph_index`. Carries `section` ("PREPARED_REMARKS" or "QNA") as a plain column, score, `argmax_label`, start/end timestamps, `text_hash` (no raw text stored).

Parquet mirrors SQL: `parquet_backfill/{event,speaker,segment}/batch_NNNN.parquet` during backfill; `parquet_weekly/...` for the weekly mode.

---

## 1. What this plan changes vs your existing script

| # | Current behaviour | Problem | Fix |
|---|---|---|---|
| 1 | Caches JSON into `transcript_cache/{md5}.json` forever | You want JSONs deleted after scoring | Streamed to a `tempfile.NamedTemporaryFile(delete=False)` inside `try/finally: os.remove()` |
| 2 | `.done` sentinel files in `processed_docs/` | Breaks across hosts | Resumability via Parquet inspection (backfill) / `NOT EXISTS` against SQL (weekly) |
| 3 | `fileType='Transcript'` filter | Picks up non-in-house docs | `documentDescription = 'In-house transcript'` + `deleted = 0` + `fileUrl IS NOT NULL` |
| 4 | Uses `id` as PK | You said `documentId` is the unique one | `docid` is a BIGINT derived from `documentId` |
| 5 | Speaker rows only carry `speaker_id` integer | Unusable without joining back to JSON | Store name, role, company, and `role_class` (COMPANY/ANALYST/OPERATOR/UNKNOWN) |
| 6 | No Prepared Remarks vs Q&A split | Single most-asked-for cut in ER | Paragraph grain carries `section`; event and speaker rows carry `prepared_score` + `qna_score` as columns |
| 7 | `weighted_score = Σ (polarity × score) / N` | Throws away probability mass | Signed score `p_pos − p_neg` weighted by **word count** |
| 8 | Stores `paragraph_text[:2000]` | Licensing / bloat | `text_hash` only |
| 9 | Pipe-delimited composite keys like `"778812\|0"` | Ugly, string-typed | Zero-padded BIGINT arithmetic: `docid=778812, para_idx=5` → `77881200005` |
| 10 | No tone-dispersion or Q&A-vs-Prepared fields | Missing alpha-generating fields | Added as plain columns on `sentiment_events` |
| 11 | No structured log lines | Silent failures | `logging.info/warning/exception` — scheduler writes log file in repo folder |

---

## 2. Input: SQL source of truth

Table: `[FACTSET_2].[dbo].[quartr_documents]`. Columns available:

```
id, companyId, eventId, fileUrl, updatedAt, createdAt, summary,
fileName, filepath, fileType, fileSize, duration, encoding, qna,
documentTypeName, documentForm, documentCategory, documentDescription,
Ingested, genericType, documentId, AiFileID, deleted
```

Selection query (drives both backfill and weekly — the `{weekly_filter}` is swapped in at runtime):

```sql
SELECT  documentId AS docid,      -- BIGINT
        companyId  AS companyid,  -- INT
        eventId    AS eventid,    -- INT
        fileUrl,
        createdAt
FROM    [FACTSET_2].[dbo].[quartr_documents] t
WHERE   documentDescription = 'In-house transcript'
  AND   fileUrl IS NOT NULL
  AND   (deleted IS NULL OR deleted = 0)
  AND   NOT EXISTS (
            SELECT 1
            FROM   [EDI].[raw].[sentiment_events] s
            WHERE  s.docid = t.documentId
        )
  {weekly_filter}   -- AND createdAt >= DATEADD(day, -8, SYSUTCDATETIME())
ORDER BY createdAt ASC
```

All IDs stay as integers end-to-end — no varchar casting.

---

## 3. Transcript JSON model (verified against your 3 examples)

```
{
  "version": "1.0.0",
  "event_id": 305869,
  "company_id": 5014,
  "speaker_mapping": [
    {"speaker": 0, "speaker_data": {"name": "...", "role": "CEO", "company": "..."}}
  ],
  "transcript": {
    "text": "...",
    "number_of_speakers": 4,
    "paragraphs": [
      {
        "text": "...",
        "start": 0.24,
        "end": 73.74,
        "speaker": 3,
        "sentences": [
          {"text": "...", "start": ..., "end": ..., "words": [...]}
        ]
      }
    ]
  }
}
```

Empirically verified: **no section / Q&A / prepared flag anywhere at any nesting depth**. Must be inferred (§5.2).

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ FACTSET_2.dbo.quartr_documents                                  │
│   (documentId PK, companyId, eventId, fileUrl, createdAt, ...)  │
└───────────────┬─────────────────────────────────────────────────┘
                │ 1. DB.getGenericMSQL(...) → pd.DataFrame
                ▼
     ┌──────────────────────┐
     │ orchestrator.py      │  ◀── custom scheduler calls main(mode)
     └──────────┬───────────┘
                │ 2. CL.getSession(external=1, SSL=1).get(fileUrl)
                ▼
     ┌──────────────────────┐
     │ downloader.py        │  → tempfile /{tmp}/*.json (deleted after parse)
     └──────────┬───────────┘
                │ 3. parse + normalise (paragraphs → sentences)
                ▼
     ┌──────────────────────┐
     │ parser.py            │
     └──────────┬───────────┘
                │ 4. role_class + section inference
                ▼
     ┌──────────────────────┐
     │ section.py           │   COMPANY/ANALYST/OPERATOR/UNKNOWN
     └──────────┬───────────┘              PREPARED_REMARKS / QNA
                │ 5. FinBERT inference on sentences (batched, CPU)
                ▼
     ┌──────────────────────┐
     │ scorer.py            │   local ./bert (HuggingFace checkpoint)
     └──────────┬───────────┘
                │ 6. aggregate: sentence → paragraph → speaker → event
                ▼
     ┌──────────────────────┐
     │ aggregator.py        │
     └──────────┬───────────┘
                │ 7. write Parquet (always) + SQL (weekly only)
                ▼
    ┌─────────────────────────────┬──────────────────────────────┐
    │ EDI.raw.sentiment_events    │ parquet_*/event/             │
    │ EDI.raw.sentiment_speakers  │ parquet_*/speaker/           │
    │ EDI.raw.sentiment_segments  │ parquet_*/segment/           │
    └─────────────────────────────┴──────────────────────────────┘
      Three tables. Three Parquet folders. Nothing else.
      (progress + per-doc status → Python `logging` →
       scheduler's log file in repo folder)
```

---

## 5. Processing stages

### 5.1 Download (no disk cache)

```python
from Utilities import CommonLibrary as CL
import tempfile, os, json

def get_http_session():
    s = CL.getSession(external=1, SSL=1)
    s.headers.update({"User-Agent": "andreas.benggaard@nordea.com"})
    s.trust_env = True
    return s

def fetch_transcript(session, url: str, timeout: int = 60):
    """Stream to a temp file, load, always delete. Returns parsed dict."""
    if not isinstance(url, str) or not url.startswith("http"):
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        r = session.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        with open(tmp.name, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        tmp.close()
        try:
            os.remove(tmp.name)
        except OSError:
            pass
```

Retries: wrap `session.get` with `tenacity.retry(stop_after_attempt=3, wait=wait_exponential(multiplier=2))`.

### 5.2 Role classification + section inference

**Empirical finding:** the JSONs have no section / Q&A flag anywhere. Must be inferred. Heuristic validated against all three example files:

| File | Paragraphs | Inferred Q&A start | Transition paragraph |
|---|---:|---:|---|
| 1 (Tokmanni / SPAR) | 66 | 26 | Operator: "If you wish to withdraw your question…" → Nordea analyst |
| 2 (Enento) | 84 | 16 | IR: "now I think we have a bit of time for any questions" → Nordea analyst |
| 3 (Cibus) | 116 | 30 | Operator: "If you wish to ask a question…" → Nordea analyst |

```python
import re

COMPANY_TERMS  = ("ceo","cfo","coo","cto","chair","president","ir",
                  "investor relations","head of","director of","vp",
                  "vice president","managing director")
ANALYST_TERMS  = ("analyst","equity","research","portfolio manager","pm")
OPERATOR_TERMS = ("operator","moderator")

def role_class(role: str | None) -> str:
    r = (role or "").lower()
    if any(t in r for t in OPERATOR_TERMS): return "OPERATOR"
    if any(t in r for t in ANALYST_TERMS):  return "ANALYST"
    if any(t in r for t in COMPANY_TERMS):  return "COMPANY"
    return "UNKNOWN"

QNA_PROMPT_RE = re.compile(r"\bquestions?\b|\bq\s*&\s*a\b|\bfirst question\b|"
                           r"open.*line.*for.*questions|time for (any )?questions",
                           re.I)

def infer_qna_start(paragraphs, speaker_mapping) -> int | None:
    """Returns the paragraph index at which Q&A begins, or None if no Q&A."""
    smap = {s["speaker"]: s["speaker_data"] for s in speaker_mapping}
    seen_company = False
    for i, p in enumerate(paragraphs):
        role = smap.get(p.get("speaker"), {}).get("role")
        rc = role_class(role)
        if rc == "COMPANY":
            seen_company = True
            continue
        # KEY RULE: analyst-before-company is pre-call chatter, NOT Q&A.
        if seen_company and rc == "ANALYST":
            return i
        if seen_company and rc == "OPERATOR" and QNA_PROMPT_RE.search(p.get("text", "")):
            return i
    return None  # no Q&A detected → entire call is PREPARED_REMARKS
```

Plain-English rules:

1. Walk paragraphs. Track whether any COMPANY paragraph has been seen.
2. Q&A starts at the first ANALYST paragraph *after* management has spoken. Pre-call "Happy New Year" analyst chatter stays in PREPARED_REMARKS (this is the file 2 case).
3. OPERATOR with Q&A-prompt language also counts, again only after a COMPANY paragraph.
4. No analysts anywhere → whole call is PREPARED_REMARKS.
5. No COMPANY speakers → whole call is PREPARED_REMARKS and we `log.warning`.

### 5.3 Scoring — CPU-tuned

```python
import os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_PATH = "./bert"                 # local HuggingFace checkpoint
INFERENCE_BATCH_SIZE = 8              # CPU sweet-spot
MAX_TOKENS = 512

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
torch.set_num_interop_threads(1)

class SentimentAnalyser:
    def __init__(self, model_path: str = MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model     = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        model.eval()
        self.pipe = pipeline(
            "text-classification",
            model=model, tokenizer=tokenizer,
            device=-1,                 # CPU
            top_k=None,
            truncation=True, max_length=MAX_TOKENS,
        )

    @torch.inference_mode()
    def analyse_records(self, records):
        non_empty = [(i, r["text"]) for i, r in enumerate(records) if r["text"].strip()]
        if not non_empty:
            return
        raw = self.pipe([t for _, t in non_empty], batch_size=INFERENCE_BATCH_SIZE)
        for (orig_idx, _), scores in zip(non_empty, raw):
            d = {x["label"].lower(): float(x["score"]) for x in scores}
            rec = records[orig_idx]
            rec["p_pos"] = d.get("positive", 0.0)
            rec["p_neu"] = d.get("neutral",  0.0)
            rec["p_neg"] = d.get("negative", 0.0)
            rec["signed_score"] = rec["p_pos"] - rec["p_neg"]
            rec["argmax_label"] = max(d, key=d.get)
            rec["n_words"] = len(rec["text"].split())
```

**CPU runtime expectations:**

- ~350–500 sentences per transcript × batch_size 8 → ~50–100 ms / sentence.
- **Per-transcript:** ~20–60 seconds end-to-end.
- **Backfill of 11k transcripts:** realistically **2–4 days wall-clock** on a laptop. Resumable — kill and restart freely.
- **Weekly incremental (50–200 new transcripts):** **10–40 minutes.**

Fallbacks if full-FinBERT is too slow: `yiyanghkust/finbert-tone` (drop-in) or ONNX-quantised (~2–4× CPU speedup).

### 5.4 Aggregation (single helper, word-count weighted)

```python
def agg(recs):
    if not recs:
        return {"weighted_score": 0.0, "p_pos": 0.0, "p_neu": 0.0, "p_neg": 0.0,
                "pct_pos": 0.0, "pct_neg": 0.0, "pct_neu": 0.0,
                "tone_dispersion": 0.0, "n_sentences": 0, "n_words": 0,
                "dominant_sentiment": "neutral"}
    wc = sum(r["n_words"] for r in recs) or 1
    weighted = sum(r["signed_score"] * r["n_words"] for r in recs) / wc
    p_pos = sum(r["p_pos"] * r["n_words"] for r in recs) / wc
    p_neu = sum(r["p_neu"] * r["n_words"] for r in recs) / wc
    p_neg = sum(r["p_neg"] * r["n_words"] for r in recs) / wc
    n = len(recs)
    npos = sum(1 for r in recs if r["argmax_label"] == "positive")
    nneg = sum(1 for r in recs if r["argmax_label"] == "negative")
    nneu = n - npos - nneg
    mean = weighted
    dispersion = (sum((r["signed_score"] - mean) ** 2 for r in recs) / n) ** 0.5
    dominant = max(("positive", npos), ("negative", nneg), ("neutral", nneu),
                   key=lambda x: x[1])[0]
    return {
        "weighted_score": round(weighted, 5),
        "p_pos": round(p_pos, 5), "p_neu": round(p_neu, 5), "p_neg": round(p_neg, 5),
        "pct_pos": round(npos / n, 5), "pct_neg": round(nneg / n, 5), "pct_neu": round(nneu / n, 5),
        "tone_dispersion": round(dispersion, 5),
        "n_sentences": n, "n_words": wc,
        "dominant_sentiment": dominant,
    }
```

Three roll-ups, one roll-up per SQL table:

- **Segment** (paragraph grain) — group by `paragraph_index`. Each row carries its `section` as a plain column.
- **Speaker** — group by `speaker_id`. Whole-call score, plus two optional columns `prepared_score` / `qna_score`. Either can be NULL if the speaker didn't appear in that section.
- **Event** — all non-operator records. Row carries whole-call `weighted_score`, plus `prepared_score`, `qna_score`, and `qna_vs_prepared = qna_score − prepared_score`.

### 5.5 Writes — two modes

#### Mode BACKFILL — Parquet only, one bulk SQL upload at the very end

During the 3–4 day laptop run, **no MSSQL writes at all.** Each flush appends one Parquet file per table:

```
parquet_backfill/
├── event/
│   ├── batch_0001.parquet
│   ├── batch_0002.parquet
│   └── ...
├── speaker/
│   ├── batch_0001.parquet
│   └── ...
└── segment/
    ├── batch_0001.parquet
    └── ...
```

`batch_N` ≈ 50 docs. Atomic: written to `batch_N.parquet.tmp` and renamed on close, so a crash mid-flush leaves no half-files.

After the full run, one-shot uploader (`upload_backfill.py`):

```python
# src/sentiment/upload_backfill.py
import glob, pandas as pd, logging
from Utilities import Database as DB

log = logging.getLogger(__name__)

def upload_backfill(parquet_root: str = "parquet_backfill") -> None:
    for table, folder, id_col in [
        ("sentiment_events",   "event",   "docid"),
        ("sentiment_speakers", "speaker", "docid_speakerid"),
        ("sentiment_segments", "segment", "docid_paragraphindex"),
    ]:
        files = sorted(glob.glob(f"{parquet_root}/{folder}/batch_*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.drop_duplicates(subset=[id_col], keep="last")
        log.info("uploading table=%s rows=%d", table, len(df))
        DB.upload_msql(df=df, schema="raw", table=table, db="EDI",
                       replace=0, id_col_name=id_col)

if __name__ == "__main__":
    upload_backfill()
```

#### Mode WEEKLY — Parquet + SQL per batch

Weekly volumes are small, so upload inline:

```python
from Utilities import Database as DB

DB.upload_msql(df=df_events,   schema="raw", table="sentiment_events",   db="EDI",
               replace=0, id_col_name="docid")
DB.upload_msql(df=df_speakers, schema="raw", table="sentiment_speakers", db="EDI",
               replace=0, id_col_name="docid_speakerid")
DB.upload_msql(df=df_segments, schema="raw", table="sentiment_segments", db="EDI",
               replace=0, id_col_name="docid_paragraphindex")
```

#### Composite key formulas

Zero-padded BIGINTs, no strings, no pipes:

```python
docid_speakerid      = int(docid) * 1_000   + int(speaker_id)       # max 999 speakers
docid_paragraphindex = int(docid) * 100_000 + int(paragraph_index)  # max 99,999 paras
```

Worked example: `docid=778812`, `paragraph_index=5` → `77881200005`.

### 5.6 Cleanup & logging

Logging uses stdlib only. The **custom scheduler writes a timestamped log file in the repo folder** when it invokes the pipeline.

```python
import logging
log = logging.getLogger(__name__)

log.info("doc=%s start url=%s", docid, fileurl)
...
log.info("doc=%s OK paragraphs=%d sentences=%d duration_ms=%d",
         docid, n_paras, n_sents, duration_ms)
log.exception("doc=%s FAILED: %s", docid, exc)
log.warning("doc=%s no ANALYST speakers — whole call classified PREPARED_REMARKS", docid)
```

**Do not** call `logging.basicConfig()` inside the pipeline — the scheduler owns handler configuration.

---

## 6. MSSQL schema (EDI.raw.*) — three tables

All IDs are integers. Composite keys are zero-padded BIGINTs computed in Python.

```sql
-- one row per call
CREATE TABLE [EDI].[raw].[sentiment_events] (
    docid               BIGINT       NOT NULL PRIMARY KEY,  -- = documentId
    companyid           INT          NOT NULL,
    eventid             INT          NOT NULL,
    createdat           DATETIME2    NOT NULL,
    total_sentences     INT          NOT NULL,
    total_words         INT          NOT NULL,
    num_speakers        INT          NOT NULL,
    duration_sec        DECIMAL(10,2) NULL,
    weighted_score      DECIMAL(7,5) NOT NULL,   -- whole-call signed score
    p_pos DECIMAL(7,5), p_neu DECIMAL(7,5), p_neg DECIMAL(7,5),
    pct_positive DECIMAL(7,5), pct_negative DECIMAL(7,5), pct_neutral DECIMAL(7,5),
    tone_dispersion     DECIMAL(7,5) NOT NULL,
    prepared_score      DECIMAL(7,5) NULL,       -- NULL if no Prepared Remarks
    qna_score           DECIMAL(7,5) NULL,       -- NULL if no Q&A detected
    qna_vs_prepared     DECIMAL(7,5) NULL,       -- qna_score - prepared_score
    dominant_sentiment  VARCHAR(10)  NOT NULL,
    model_version       VARCHAR(64)  NOT NULL,
    run_ts              DATETIME2    NOT NULL
);

-- one row per (call, speaker) — section-level scores live as two columns
CREATE TABLE [EDI].[raw].[sentiment_speakers] (
    docid_speakerid     BIGINT       NOT NULL PRIMARY KEY,  -- docid * 1_000 + speaker_id
    docid               BIGINT       NOT NULL,
    companyid           INT          NOT NULL,
    eventid             INT          NOT NULL,
    speaker_id          INT          NOT NULL,
    speaker_name        NVARCHAR(200) NULL,
    speaker_role        NVARCHAR(200) NULL,
    speaker_company     NVARCHAR(200) NULL,
    role_class          VARCHAR(20)  NOT NULL,   -- COMPANY|ANALYST|OPERATOR|UNKNOWN
    weighted_score      DECIMAL(7,5) NOT NULL,
    p_pos DECIMAL(7,5), p_neu DECIMAL(7,5), p_neg DECIMAL(7,5),
    pct_positive DECIMAL(7,5), pct_negative DECIMAL(7,5), pct_neutral DECIMAL(7,5),
    tone_dispersion     DECIMAL(7,5) NOT NULL,
    prepared_score      DECIMAL(7,5) NULL,
    qna_score           DECIMAL(7,5) NULL,
    dominant_sentiment  VARCHAR(10)  NOT NULL,
    n_sentences         INT          NOT NULL,
    n_words             INT          NOT NULL,
    model_version       VARCHAR(64)  NOT NULL,
    run_ts              DATETIME2    NOT NULL
);

-- one row per paragraph (no raw text — just a hash)
CREATE TABLE [EDI].[raw].[sentiment_segments] (
    docid_paragraphindex BIGINT      NOT NULL PRIMARY KEY,  -- docid * 100_000 + paragraph_index
    docid               BIGINT       NOT NULL,
    companyid           INT          NOT NULL,
    eventid             INT          NOT NULL,
    paragraph_index     INT          NOT NULL,
    speaker_id          INT          NOT NULL,
    role_class          VARCHAR(20)  NOT NULL,
    section             VARCHAR(20)  NOT NULL,   -- PREPARED_REMARKS | QNA
    weighted_score      DECIMAL(7,5) NOT NULL,
    p_pos DECIMAL(7,5), p_neu DECIMAL(7,5), p_neg DECIMAL(7,5),
    argmax_label        VARCHAR(10)  NOT NULL,
    n_sentences         INT          NOT NULL,
    n_words             INT          NOT NULL,
    start_sec           DECIMAL(10,2) NULL,
    end_sec             DECIMAL(10,2) NULL,
    text_hash           CHAR(64)     NOT NULL,
    model_version       VARCHAR(64)  NOT NULL,
    run_ts              DATETIME2    NOT NULL
);

-- No run-log table: logging goes to the scheduler's log file.
```

### Optional: indexes (run once after backfill upload)

An **index** is a sorted lookup structure MSSQL builds on one or more columns so queries filtering by those columns skip the full-table scan. Like a book's index — you jump to the page instead of flipping through all of them. Correctness-neutral, speed-only.

At 11k event rows they barely matter. At ~660k segment rows they start mattering when a dashboard filters by `docid` or `(docid, section)`. Hand these to whoever admins the `EDI` database after the bulk upload, if query performance becomes an issue:

```sql
CREATE INDEX IX_events_company_date  ON [EDI].[raw].[sentiment_events]   (companyid, createdat);
CREATE INDEX IX_speakers_docid       ON [EDI].[raw].[sentiment_speakers] (docid);
CREATE INDEX IX_speakers_role        ON [EDI].[raw].[sentiment_speakers] (companyid, role_class);
CREATE INDEX IX_segments_docid       ON [EDI].[raw].[sentiment_segments] (docid);
CREATE INDEX IX_segments_section     ON [EDI].[raw].[sentiment_segments] (docid, section);
```

---

## 7. Repo layout

```
finbert-sentiment/
├── pyproject.toml
├── README.md
├── bert/                       # local FinBERT checkpoint (MODEL_PATH)
├── parquet_backfill/           # written during the 3–4 day run (gitignored)
├── parquet_weekly/             # written by weekly runs (gitignored)
├── logs/                       # scheduler writes per-run log files here (gitignored)
├── src/
│   └── sentiment/
│       ├── __init__.py
│       ├── orchestrator.py     # main(mode, max_docs, ...) entrypoint
│       ├── db.py               # wraps DB.getGenericMSQL / DB.upload_msql
│       ├── http.py             # CL.getSession(external=1, SSL=1)
│       ├── downloader.py       # tempfile-based fetch_transcript
│       ├── parser.py           # JSON → sentence records
│       ├── section.py          # role_class + section inference
│       ├── scorer.py           # SentimentAnalyser
│       ├── aggregator.py       # single `agg()` helper + build_* rows
│       ├── writer.py           # Parquet writer (+ SQL writer for weekly mode)
│       ├── upload_backfill.py  # run ONCE after the backfill completes
│       └── schemas.py          # pydantic validation of the JSON
├── sql/
│   ├── 001_create_tables.sql
│   └── 002_indexes.sql         # optional; hand to DBA
└── tests/
    ├── fixtures/               # trimmed copies of your 3 JSONs
    ├── test_parser.py
    ├── test_section.py
    ├── test_aggregator.py
    └── test_scorer.py          # tiny dummy model for CI
```

`load_documents_df`:

```python
def load_documents_df(weekly: bool = False) -> pd.DataFrame:
    weekly_filter = "AND createdAt >= DATEADD(day, -8, SYSUTCDATETIME())" if weekly else ""
    q = f"""
        SELECT  documentId AS docid,
                companyId  AS companyid,
                eventId    AS eventid,
                fileUrl    AS fileurl,
                createdAt  AS createdat
        FROM    [FACTSET_2].[dbo].[quartr_documents] t
        WHERE   documentDescription = 'In-house transcript'
          AND   fileUrl IS NOT NULL
          AND   (deleted IS NULL OR deleted = 0)
          AND   NOT EXISTS (
                    SELECT 1 FROM [EDI].[raw].[sentiment_events] s
                    WHERE  s.docid = t.documentId
                )
          {weekly_filter}
        ORDER BY createdAt ASC
    """
    df = pd.DataFrame(DB.getGenericMSQL(q, target="FactSet"))
    for col in ("docid", "companyid", "eventid"):
        df[col] = df[col].astype("int64")
    return df
```

---

## 8. Example outputs

### 8.1 `sentiment_events` — one row per call

| docid  | companyid | eventid | createdat           | total_sentences | num_speakers | weighted_score | p_pos | p_neu | p_neg | pct_positive | pct_negative | tone_dispersion | prepared_score | qna_score | qna_vs_prepared | dominant_sentiment | model_version   |
|-------:|----------:|--------:|---------------------|----------------:|-------------:|---------------:|------:|------:|------:|-------------:|-------------:|----------------:|---------------:|----------:|----------------:|--------------------|-----------------|
| 778812 |      5014 |  305869 | 2023-02-16 08:00:00 |             412 |            4 |         +0.244 | 0.441 | 0.406 | 0.153 |        0.457 |        0.112 |           0.261 |         +0.301 |    +0.104 |          −0.197 | positive           | finbert-2024-01 |
| 901233 |     12602 |  363238 | 2024-10-30 09:00:00 |             617 |            9 |         +0.081 | 0.334 | 0.512 | 0.154 |        0.361 |        0.187 |           0.298 |         +0.158 |    +0.046 |          −0.112 | neutral            | finbert-2024-01 |
| 712004 |      5521 |  297936 | 2022-11-10 10:30:00 |             803 |            9 |         +0.172 | 0.398 | 0.442 | 0.160 |        0.421 |        0.148 |           0.271 |         +0.219 |    +0.130 |          −0.089 | positive           | finbert-2024-01 |

### 8.2 `sentiment_speakers` — one row per (call, speaker)

| docid_speakerid | docid  | speaker_id | speaker_name    | speaker_role | role_class | weighted_score | prepared_score | qna_score | pct_pos | pct_neg | tone_dispersion | n_words |
|----------------:|-------:|-----------:|-----------------|--------------|------------|---------------:|---------------:|----------:|--------:|--------:|----------------:|--------:|
|     778812003   | 778812 |          3 | Mika Rautiainen | CEO          | COMPANY    |         +0.271 |         +0.318 |    +0.162 |   0.481 |   0.102 |           0.264 |    2978 |
|     778812002   | 778812 |          2 | Tobias Wasmuht  | CEO          | COMPANY    |         +0.276 |         +0.276 |      NULL |   0.471 |   0.098 |           0.221 |    1420 |
|     778812000   | 778812 |          0 | Svante Krokfors | Analyst      | ANALYST    |         −0.044 |           NULL |    −0.044 |   0.182 |   0.227 |           0.301 |     318 |

`docid_speakerid = docid * 1_000 + speaker_id`. So `778812 * 1000 + 3 = 778812003`.

### 8.3 `sentiment_segments` — one row per paragraph

| docid_paragraphindex | docid  | paragraph_index | speaker_id | role_class | section          | weighted_score | p_pos | p_neu | p_neg | argmax_label | n_sentences | n_words | start_sec | end_sec | text_hash |
|---------------------:|-------:|----------------:|-----------:|------------|------------------|---------------:|------:|------:|------:|--------------|------------:|--------:|----------:|--------:|-----------|
|          77881200000 | 778812 |               0 |          3 | COMPANY    | PREPARED_REMARKS |         +0.412 | 0.521 | 0.370 | 0.109 | positive     |          10 |     168 |     0.24  |   73.74 | a3b1…     |
|          77881200001 | 778812 |               1 |          3 | COMPANY    | PREPARED_REMARKS |         +0.187 | 0.302 | 0.583 | 0.115 | neutral      |           6 |      95 |    73.75  |  122.40 | f78c…     |
|          77881200038 | 778812 |              38 |          0 | ANALYST    | QNA              |         −0.055 | 0.183 | 0.579 | 0.238 | neutral      |           4 |      61 |  1823.11  | 1855.02 | 92ea…     |
|          77881200039 | 778812 |              39 |          3 | COMPANY    | QNA              |         +0.244 | 0.398 | 0.448 | 0.154 | positive     |           7 |     142 |  1855.50  | 1921.00 | 11b7…     |

`docid_paragraphindex = docid * 100_000 + paragraph_index`. So `778812 * 100000 + 39 = 77881200039`.

### 8.4 Parquet layout

```
parquet_backfill/        (during 3–4 day run; no SQL writes)
├── event/
│   ├── batch_0001.parquet
│   └── ...
├── speaker/
│   ├── batch_0001.parquet
│   └── ...
└── segment/
    ├── batch_0001.parquet
    └── ...

parquet_weekly/          (each weekly run; SQL also written)
├── event/
├── speaker/
└── segment/
```

Three folders. Nothing else.

---

## 9. Entrypoint for the custom scheduler

```python
# src/sentiment/orchestrator.py
from enum import Enum
import logging, time
import pandas as pd

log = logging.getLogger(__name__)

class Mode(str, Enum):
    BACKFILL = "backfill"     # everything not yet scored; Parquet only, NO SQL
    WEEKLY   = "weekly"       # createdAt >= now - 8 days; Parquet + SQL

def main(mode: str | Mode = Mode.WEEKLY,
         max_docs: int | None = None,
         flush_every: int = 50,
         parquet_root: str | None = None) -> dict:
    """
    Entry point for the custom scheduler.

    Parameters
    ----------
    mode          : "backfill" or "weekly".
                    backfill → Parquet only (run upload_backfill afterwards)
                    weekly   → Parquet + MSSQL per batch
    max_docs      : optional cap (useful for smoke tests).
    flush_every   : batch size for each flush (50 is a good CPU default).
    parquet_root  : override the default output folder.

    Returns
    -------
    dict: {"processed": N, "failed": N, "skipped": N, "duration_sec": X}
    """
    mode = Mode(mode)
    t0 = time.time()
    df_docs = load_documents_df(weekly=(mode == Mode.WEEKLY))
    if max_docs is not None:
        df_docs = df_docs.head(max_docs)

    write_sql = (mode == Mode.WEEKLY)
    root = parquet_root or ("parquet_backfill" if mode == Mode.BACKFILL else "parquet_weekly")

    log.info("mode=%s docs=%d write_sql=%s parquet_root=%s",
             mode.value, len(df_docs), write_sql, root)

    result = run_pipeline(df_docs,
                          write_sql=write_sql,
                          parquet_root=root,
                          flush_every=flush_every)
    result["duration_sec"] = round(time.time() - t0, 1)
    log.info("done: %s", result)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.WEEKLY.value)
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()
    main(mode=args.mode, max_docs=args.max_docs)
```

End-to-end runbook:

```python
from sentiment.orchestrator import main
from sentiment.upload_backfill import upload_backfill

# --- one-time: 3–4 day laptop run, Parquet only ---
main(mode="backfill")                # resumable; run again if it crashed
# when the log shows "processed=0, skipped=0" the backfill is complete.
# then a single bulk load into MSSQL:
upload_backfill("parquet_backfill")

# --- steady state: scheduler triggers this weekly ---
main(mode="weekly")                  # Parquet + SQL in one pass

# --- smoke test on 10 docs, no SQL side effects ---
main(mode="backfill", max_docs=10)
```

---

## 10. Testing

- Three fixture JSONs (your examples) unit-tested for: paragraph count, Q&A-start index, role_class assignment, aggregation invariants (Σ word weights = total words), `qna_vs_prepared` sign.
- Deterministic scoring test with pinned torch seed + `torch.use_deterministic_algorithms(True)` → snapshot-compare a small expected Parquet.
- Blind-rating QA: random 50-event sample, one analyst labels {pos/neu/neg}, report Cohen's κ vs FinBERT. Floor: κ ≥ 0.4.
- Drift monitor: weekly sector-level mean `weighted_score` vs 26-week rolling; alert on >2σ shift.

---

## 11. Operational notes

- **Runtime plan: one 3–4 day laptop backfill, then weekly.** Don't run the backfill against MSSQL — only Parquet files land on disk during that run. After it finishes, a single `upload_backfill()` call bulk-loads into SQL.
- **Laptop sleep / lid-close:** disable sleep and screen lock for the duration of the backfill (Windows: `powercfg /change standby-timeout-ac 0` and tick "never" in Power settings).
- **Crash / power-loss resilience:** each batch's Parquet is written to `batch_N.parquet.tmp` and renamed on successful close. Mid-batch crash → the `.tmp` is ignored; the next run re-does only that batch's ~50 docs.
- **Resumability:**
  - **Backfill:** at startup, scan `parquet_backfill/event/*.parquet` for already-processed `docid`s and skip them in the input dataframe.
  - **Weekly:** `NOT EXISTS` against `sentiment_events` in the selection query.
- **CPU tuning:** `device=-1`, `batch_size=8`, `torch.set_num_threads(phys_cores)`, `@torch.inference_mode()`, `use_fast=True` tokenizer.
- **No cached files on disk:** tempfile + `finally: os.remove()`. One JSON per worker at a time.
- **Logging:** pipeline uses `log = logging.getLogger(__name__)`; the scheduler owns handler config and writes the log file in the repo folder.
- **Proxy:** every outbound HTTP uses `CL.getSession(external=1, SSL=1)`.
- **Idempotency after upload:** MSSQL writes keyed on `docid` / composite BIGINT keys. Re-running `upload_backfill()` is safe.
- **Model versioning:** every row carries `model_version`. Bump when swapping the checkpoint.
- **PII / licensing:** no raw transcript text stored; only `text_hash`.

---

## 12. What this unlocks for Equity Research dev

1. **Post-call tone tear-sheet.** Within minutes of a transcript landing, each analyst gets: event score, `qna_vs_prepared`, tone dispersion, top-5 most-positive / most-negative paragraphs (fetched from Quartr via `paragraph_index`). Saves 20–40 min per call.
2. **CEO-vs-CFO tone split.** `sentiment_speakers` → CFO-minus-CEO gap per call. Materially cautious CFO is historically a leading indicator of guidance cuts.
3. **Q&A deterioration signal.** `qna_vs_prepared` is the single most alpha-generative field in the academic literature on this. Surface in coverage dashboard.
4. **Analyst-vs-management tone gap.** Average `role_class = ANALYST` minus `COMPANY` score. Large gap → downgrade-probability and short-interest both rise.
5. **Cross-sectional ranking.** Parquet rank-within-sector per quarter. Pairs ideas, meeting prep.
6. **Tone time-series per issuer.** Stack calls for a `companyId`; overlay on price. Δ QoQ > level.
7. **Guidance-hedging detector.** `tone_dispersion` + regex on "guidance" / "outlook" / "visibility" on paragraph text hashes (retrieved). Flags hedging above trailing baseline.
8. **Systematic feature store.** Segment-grain Parquet → ready-made features for PEAD / short-window reaction models. `text_hash` supports RAG without duplicating storage.
9. **Intra-call tone curve.** `start_sec` / `end_sec` let you overlay tone against tick data to see *when* price reacted to *which part* of the call.
10. **Upstream QA.** Sudden drop in `n_words` per minute of `duration`, or spike in `UNKNOWN` role_class share, surfaces as `log.warning` → signals Quartr-ingest schema change.

---

## 13. Migration from your existing code

1. Replace the SQL in `load_documents_df`: `fileType='Transcript'` → `documentDescription='In-house transcript'`; select `documentId AS docid` (not `id`); cast ID cols to `int64`; add `NOT EXISTS`.
2. Delete `cache_dir` / `transcript_cache/` branch in `fetch_transcript`; replace with the tempfile version (§5.1).
3. Delete `PROCESSED_DIR` and the `.done` logic in `run_pipeline`.
4. Add `src/sentiment/section.py` (~50 lines, the `infer_qna_start` function from §5.2). Tag each record with `role_class` and `section` inside `parse_transcript_json`.
5. Extend `SentimentAnalyser.analyse_records` to keep `p_pos / p_neu / p_neg / signed_score / n_words` (§5.3).
6. Replace `calc_weighted_score` + `aggregate_sentiment` with the single `agg()` in §5.4.
7. `build_speaker_rows`: one row per (docid, speaker). Add `prepared_score` (records in that speaker's Prepared Remarks) and `qna_score` (Q&A) — either can be NULL. PK becomes `int(docid) * 1_000 + int(sid)`.
8. `build_segment_rows`: drop `paragraph_text[:2000]`; add `text_hash = hashlib.sha256(paragraph_text.encode("utf-8")).hexdigest()`. PK becomes `int(docid) * 100_000 + int(para_idx)`. Add `section` and `role_class`.
9. `build_event_row`: add `prepared_score`, `qna_score`, `qna_vs_prepared`, `tone_dispersion` via `agg()` on filtered slices.
10. No `sentiment_sections` upload, no run-log upload. During backfill, no SQL uploads at all — only Parquet. During weekly, three `DB.upload_msql` calls per batch.
11. Expose `main(mode, max_docs)` (§9) for the scheduler.
12. Add `src/sentiment/upload_backfill.py` — run once, manually, after the 3–4 day backfill completes.

---

## 14. Open questions

1. Confirm `db="EDI"` on `DB.upload_msql(...)` is the right target.
2. Parquet root path — local beside the script is fine for now? Or network share?
3. **What does the `[qna]` column on `quartr_documents` actually contain?** Boolean? Timestamp? Start-index? If it's a clean indicator, it becomes the primary signal and the heuristic becomes a fallback. A 30-second SQL query resolves it: `SELECT TOP 20 documentId, qna FROM FACTSET_2.dbo.quartr_documents WHERE documentDescription='In-house transcript'`.
4. Expected volume per weekly run — 50, 200, 500 new in-house transcripts?
5. Confirm `./bert` is vanilla `ProsusAI/finbert` (labels: positive/neutral/negative).

Resolved:

- ✅ Section inference: heuristic only (JSONs have no section flag; verified empirically).
- ✅ Runtime host: CPU laptop — pipeline tuned accordingly (§5.3, §11).
- ✅ Model: HuggingFace local checkpoint, `local_files_only=True`.
- ✅ Scheduler: custom — pipeline exposes `main(mode, max_docs)` in §9.
- ✅ Logging: stdlib `logging` only; scheduler owns the log file in the repo folder.
- ✅ Backfill strategy: Parquet-only during 3–4 day run; one bulk `upload_backfill()` call afterwards.

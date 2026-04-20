import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .pipeline import (
    SentimentAnalyser,
    build_event_row,
    build_segment_rows,
    build_speaker_rows,
    fetch_transcript,
    get_http_session,
    load_documents_df,
    parse_transcript_json,
    upload_events,
    upload_segments,
    upload_speakers,
)

log = logging.getLogger(__name__)


class Mode(str, Enum):
    BACKFILL = "backfill"
    WEEKLY   = "weekly"


# -----------------------
# PARQUET WRITER (atomic)
# -----------------------
def _write_parquet_atomic(df: pd.DataFrame, folder: Path, batch_idx: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"batch_{batch_idx:04d}.parquet"
    tmp   = folder / f"batch_{batch_idx:04d}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)


def _next_batch_idx(root: Path) -> int:
    event_dir = root / "event"
    if not event_dir.exists():
        return 1
    existing = sorted(event_dir.glob("batch_*.parquet"))
    return int(existing[-1].stem.split("_")[1]) + 1 if existing else 1


def _already_processed(root: Path) -> set[int]:
    event_dir = root / "event"
    if not event_dir.exists():
        return set()
    done: set[int] = set()
    for f in event_dir.glob("batch_*.parquet"):
        try:
            done.update(int(x) for x in pd.read_parquet(f, columns=["docid"])["docid"].tolist())
        except Exception:
            log.exception("failed reading %s", f)
    return done


def _flush(event_rows, speaker_rows, segment_rows, root: Path, batch_idx: int, write_sql: bool):
    df_e = pd.DataFrame(event_rows)
    df_sp = pd.DataFrame(speaker_rows)
    df_sg = pd.DataFrame(segment_rows)

    _write_parquet_atomic(df_e,  root / "event",   batch_idx)
    _write_parquet_atomic(df_sp, root / "speaker", batch_idx)
    _write_parquet_atomic(df_sg, root / "segment", batch_idx)

    log.info("flushed batch=%d events=%d speakers=%d segments=%d",
             batch_idx, len(df_e), len(df_sp), len(df_sg))

    if write_sql:
        upload_events(df_e)
        upload_speakers(df_sp)
        upload_segments(df_sg)
        log.info("sql upload ok batch=%d", batch_idx)


# -----------------------
# PIPELINE RUNNER
# -----------------------
def run_pipeline(df_docs: pd.DataFrame, write_sql: bool, parquet_root: str, flush_every: int = 50) -> dict:
    root = Path(parquet_root)
    session = get_http_session()
    analyser = SentimentAnalyser()

    done = _already_processed(root)
    if done:
        df_docs = df_docs[~df_docs["docid"].isin(done)].reset_index(drop=True)
        log.info("resumed: skipping %d already-processed docs", len(done))

    event_rows, speaker_rows, segment_rows = [], [], []
    processed = failed = skipped = 0
    batch_idx = _next_batch_idx(root)

    for i, (_, row) in enumerate(tqdm(df_docs.iterrows(), total=len(df_docs), desc="Processing"), start=1):
        docid    = int(row["docid"])
        fileurl  = row["fileurl"]
        log.info("doc=%s start url=%s", docid, fileurl)
        t0 = time.time()

        try:
            data = fetch_transcript(session, fileurl)
            if data is None:
                log.warning("doc=%s skipped (bad url)", docid)
                skipped += 1
                continue

            records, meta = parse_transcript_json(data, docid)
            if not records:
                log.warning("doc=%s no records after parse", docid)
                skipped += 1
                continue

            analyser.analyse_records(records)
            records = [r for r in records if "signed_score" in r]
            if not records:
                skipped += 1
                continue

            run_ts = datetime.now(timezone.utc)
            event_rows.append(build_event_row(
                records, docid, int(row["companyid"]), int(row["eventid"]),
                row["createdat"], meta["number_of_speakers"], run_ts,
            ))
            speaker_rows.extend(build_speaker_rows(records, meta, docid, int(row["companyid"]), int(row["eventid"]), run_ts))
            segment_rows.extend(build_segment_rows(records, docid, int(row["companyid"]), int(row["eventid"]), run_ts))

            processed += 1
            log.info("doc=%s OK paragraphs=%d sentences=%d duration_ms=%d",
                     docid, len({r["paragraph_index"] for r in records}), len(records),
                     int((time.time() - t0) * 1000))

        except Exception as exc:
            failed += 1
            log.exception("doc=%s FAILED: %s", docid, exc)

        if len(event_rows) >= flush_every:
            _flush(event_rows, speaker_rows, segment_rows, root, batch_idx, write_sql)
            event_rows.clear(); speaker_rows.clear(); segment_rows.clear()
            batch_idx += 1

    if event_rows:
        _flush(event_rows, speaker_rows, segment_rows, root, batch_idx, write_sql)

    return {"processed": processed, "failed": failed, "skipped": skipped}


# -----------------------
# ENTRYPOINT
# -----------------------
def main(
    mode: str | Mode = Mode.WEEKLY,
    max_docs: int | None = None,
    flush_every: int = 50,
    parquet_root: str | None = None,
) -> dict:
    mode = Mode(mode)
    t0 = time.time()

    df_docs = load_documents_df(weekly=(mode == Mode.WEEKLY))
    if max_docs is not None:
        df_docs = df_docs.head(max_docs)

    write_sql = (mode == Mode.WEEKLY)
    root = parquet_root or ("parquet_backfill" if mode == Mode.BACKFILL else "parquet_weekly")

    log.info("mode=%s docs=%d write_sql=%s parquet_root=%s", mode.value, len(df_docs), write_sql, root)

    result = run_pipeline(df_docs, write_sql=write_sql, parquet_root=root, flush_every=flush_every)
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

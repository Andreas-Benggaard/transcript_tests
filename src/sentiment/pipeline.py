import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from Utilities import CommonLibrary as CL
from Utilities import Database as DB

from . import MODEL_VERSION

log = logging.getLogger(__name__)

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "./bert"
MAX_TOKENS = 512
INFERENCE_BATCH_SIZE = 8

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
torch.set_num_interop_threads(1)


# -----------------------
# HTTP
# -----------------------
def get_http_session():
    s = CL.getSession(external=1, SSL=1)
    s.headers.update({"User-Agent": "andreas.benggaard@nordea.com"})
    s.trust_env = True
    return s


# -----------------------
# FETCH TRANSCRIPT (tempfile, no cache)
# -----------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2), reraise=True)
def _http_get(session, url, timeout):
    r = session.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    return r


def fetch_transcript(session, url: str, timeout: int = 60):
    if not isinstance(url, str) or not url.startswith("http"):
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        r = _http_get(session, url, timeout)
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        with open(tmp.name, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.remove(tmp.name)
        except OSError:
            pass


# -----------------------
# ROLE CLASS + SECTION INFERENCE
# -----------------------
_COMPANY_TERMS  = ("ceo","cfo","coo","cto","chair","president","ir",
                   "investor relations","head of","director of","vp",
                   "vice president","managing director")
_ANALYST_TERMS  = ("analyst","equity","research","portfolio manager","pm")
_OPERATOR_TERMS = ("operator","moderator")
_QNA_RE = re.compile(
    r"\bquestions?\b|\bq\s*&\s*a\b|\bfirst question\b|"
    r"open.*line.*for.*questions|time for (any )?questions",
    re.I,
)


def role_class(role: str | None) -> str:
    r = (role or "").lower()
    if any(t in r for t in _OPERATOR_TERMS): return "OPERATOR"
    if any(t in r for t in _ANALYST_TERMS):  return "ANALYST"
    if any(t in r for t in _COMPANY_TERMS):  return "COMPANY"
    return "UNKNOWN"


def infer_qna_start(paragraphs, speaker_mapping) -> int | None:
    smap = {s["speaker"]: s["speaker_data"] for s in speaker_mapping}
    seen_company = False
    for i, p in enumerate(paragraphs):
        rc = role_class(smap.get(p.get("speaker"), {}).get("role"))
        if rc == "COMPANY":
            seen_company = True
            continue
        if seen_company and rc == "ANALYST":
            return i
        if seen_company and rc == "OPERATOR" and _QNA_RE.search(p.get("text", "") or ""):
            return i
    return None


# -----------------------
# PARSE TRANSCRIPT -> SENTENCE RECORDS
# -----------------------
def parse_transcript_json(data: dict, docid: int) -> tuple[list[dict[str, Any]], dict]:
    transcript = data["transcript"]
    paragraphs = transcript.get("paragraphs", []) or []
    speaker_mapping = data.get("speaker_mapping", []) or []
    smap = {s["speaker"]: s["speaker_data"] for s in speaker_mapping}

    qna_start = infer_qna_start(paragraphs, speaker_mapping)
    if not any(role_class(v.get("role")) == "COMPANY" for v in smap.values()):
        log.warning("doc=%s no COMPANY speakers — whole call classified PREPARED_REMARKS", docid)

    records: list[dict[str, Any]] = []
    for p_idx, para in enumerate(paragraphs):
        speaker_id = para.get("speaker")
        rc = role_class(smap.get(speaker_id, {}).get("role") if speaker_id is not None else None)
        section = "QNA" if (qna_start is not None and p_idx >= qna_start) else "PREPARED_REMARKS"

        for s_idx, sent in enumerate(para.get("sentences", []) or []):
            text = (sent.get("text") or "").strip()
            if not text:
                continue
            records.append({
                "docid": int(docid),
                "paragraph_index": p_idx,
                "sentence_index": s_idx,
                "speaker_id": int(speaker_id) if speaker_id is not None else -1,
                "role_class": rc,
                "section": section,
                "text": text,
                "sentence_start": sent.get("start"),
                "sentence_end": sent.get("end"),
                "paragraph_start": para.get("start"),
                "paragraph_end": para.get("end"),
                "paragraph_text": para.get("text", "") or "",
            })

    meta = {
        "number_of_speakers": transcript.get("number_of_speakers", 0),
        "speaker_mapping": smap,
    }
    return records, meta


# -----------------------
# SENTIMENT ANALYSER (CPU-tuned)
# -----------------------
class SentimentAnalyser:
    def __init__(self, model_path: str = MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        model.eval()
        self.pipe = pipeline(
            "text-classification",
            model=model, tokenizer=tokenizer,
            device=-1, top_k=None,
            truncation=True, max_length=MAX_TOKENS,
        )

    @torch.inference_mode()
    def analyse_records(self, records: list[dict]) -> None:
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


# -----------------------
# AGGREGATION (word-count weighted)
# -----------------------
def agg(recs: list[dict]) -> dict:
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
    dispersion = (sum((r["signed_score"] - weighted) ** 2 for r in recs) / n) ** 0.5
    dominant = max(("positive", npos), ("negative", nneg), ("neutral", nneu), key=lambda x: x[1])[0]
    return {
        "weighted_score": round(weighted, 5),
        "p_pos": round(p_pos, 5), "p_neu": round(p_neu, 5), "p_neg": round(p_neg, 5),
        "pct_pos": round(npos / n, 5), "pct_neg": round(nneg / n, 5), "pct_neu": round(nneu / n, 5),
        "tone_dispersion": round(dispersion, 5),
        "n_sentences": n, "n_words": wc, "dominant_sentiment": dominant,
    }


def _section_score(recs: list[dict], section: str) -> float | None:
    sub = [r for r in recs if r["section"] == section]
    return agg(sub)["weighted_score"] if sub else None


# -----------------------
# ROW BUILDERS
# -----------------------
def build_event_row(records, docid, companyid, eventid, createdat, num_speakers, run_ts):
    non_op = [r for r in records if r["role_class"] != "OPERATOR"]
    a = agg(non_op)
    prepared = _section_score(non_op, "PREPARED_REMARKS")
    qna = _section_score(non_op, "QNA")
    ends = [r["paragraph_end"] for r in records if r.get("paragraph_end") is not None]
    return {
        "docid": int(docid), "companyid": int(companyid), "eventid": int(eventid),
        "createdat": createdat, "total_sentences": a["n_sentences"],
        "total_words": a["n_words"], "num_speakers": int(num_speakers),
        "duration_sec": round(max(ends), 2) if ends else None,
        "weighted_score": a["weighted_score"],
        "p_pos": a["p_pos"], "p_neu": a["p_neu"], "p_neg": a["p_neg"],
        "pct_positive": a["pct_pos"], "pct_negative": a["pct_neg"], "pct_neutral": a["pct_neu"],
        "tone_dispersion": a["tone_dispersion"], "prepared_score": prepared, "qna_score": qna,
        "qna_vs_prepared": round(qna - prepared, 5) if (prepared is not None and qna is not None) else None,
        "dominant_sentiment": a["dominant_sentiment"],
        "model_version": MODEL_VERSION, "run_ts": run_ts,
    }


def build_speaker_rows(records, meta, docid, companyid, eventid, run_ts):
    smap = meta.get("speaker_mapping", {})
    groups: dict[int, list] = {}
    for r in records:
        groups.setdefault(r["speaker_id"], []).append(r)

    rows = []
    for sid, recs in groups.items():
        a = agg(recs)
        sd = smap.get(sid, {}) if sid >= 0 else {}
        rows.append({
            "docid_speakerid": int(docid) * 1_000 + int(sid),
            "docid": int(docid), "companyid": int(companyid), "eventid": int(eventid),
            "speaker_id": int(sid),
            "speaker_name": sd.get("name"), "speaker_role": sd.get("role"),
            "speaker_company": sd.get("company"), "role_class": recs[0]["role_class"],
            "weighted_score": a["weighted_score"],
            "p_pos": a["p_pos"], "p_neu": a["p_neu"], "p_neg": a["p_neg"],
            "pct_positive": a["pct_pos"], "pct_negative": a["pct_neg"], "pct_neutral": a["pct_neu"],
            "tone_dispersion": a["tone_dispersion"],
            "prepared_score": _section_score(recs, "PREPARED_REMARKS"),
            "qna_score": _section_score(recs, "QNA"),
            "dominant_sentiment": a["dominant_sentiment"],
            "n_sentences": a["n_sentences"], "n_words": a["n_words"],
            "model_version": MODEL_VERSION, "run_ts": run_ts,
        })
    return rows


def build_segment_rows(records, docid, companyid, eventid, run_ts):
    groups: dict[int, list] = {}
    for r in records:
        groups.setdefault(r["paragraph_index"], []).append(r)

    rows = []
    for p_idx, recs in sorted(groups.items()):
        a = agg(recs)
        first = recs[0]
        text_hash = hashlib.sha256((first.get("paragraph_text") or "").encode("utf-8")).hexdigest()
        rows.append({
            "docid_paragraphindex": int(docid) * 100_000 + int(p_idx),
            "docid": int(docid), "companyid": int(companyid), "eventid": int(eventid),
            "paragraph_index": int(p_idx), "speaker_id": int(first["speaker_id"]),
            "role_class": first["role_class"], "section": first["section"],
            "weighted_score": a["weighted_score"],
            "p_pos": a["p_pos"], "p_neu": a["p_neu"], "p_neg": a["p_neg"],
            "argmax_label": a["dominant_sentiment"],
            "n_sentences": a["n_sentences"], "n_words": a["n_words"],
            "start_sec": first.get("paragraph_start"), "end_sec": first.get("paragraph_end"),
            "text_hash": text_hash, "model_version": MODEL_VERSION, "run_ts": run_ts,
        })
    return rows


# -----------------------
# DB IO
# -----------------------
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
    if df.empty:
        return df
    for col in ("docid", "companyid", "eventid"):
        df[col] = df[col].astype("int64")
    return df


def upload_events(df: pd.DataFrame):
    DB.upload_msql(df=df.drop_duplicates(subset=["docid"], keep="last"),
                   schema="raw", table="sentiment_events", db="EDI", replace=0, id_col_name="docid")


def upload_speakers(df: pd.DataFrame):
    DB.upload_msql(df=df.drop_duplicates(subset=["docid_speakerid"], keep="last"),
                   schema="raw", table="sentiment_speakers", db="EDI", replace=0, id_col_name="docid_speakerid")


def upload_segments(df: pd.DataFrame):
    DB.upload_msql(df=df.drop_duplicates(subset=["docid_paragraphindex"], keep="last"),
                   schema="raw", table="sentiment_segments", db="EDI", replace=0, id_col_name="docid_paragraphindex")

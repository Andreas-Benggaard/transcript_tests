from __future__ import annotations

from typing import Any

from .base import SentenceRecord


def parse_transcript(data: dict[str, Any]) -> list[SentenceRecord]:
    """Parse either transcript format into a flat list of SentenceRecords.

    In-house format: has 'speaker_mapping' key with named speakers.
    Timestamped format: no speaker_mapping, speaker fields are None.
    """
    event_id = data["event_id"]
    company_id = data["company_id"]

    speaker_map: dict[int, dict[str, str]] = {}
    for entry in data.get("speaker_mapping", []):
        speaker_map[entry["speaker"]] = entry.get("speaker_data", {})

    records: list[SentenceRecord] = []
    for para_idx, paragraph in enumerate(data["transcript"]["paragraphs"]):
        speaker_id: int = paragraph.get("speaker", -1)
        speaker_data = speaker_map.get(speaker_id, {})

        for sent_idx, sentence in enumerate(paragraph.get("sentences", [])):
            text = sentence.get("text", "").strip()
            if not text:
                continue
            records.append(
                SentenceRecord(
                    event_id=event_id,
                    company_id=company_id,
                    paragraph_index=para_idx,
                    sentence_index=sent_idx,
                    speaker_id=speaker_id,
                    speaker_name=speaker_data.get("name"),
                    speaker_role=speaker_data.get("role"),
                    speaker_company=speaker_data.get("company"),
                    text=text,
                    start_time=sentence.get("start", 0.0),
                    end_time=sentence.get("end", 0.0),
                )
            )

    return records

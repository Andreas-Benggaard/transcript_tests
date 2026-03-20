class SentenceRecord:
    def __init__(
        self,
        event_id,
        company_id,
        paragraph_index,
        sentence_index,
        speaker_id,
        speaker_name,
        speaker_role,
        speaker_company,
        text,
        start_time,
        end_time,
        speaker_type="unknown",
        section="unknown",
    ):
        self.event_id = event_id
        self.company_id = company_id
        self.paragraph_index = paragraph_index
        self.sentence_index = sentence_index
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.speaker_role = speaker_role
        self.speaker_company = speaker_company
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_type = speaker_type
        self.section = section


def parse_transcript(data):
    """Parse either transcript format into a flat list of SentenceRecords.

    In-house format: has 'speaker_mapping' key with named speakers.
    Timestamped format: no speaker_mapping, speaker fields are None.
    """
    event_id = data["event_id"]
    company_id = data["company_id"]

    # Build speaker lookup table
    speaker_map = {}
    for entry in data.get("speaker_mapping", []):
        speaker_map[entry["speaker"]] = entry.get("speaker_data", {})

    records = []
    for para_idx, paragraph in enumerate(data["transcript"]["paragraphs"]):
        speaker_id = paragraph.get("speaker", -1)
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

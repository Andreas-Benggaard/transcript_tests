from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SentenceRecord:
    event_id: int
    company_id: int
    paragraph_index: int
    sentence_index: int
    speaker_id: int
    speaker_name: Optional[str]
    speaker_role: Optional[str]
    speaker_company: Optional[str]
    text: str
    start_time: float
    end_time: float
    speaker_type: str = field(default="unknown")
    section: str = field(default="unknown")

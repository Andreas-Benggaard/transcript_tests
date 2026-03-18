from __future__ import annotations

from ..config import ANALYST_KEYWORDS, MANAGEMENT_KEYWORDS, MODERATOR_KEYWORDS


def classify_speaker(
    speaker_role: str | None,
    speaker_company: str | None = None,
) -> str:
    """Classify a speaker as management / analyst / moderator / unknown.

    Uses keyword matching on the role string. Falls back to 'unknown'
    when no role is available.
    """
    if speaker_role is None:
        return "unknown"

    role_lower = speaker_role.lower()

    if _matches_any(role_lower, MODERATOR_KEYWORDS):
        return "moderator"
    # Check analyst BEFORE management — bank analysts often have titles
    # like "VP - Equity Research" where "vp" would false-match management.
    if _matches_any(role_lower, ANALYST_KEYWORDS):
        return "analyst"
    if _matches_any(role_lower, MANAGEMENT_KEYWORDS):
        return "management"

    return "unknown"


def _matches_any(text: str, keywords: frozenset[str]) -> bool:
    return any(kw in text for kw in keywords)

"""Quick smoke test against the two example transcripts."""

import json
from datetime import date
from pathlib import Path

from src.pipeline import run_sentiment_pipeline

ROOT = Path(__file__).parent


def get_transcripts() -> list[dict]:
    inhouse = json.loads((ROOT / "inhouse_example.json").read_text())
    inhouse["date"] = date(2025, 1, 15)

    timestamped = json.loads((ROOT / "timestamped_transcript_example.json").read_text())
    timestamped["date"] = date(2019, 5, 8)

    return [inhouse, timestamped]


if __name__ == "__main__":
    transcripts = get_transcripts()
    result = run_sentiment_pipeline(transcripts)

    # Save CSVs
    result.events_df.to_csv(ROOT / "sentiment_events.csv", index=False)
    result.events_by_section_df.to_csv(ROOT / "sentiment_events_by_section.csv", index=False)
    print("Saved: sentiment_events.csv, sentiment_events_by_section.csv")

    print("\n=== EVENT-LEVEL SENTIMENT ===")
    cols = ["event_id", "company_id", "date", "sentiment_label",
            "net_sentiment", "management_net_sentiment", "analyst_net_sentiment",
            "sentence_count"]
    print(result.events_df[[c for c in cols if c in result.events_df.columns]].to_string(index=False))

    print("\n=== BY SECTION ===")
    cols = ["event_id", "section", "sentiment_label", "net_sentiment", "sentence_count"]
    print(result.events_by_section_df[[c for c in cols if c in result.events_by_section_df.columns]].to_string(index=False))

    print("\n=== PARAGRAPH-LEVEL (first 10) ===")
    cols = ["event_id", "paragraph_index", "speaker_name", "speaker_type",
            "section", "sentiment_label", "net_sentiment"]
    print(result.paragraphs_df[[c for c in cols if c in result.paragraphs_df.columns]].head(10).to_string(index=False))

    print("\n=== SENTENCE-LEVEL (first 10) ===")
    cols = ["event_id", "speaker_name", "speaker_type", "section",
            "text", "sentiment_label", "sentiment_positive"]
    print(result.sentences_df[[c for c in cols if c in result.sentences_df.columns]].head(10).to_string(index=False))

import tempfile
from pathlib import Path

# Batch processing
BATCH_SIZE = 10
FINBERT_BATCH_SIZE = 32
DOWNLOAD_TIMEOUT_SECONDS = 30

# Sentiment thresholds for event-level labeling
NET_SENTIMENT_POSITIVE_THRESHOLD = 0.05
NET_SENTIMENT_NEGATIVE_THRESHOLD = -0.05

# Paths
TEMP_DIR = Path(tempfile.gettempdir()) / "transcripts_pipeline"
STATE_FILE = Path("~/.transcripts_pipeline_last_run").expanduser()

# Speaker classification keywords (lowercase)
MANAGEMENT_KEYWORDS = frozenset({
    "ceo", "cfo", "coo", "cto", "president", "chief",
    "evp", "svp", "vp", "vice president",
    "director", "head of", "officer", "managing director",
    "founder", "co-founder", "chairman",
})
ANALYST_KEYWORDS = frozenset({
    "analyst", "research", "equity research",
    "portfolio manager", "fund manager",
})
MODERATOR_KEYWORDS = frozenset({
    "operator", "moderator", "conference", "investor relations", "ir",
})

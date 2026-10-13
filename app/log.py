"""Logging utilities for unanswered queries and telemetry."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
UNANSWERED_LOG_FILE = LOGS_DIR / "unanswered.jsonl"


def log_unanswered(question_len: int, reason: str, request_id: str) -> None:
    """Append an unanswered query record to logs/unanswered.jsonl safely."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_len": question_len,
            "reason": reason,
            "request_id": request_id,
        }
        with open(UNANSWERED_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write to unanswered log: %s", exc)

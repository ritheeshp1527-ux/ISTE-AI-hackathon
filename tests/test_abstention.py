"""Tests for the abstention / thresholding path."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import answer
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_abstention_on_force_low_score(client, monkeypatch):
    log_file = Path("logs/unanswered.jsonl")
    if log_file.exists():
        log_file.unlink()

    # Define llm_guard that raises if called, proving call_llm was never invoked on low-score path
    def llm_guard(*args, **kwargs):
        raise AssertionError("call_llm should never be invoked on below_threshold branch")

    monkeypatch.setattr(answer, "call_llm", llm_guard)

    payload = {
        "question": "What is the procedure for sample storage?",
        "user": "alice",
        "force_low_score": True,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify abstention answer contract
    assert data["status"] == "insufficient_evidence"
    assert data["citations"] == []
    assert data["answer"] == ""
    assert "latency_ms" in data
    assert "passages" in data

    # Verify log entry in logs/unanswered.jsonl
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reason"] == "below_threshold"
    assert record["question_len"] == len(payload["question"])
    assert "timestamp" in record
    assert "request_id" in record
    assert record["request_id"] == data["request_id"]


def test_normal_retrieve_returns_answered(client, monkeypatch):
    monkeypatch.setattr(answer, "call_llm", lambda s, u: "Normal answered response [S1].")

    payload = {
        "question": "What is the procedure for sample storage?",
        "user": "alice",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert data["answer"] != ""

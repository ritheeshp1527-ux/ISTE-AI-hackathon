"""Tests for citation validation and abstention when evidence citations fail."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.answer import answer_question, validate_citations
from app.contracts import Chunk, Passage
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def make_passages(n: int) -> list[Passage]:
    passages = []
    for i in range(1, n + 1):
        chunk = Chunk(
            chunk_id=f"chunk-{i}",
            doc_id=f"DOC-{i}",
            title=f"Document {i}",
            text=f"Text for passage {i}",
            page=i,
            access_group="general",
        )
        passages.append(Passage(chunk=chunk, score=0.9))
    return passages


def test_validate_citations_strips_invalid_markers_and_brackets():
    passages = make_passages(3)
    raw = "Store cold [S1] and [S9]"
    clean_text, citations, has_valid = validate_citations(raw, passages)

    assert has_valid is True
    assert len(citations) == 1
    assert citations[0].source_id == "DOC-1"
    assert citations[0].title == "Document 1"
    assert citations[0].page == 1

    # Clean text has no remaining bracket markers
    assert "[" not in clean_text
    assert "]" not in clean_text
    assert "[S9]" not in clean_text
    assert "[S1]" not in clean_text
    assert clean_text == "Store cold and"


def test_endpoint_no_valid_citations_returns_insufficient_evidence(client, monkeypatch):
    log_file = Path("logs/unanswered.jsonl")
    if log_file.exists():
        log_file.unlink()

    monkeypatch.setattr("app.answer.call_llm", lambda s, u: "[S4] [S7]")

    response = client.post("/api/chat", json={
        "question": "Where are the chemicals stored?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_evidence"
    assert data["answer"] == ""
    assert data["citations"] == []

    # Verify log entry with reason "no_valid_citations"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reason"] == "no_valid_citations"


def test_no_markers_at_all_has_valid_false():
    passages = make_passages(3)
    raw = "This is a statement without any citations."
    clean_text, citations, has_valid = validate_citations(raw, passages)
    assert has_valid is False
    assert citations == []


def test_citations_deduplicated_in_first_appearance_order():
    passages = make_passages(3)
    raw = "Observation [S2] verified again [S2] and also [S1]."
    clean_text, citations, has_valid = validate_citations(raw, passages)

    assert has_valid is True
    assert len(citations) == 2
    assert citations[0].source_id == "DOC-2"
    assert citations[1].source_id == "DOC-1"


def test_s10_validity_with_ten_vs_three_passages():
    raw = "Standard guideline [S10]."

    # 10 passages: S10 is valid
    passages_10 = make_passages(10)
    clean_10, citations_10, has_valid_10 = validate_citations(raw, passages_10)
    assert has_valid_10 is True
    assert len(citations_10) == 1
    assert citations_10[0].source_id == "DOC-10"

    # 3 passages: S10 is invalid
    passages_3 = make_passages(3)
    clean_3, citations_3, has_valid_3 = validate_citations(raw, passages_3)
    assert has_valid_3 is False
    assert len(citations_3) == 0


def test_llm_said_not_found_returns_insufficient_evidence(client, monkeypatch):
    log_file = Path("logs/unanswered.jsonl")
    if log_file.exists():
        log_file.unlink()

    monkeypatch.setattr(
        "app.answer.call_llm",
        lambda s, u: "The passages do not contain the answer.",
    )

    response = client.post("/api/chat", json={
        "question": "What is the secret formula?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_evidence"
    assert data["answer"] == ""
    assert data["citations"] == []

    # Verify log entry with reason "llm_said_not_found"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reason"] == "llm_said_not_found"


def test_clean_text_never_contains_bracket_s_pattern():
    passages = make_passages(5)
    samples = [
        "Result [S1] and [S2] followed by [S99].",
        "[S1][S2][S3]",
        "Prefix [S5] suffix",
        "[S4]",
    ]
    for sample in samples:
        clean_text, _, _ = validate_citations(sample, passages)
        assert not re.search(r"\[S", clean_text)
        assert not re.search(r"\[S\d*\]", clean_text)


def test_normal_grounded_answer_two_citations_and_stripped_markers(client, monkeypatch):
    monkeypatch.setattr("app.answer.call_llm", lambda s, u: "Store at 2-8 °C [S1] [S2]")

    response = client.post("/api/chat", json={
        "question": "What are storage rules?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "answered"
    assert len(data["citations"]) == 2
    # Verify citations order
    assert data["citations"][0]["title"] == "Sample Handling Procedure"
    assert data["citations"][1]["title"] == "General Laboratory Safety"
    # Verify answer text has markers stripped
    assert "[S1]" not in data["answer"]
    assert "[S2]" not in data["answer"]
    assert "[" not in data["answer"]
    assert "]" not in data["answer"]
    assert data["answer"] == "Store at 2-8 °C"

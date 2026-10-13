"""Tests for graceful error handling at the /api/chat boundary."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError, LLMTimeout
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_llm_timeout_handled_gracefully(client, monkeypatch):
    log_file = Path("logs/unanswered.jsonl")
    if log_file.exists():
        log_file.unlink()

    def raise_timeout(*args, **kwargs):
        raise LLMTimeout("Gateway timeout after 2 retries")

    monkeypatch.setattr("app.answer.call_llm", raise_timeout)

    response = client.post("/api/chat", json={
        "question": "What is the procedure?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["answer"] == ""
    assert data["citations"] == []
    assert data["conflict"] is None
    assert data["error_message"] == "The answering service timed out. Please try again."
    assert "request_id" in data
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reason"] == "llm_timeout"


def test_llm_error_handled_gracefully(client, monkeypatch):
    log_file = Path("logs/unanswered.jsonl")
    if log_file.exists():
        log_file.unlink()

    def raise_llm_error(*args, **kwargs):
        raise LLMError("Upstream API server error 500")

    monkeypatch.setattr("app.answer.call_llm", raise_llm_error)

    response = client.post("/api/chat", json={
        "question": "What is the procedure?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["answer"] == ""
    assert data["citations"] == []
    assert data["conflict"] is None
    assert data["error_message"] == "The answering service failed. Please try again."
    assert "request_id" in data
    assert "latency_ms" in data

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reason"] == "llm_error"


def test_retrieve_unexpected_exception_handled_gracefully(client, monkeypatch):
    log_file = Path("logs/unanswered.jsonl")
    if log_file.exists():
        log_file.unlink()

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("Vector database unhandled connection crash Error")

    monkeypatch.setattr("app.main.stub_retrieve", raise_runtime_error)

    response = client.post("/api/chat", json={
        "question": "What is the procedure?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["answer"] == ""
    assert data["citations"] == []
    assert data["conflict"] is None
    assert data["error_message"] == "The answering service failed. Please try again."
    assert "request_id" in data
    assert "latency_ms" in data

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["reason"] == "unexpected_error"


def test_error_messages_contain_no_internal_details(client, monkeypatch):
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("SecretInternalDatabaseCrashError with Traceback stack details")

    monkeypatch.setattr("app.main.stub_retrieve", raise_runtime_error)

    response = client.post("/api/chat", json={
        "question": "What is the procedure?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    err_msg = data["error_message"]
    assert "Traceback" not in err_msg
    assert "RuntimeError" not in err_msg
    assert "SecretInternalDatabaseCrashError" not in err_msg
    assert "stack" not in err_msg


def test_validation_400_rejections_passthrough_unchanged(client):
    # Empty question
    res_empty = client.post("/api/chat", json={"question": "", "user": "alice"})
    assert res_empty.status_code == 400
    assert "error" in res_empty.json()
    assert res_empty.json().get("status") != "error"

    # Whitespace question
    res_ws = client.post("/api/chat", json={"question": "   ", "user": "alice"})
    assert res_ws.status_code == 400
    assert "error" in res_ws.json()

    # Unknown user
    res_user = client.post("/api/chat", json={"question": "Valid question?", "user": "eve"})
    assert res_user.status_code == 400
    assert "error" in res_user.json()

    # Question too long
    res_long = client.post("/api/chat", json={"question": "x" * 2001, "user": "alice"})
    assert res_long.status_code == 400
    assert "error" in res_long.json()


def test_latency_ms_present_on_error_responses(client, monkeypatch):
    monkeypatch.setattr("app.answer.call_llm", lambda s, u: (_ for _ in ()).throw(LLMTimeout("timeout")))

    response = client.post("/api/chat", json={
        "question": "What is the procedure?",
        "user": "alice",
    })
    assert response.status_code == 200
    data = response.json()
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0

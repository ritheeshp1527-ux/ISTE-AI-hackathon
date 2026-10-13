"""Integration tests for FastAPI skeleton routes."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.answer.call_llm", lambda s, u: "Store samples at 2-8 °C [S1].")
    return TestClient(app)


def test_health_check_returns_200(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_valid_question_and_alice_returns_200(client):
    payload = {
        "question": "What is the procedure for handling samples?",
        "user": "alice",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "answered"
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))
    assert "passages" in data
    assert isinstance(data["passages"], list)
    assert len(data["passages"]) > 0
    # Check structure of passages
    first_passage = data["passages"][0]
    assert "title" in first_passage
    assert "page" in first_passage
    assert "score" in first_passage


def test_chat_empty_question_returns_400(client):
    # Empty string
    response = client.post("/api/chat", json={"question": "", "user": "alice"})
    assert response.status_code == 400
    assert "error" in response.json()

    # Whitespace only
    response_ws = client.post("/api/chat", json={"question": "   ", "user": "alice"})
    assert response_ws.status_code == 400
    assert "error" in response_ws.json()


def test_chat_unknown_user_returns_400(client):
    payload = {
        "question": "Where is the lab located?",
        "user": "eve",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()


def test_chat_bob_contains_no_quality_passages(client):
    payload = {
        "question": "Where do we store chemical samples?",
        "user": "bob",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    passages = data.get("passages", [])
    assert len(passages) > 0

    # The quality chunk has title "Sample Handling Procedure" and access_group "quality"
    # Neither should be accessible to bob
    titles = [p.get("title") for p in passages]
    assert "Sample Handling Procedure" not in titles
    for p in passages:
        assert p.get("access_group") != "quality"


def test_search_endpoint_returns_passages(client):
    payload = {
        "question": "What are safety guidelines?",
        "user": "alice",
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "passages" in data
    assert isinstance(data["passages"], list)
    assert len(data["passages"]) > 0


def test_documents_returns_not_implemented(client):
    payload = {
        "filename": "test.pdf",
        "access_group": "quality",
    }
    response = client.post("/api/documents", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "not_implemented"}


def test_chat_question_too_long_returns_400(client):
    payload = {
        "question": "A" * 2001,
        "user": "alice",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()

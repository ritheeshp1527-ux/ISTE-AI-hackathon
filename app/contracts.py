"""Data contracts and schemas for the RAG knowledge assistant."""

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


USERS: dict[str, list[str]] = {
    "alice": ["quality", "operations", "general"],
    "bob": ["operations", "general"],
}


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    page: int | None = None
    section: str | None = None
    access_group: str = "general"


class Passage(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    source_id: str
    title: str
    page: int | None = None
    section: str | None = None
    url: str | None = None
    chunk_id: str | None = None


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    status: str = "answered"
    request_id: str
    conflict: str | None = None
    error_message: str | None = None

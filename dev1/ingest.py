"""dev1/ingest.py

Dev 1 document ingestion module connecting to src.retrieval.ingest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contracts import Document, DocumentChunk, IngestionContract
from src.retrieval.ingest import (
    DocumentIngester,
    chunk_document_pages,
    chunk_text,
    extract_document,
    extract_docx,
    extract_pdf,
    extract_txt,
    get_embedding_text,
    ingest_document,
    ingest_manifest,
)


def load_document(file_path: str | Path) -> Document:
    """Load a raw document from the filesystem and extract text and metadata."""
    ingester = DocumentIngester()
    return ingester.load_document(str(file_path))


def chunk_document(
    document: Document,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Split a loaded document into smaller discrete chunks for indexing."""
    ingester = DocumentIngester()
    return ingester.chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def ingest_documents(
    directory_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """Batch ingest documents found in directory."""
    path = Path(directory_path)
    if (path / "manifest.csv").exists():
        return ingest_manifest(path / "manifest.csv", base_dir=path)
    all_chunks: list[DocumentChunk] = []
    ingester = DocumentIngester()
    for ext in ("*.pdf", "*.docx", "*.txt", "*.md"):
        for file in path.glob(ext):
            doc = ingester.load_document(str(file))
            all_chunks.extend(ingester.chunk_document(doc))
    return all_chunks


__all__ = [
    "load_document",
    "chunk_document",
    "ingest_documents",
    "DocumentIngester",
    "extract_document",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
    "chunk_text",
    "chunk_document_pages",
    "ingest_document",
    "ingest_manifest",
    "get_embedding_text",
]

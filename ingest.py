"""ingest.py

Root-level entry point exposing document ingestion and chunking components.
"""

from dev1.ingest import (
    DocumentIngester,
    chunk_document,
    chunk_document_pages,
    chunk_text,
    extract_document,
    extract_docx,
    extract_pdf,
    extract_txt,
    get_embedding_text,
    ingest_document,
    ingest_documents,
    ingest_manifest,
    load_document,
)

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

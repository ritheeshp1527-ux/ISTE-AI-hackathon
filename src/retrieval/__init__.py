"""Retrieval package containing ingestion, extraction, indexing, and permission-aware retrieval."""

from src.retrieval.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DocumentIndexer,
    MiniLMEmbeddingFunction,
    clear_collection,
    create_index,
    delete_doc,
    get_chunk,
    get_chunks_by_doc,
    get_collection_count,
    get_default_embedding_function,
    index_chunks,
    prepare_chunk_text_for_embedding,
    prepare_query_text_for_embedding,
    query_similar,
)
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
from src.retrieval.retrieve import (
    PermissionAwareRetriever,
    retrieve,
    vector_retrieve,
)

__all__ = [
    # Ingest
    "extract_document",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
    "chunk_text",
    "chunk_document_pages",
    "ingest_document",
    "ingest_manifest",
    "DocumentIngester",
    "get_embedding_text",
    # Index
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DocumentIndexer",
    "MiniLMEmbeddingFunction",
    "create_index",
    "index_chunks",
    "delete_doc",
    "get_collection_count",
    "get_chunk",
    "get_chunks_by_doc",
    "query_similar",
    "clear_collection",
    "prepare_chunk_text_for_embedding",
    "prepare_query_text_for_embedding",
    "get_default_embedding_function",
    # Retrieve
    "PermissionAwareRetriever",
    "retrieve",
    "vector_retrieve",
]

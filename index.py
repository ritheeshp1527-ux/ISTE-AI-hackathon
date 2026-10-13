"""index.py

Root-level entry point exposing Dev 1 indexing components.
"""

from dev1.index import (
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

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "create_index",
    "index_chunks",
    "delete_doc",
    "DocumentIndexer",
    "get_collection_count",
    "get_chunk",
    "get_chunks_by_doc",
    "query_similar",
    "clear_collection",
    "MiniLMEmbeddingFunction",
    "prepare_chunk_text_for_embedding",
    "prepare_query_text_for_embedding",
    "get_default_embedding_function",
]

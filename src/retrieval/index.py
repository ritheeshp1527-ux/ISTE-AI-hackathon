"""src/retrieval/index.py

ChromaDB indexing layer for the Document Q&A System.
Persists document chunks to a ChromaDB collection named 'kb' using cosine distance
and sentence-transformers/all-MiniLM-L6-v2 embeddings.
Enforces idempotent re-ingestion, scalar metadata conversion, and clean passage storage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from contracts import (
    DocumentChunk,
    IndexingContract,
    to_chroma_metadata,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME: str = "kb"
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")


# ============================================================================
# Embedding Function Wrapper
# ============================================================================

class EmbeddingFunctionProtocol(Protocol):
    """Protocol for embedding callables compatible with Chroma."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


class MiniLMEmbeddingFunction:
    """Wrapper for sentence-transformers/all-MiniLM-L6-v2 embeddings.

    Implements lazy model loading, L2 normalization for cosine similarity,
    and query preprocessing matching document chunk preprocessing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    @staticmethod
    def name() -> str:
        return "all-MiniLM-L6-v2"

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def default_space(self) -> str:
        return "cosine"

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> Any:
        return cls()

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Batch encode a list of texts into embeddings."""
        if not input:
            return []
        embeddings = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query text using identical preprocessing."""
        clean_query = prepare_query_text_for_embedding(query)
        result = self([clean_query])
        return result[0] if result else []


_GLOBAL_DEFAULT_EMBEDDING_FN: MiniLMEmbeddingFunction | None = None


def get_default_embedding_function() -> MiniLMEmbeddingFunction:
    """Get or create singleton instance of the default embedding function."""
    global _GLOBAL_DEFAULT_EMBEDDING_FN
    if _GLOBAL_DEFAULT_EMBEDDING_FN is None:
        _GLOBAL_DEFAULT_EMBEDDING_FN = MiniLMEmbeddingFunction()
    return _GLOBAL_DEFAULT_EMBEDDING_FN


# ============================================================================
# Preprocessing Conventions
# ============================================================================

def prepare_chunk_text_for_embedding(chunk: DocumentChunk) -> str:
    """Prepare text for vector embedding by prepending document title.

    The chunk's original passage text remains clean in the chunk.text field,
    while vector embeddings benefit from title context.
    """
    return chunk.embedding_text


def prepare_query_text_for_embedding(query: str) -> str:
    """Prepare query text for retrieval embedding."""
    return query.strip()


# ============================================================================
# ChromaDB Client & Collection Management
# ============================================================================

def resolve_persist_directory(persist_directory: str | Path | None = None) -> Path:
    """Resolve directory path for ChromaDB persistent storage."""
    if persist_directory is not None:
        p = Path(persist_directory)
    else:
        p = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def get_chroma_client(persist_directory: str | Path | None = None) -> ClientAPI:
    """Create or return a persistent ChromaDB client."""
    path = resolve_persist_directory(persist_directory)
    return chromadb.PersistentClient(path=str(path))


def normalize_collection_name(name: str) -> str:
    """Normalize collection name to satisfy ChromaDB's 3-512 character constraint.

    ChromaDB enforces a strict schema requiring collection names to be between 3 and 512
    characters from [a-zA-Z0-9._-]. When the agreed short name 'kb' is used, it is
    mapped to 'kb_docs' to prevent schema validation failures.
    """
    clean = str(name).strip()
    if len(clean) < 3:
        return f"{clean}_docs"
    return clean


def get_or_create_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
) -> Collection:
    """Get or create the Chroma collection configured for cosine distance."""
    client = get_chroma_client(persist_directory)
    norm_name = normalize_collection_name(collection_name)

    if embedding_fn is not None:
        return client.get_or_create_collection(
            name=norm_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fn,
        )

    try:
        return client.get_collection(name=norm_name)
    except Exception:
        emb_fn = get_default_embedding_function()
        return client.get_or_create_collection(
            name=norm_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=emb_fn,
        )


def create_index(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
) -> Collection:
    """Initialize persistent index collection."""
    return get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


# ============================================================================
# Document Deletion (Targeted by doc_id)
# ============================================================================

def delete_doc(
    doc_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
) -> int:
    """Delete all chunks belonging to a document from the collection.

    Leaves all unrelated documents completely intact.
    Returns the count of deleted chunks.
    """
    if not doc_id:
        return 0

    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_fn=embedding_fn,
    )

    # Lookup existing chunks by doc_id
    existing = collection.get(where={"doc_id": doc_id})
    existing_ids = existing.get("ids", [])
    if not existing_ids:
        return 0

    collection.delete(ids=existing_ids)
    logger.info("Deleted %d existing chunk(s) for doc_id='%s'", len(existing_ids), doc_id)
    return len(existing_ids)


# ============================================================================
# Idempotent Indexing
# ============================================================================

def index_chunks(
    chunks: list[DocumentChunk],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
    batch_size: int = 64,
) -> int:
    """Batch-embed and index document chunks into ChromaDB.

    Idempotency: Prior to indexing new chunks, all pre-existing chunks for the
    affected document IDs are removed to eliminate duplicate or stale chunks.

    Args:
        chunks: List of DocumentChunk instances to index.
        collection_name: Chroma collection name (defaults to 'kb').
        persist_directory: Path to persistent storage.
        embedding_fn: Optional embedding callable (uses all-MiniLM-L6-v2 by default).
        batch_size: Batch size for embeddings and upserts.

    Returns:
        Total count of newly indexed chunks.
    """
    if not chunks:
        return 0

    emb_function = embedding_fn if embedding_fn is not None else get_default_embedding_function()
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_fn=emb_function,
    )

    # 1. Idempotency Gate: Purge old chunks for all documents present in this batch
    affected_doc_ids = {c.doc_id for c in chunks if c.doc_id}
    for doc_id in affected_doc_ids:
        delete_doc(
            doc_id=doc_id,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_fn=emb_function,
        )

    # 2. Batch-embed and upsert chunks
    total_indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c.chunk_id for c in batch]
        documents = [c.text for c in batch]  # Clean original passage text
        metadatas = [c.to_chroma_metadata() for c in batch]  # Chroma scalar values

        # Preprocess text for embedding (prepends title to passage)
        embedding_inputs = [prepare_chunk_text_for_embedding(c) for c in batch]
        embeddings = emb_function(embedding_inputs)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        total_indexed += len(batch)

    logger.info(
        "Successfully indexed %d chunk(s) across %d doc(s) into collection '%s'",
        total_indexed,
        len(affected_doc_ids),
        collection_name,
    )
    return total_indexed


# ============================================================================
# Inspection & Query Interfaces
# ============================================================================

def get_collection_count(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
) -> int:
    """Return total number of chunks currently stored in the collection."""
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return collection.count()


def get_chunk(
    chunk_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
) -> dict[str, Any] | None:
    """Lookup a single chunk record by chunk_id."""
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    res = collection.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])
    if not res.get("ids"):
        return None
    return {
        "chunk_id": res["ids"][0],
        "document": res["documents"][0] if res.get("documents") else "",
        "metadata": res["metadatas"][0] if res.get("metadatas") else {},
    }


def get_chunks_by_doc(
    doc_id: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Lookup all chunks belonging to a doc_id."""
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    res = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    items: list[dict[str, Any]] = []
    if res.get("ids"):
        for i, c_id in enumerate(res["ids"]):
            items.append({
                "chunk_id": c_id,
                "document": res["documents"][i] if res.get("documents") else "",
                "metadata": res["metadatas"][i] if res.get("metadatas") else {},
            })
    return items


def query_similar(
    query_text: str,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
) -> list[dict[str, Any]]:
    """Query ChromaDB collection using cosine similarity."""
    emb_function = embedding_fn if embedding_fn is not None else get_default_embedding_function()
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_fn=emb_function,
    )

    clean_query = prepare_query_text_for_embedding(query_text)
    query_emb = emb_function([clean_query])[0]

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_emb],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    res = collection.query(**query_kwargs)
    results: list[dict[str, Any]] = []
    if res.get("ids") and res["ids"][0]:
        for idx in range(len(res["ids"][0])):
            results.append({
                "chunk_id": res["ids"][0][idx],
                "document": res["documents"][0][idx] if res.get("documents") else "",
                "metadata": res["metadatas"][0][idx] if res.get("metadatas") else {},
                "distance": res["distances"][0][idx] if res.get("distances") else 0.0,
            })
    return results


def clear_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
) -> None:
    """Purge all records in a collection."""
    norm_name = normalize_collection_name(collection_name)
    client = get_chroma_client(persist_directory)
    try:
        client.delete_collection(name=norm_name)
    except Exception:
        pass


# ============================================================================
# DocumentIndexer Class Implementation
# ============================================================================

class DocumentIndexer(IndexingContract):
    """Encapsulates ChromaDB collection indexing, persistence, and queries."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | Path | None = None,
        embedding_fn: Any = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_fn = embedding_fn

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        collection_name: str | None = None,
    ) -> int:
        target = collection_name or self.collection_name
        return index_chunks(
            chunks=chunks,
            collection_name=target,
            persist_directory=self.persist_directory,
            embedding_fn=self.embedding_fn,
        )

    def delete_doc(self, doc_id: str, collection_name: str | None = None) -> int:
        target = collection_name or self.collection_name
        return delete_doc(
            doc_id=doc_id,
            collection_name=target,
            persist_directory=self.persist_directory,
            embedding_fn=self.embedding_fn,
        )

    def get_count(self, collection_name: str | None = None) -> int:
        target = collection_name or self.collection_name
        return get_collection_count(
            collection_name=target,
            persist_directory=self.persist_directory,
        )

    def get_chunk(self, chunk_id: str, collection_name: str | None = None) -> dict[str, Any] | None:
        target = collection_name or self.collection_name
        return get_chunk(
            chunk_id=chunk_id,
            collection_name=target,
            persist_directory=self.persist_directory,
        )

    def get_chunks_by_doc(self, doc_id: str, collection_name: str | None = None) -> list[dict[str, Any]]:
        target = collection_name or self.collection_name
        return get_chunks_by_doc(
            doc_id=doc_id,
            collection_name=target,
            persist_directory=self.persist_directory,
        )

    def query_similar(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        target = collection_name or self.collection_name
        return query_similar(
            query_text=query_text,
            n_results=n_results,
            where=where,
            collection_name=target,
            persist_directory=self.persist_directory,
            embedding_fn=self.embedding_fn,
        )

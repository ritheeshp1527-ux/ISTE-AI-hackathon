"""tests/test_index.py

Tests for ChromaDB indexing layer (src/retrieval/index.py):
- Collection named 'kb' with cosine distance
- Persistent storage across client reopening
- Deterministic and scalar metadata preservation
- Targeted document deletion (delete_doc) without affecting unrelated docs
- Idempotent re-ingestion eliminating duplicate and stale chunks
- Querying with cosine similarity
- Fast mocked unit tests and real sentence-transformers integration test
"""

import hashlib
from pathlib import Path
from typing import Any
import pytest

from contracts import DocumentChunk
from src.retrieval.index import (
    DEFAULT_COLLECTION_NAME,
    DocumentIndexer,
    MiniLMEmbeddingFunction,
    clear_collection,
    create_index,
    delete_doc,
    get_chunk,
    get_chunks_by_doc,
    get_collection_count,
    index_chunks,
    query_similar,
)


# ============================================================================
# Deterministic Mock Embedding Function for Fast Unit Testing
# ============================================================================

class MockEmbeddingFunction:
    """Fast deterministic mock embedding producing 384-dim pseudo-embeddings."""

    @staticmethod
    def name() -> str:
        return "mock_embedding"

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict[str, Any]:
        return {"name": "mock"}

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def default_space(self) -> str:
        return "cosine"

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> Any:
        return cls()

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in input:
            # Generate deterministic pseudo-vector based on sha256
            h = hashlib.sha256(text.encode("utf-8")).digest()
            # 384 float dimensions normalized
            raw_floats = [((b / 255.0) * 2.0 - 1.0) for b in (h * 12)[:384]]
            norm = sum(x * x for x in raw_floats) ** 0.5 or 1.0
            embeddings.append([x / norm for x in raw_floats])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self(texts)

    def embed_query(self, query: str) -> list[float]:
        return self([query.strip()])[0]


@pytest.fixture
def mock_emb_fn() -> MockEmbeddingFunction:
    return MockEmbeddingFunction()


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Generate sample chunks across two documents."""
    c1 = DocumentChunk(
        doc_id="DOC-A",
        chunk_id="DOC-A_v1.0_c0",
        chunk_index=0,
        title="Architecture Overview",
        page=1,
        text="The system architecture utilizes a distributed vector store.",
        access_group="engineering",
        modified_at="2026-10-01",
        source_path="docs/arch.pdf",
        source_type="pdf",
        version="1.0",
        status="active",
        supersedes=None,
    )
    c2 = DocumentChunk(
        doc_id="DOC-A",
        chunk_id="DOC-A_v1.0_c1",
        chunk_index=1,
        title="Architecture Overview",
        page=2,
        text="Access control is enforced at retrieval time via role filtering.",
        access_group="engineering",
        modified_at="2026-10-01",
        source_path="docs/arch.pdf",
        source_type="pdf",
        version="1.0",
        status="active",
        supersedes=None,
    )
    c3 = DocumentChunk(
        doc_id="DOC-B",
        chunk_id="DOC-B_v1.0_c0",
        chunk_index=0,
        title="Finance Policy",
        page=0,
        text="All budget allocations must receive CFO sign-off.",
        access_group="finance",
        modified_at="2026-09-15",
        source_path="docs/fin.docx",
        source_type="docx",
        version="1.0",
        status="active",
        supersedes="DOC-B-OLD",
    )
    return [c1, c2, c3]


# ============================================================================
# Unit Tests
# ============================================================================

def test_index_chunks_and_collection_count(
    tmp_path: Path,
    mock_emb_fn: MockEmbeddingFunction,
    sample_chunks: list[DocumentChunk],
):
    """Test indexing chunks into 'kb' collection and verifying counts."""
    count = index_chunks(
        sample_chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    assert count == 3
    assert get_collection_count("kb", persist_directory=tmp_path) == 3


def test_chunk_lookup_and_metadata_preservation(
    tmp_path: Path,
    mock_emb_fn: MockEmbeddingFunction,
    sample_chunks: list[DocumentChunk],
):
    """Verify that chunk lookup returns original clean passage text and scalar metadata."""
    index_chunks(
        sample_chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )

    # Lookup c1
    record = get_chunk("DOC-A_v1.0_c0", collection_name="kb", persist_directory=tmp_path)
    assert record is not None
    assert record["chunk_id"] == "DOC-A_v1.0_c0"
    assert record["document"] == "The system architecture utilizes a distributed vector store."
    assert record["metadata"]["doc_id"] == "DOC-A"
    assert record["metadata"]["page"] == 1
    assert record["metadata"]["access_group"] == "engineering"
    assert record["metadata"]["supersedes"] == ""  # None mapped to empty string for Chroma

    # Lookup c3 with supersedes
    record_b = get_chunk("DOC-B_v1.0_c0", collection_name="kb", persist_directory=tmp_path)
    assert record_b is not None
    assert record_b["metadata"]["supersedes"] == "DOC-B-OLD"

    # Lookup chunks by doc_id
    doc_a_chunks = get_chunks_by_doc("DOC-A", collection_name="kb", persist_directory=tmp_path)
    assert len(doc_a_chunks) == 2


def test_delete_doc_avoids_deleting_unrelated_documents(
    tmp_path: Path,
    mock_emb_fn: MockEmbeddingFunction,
    sample_chunks: list[DocumentChunk],
):
    """Verify that delete_doc only deletes chunks for the target doc_id."""
    index_chunks(
        sample_chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    assert get_collection_count("kb", persist_directory=tmp_path) == 3

    # Delete only DOC-A (has 2 chunks)
    deleted_count = delete_doc("DOC-A", collection_name="kb", persist_directory=tmp_path)
    assert deleted_count == 2

    # DOC-B must still be present and intact
    assert get_collection_count("kb", persist_directory=tmp_path) == 1
    doc_b_chunks = get_chunks_by_doc("DOC-B", collection_name="kb", persist_directory=tmp_path)
    assert len(doc_b_chunks) == 1
    assert doc_b_chunks[0]["chunk_id"] == "DOC-B_v1.0_c0"

    # DOC-A must be completely gone
    assert get_chunks_by_doc("DOC-A", collection_name="kb", persist_directory=tmp_path) == []


def test_idempotent_reindexing_prevents_duplicates_and_stale_chunks(
    tmp_path: Path,
    mock_emb_fn: MockEmbeddingFunction,
    sample_chunks: list[DocumentChunk],
):
    """Verify re-indexing a document purges old chunks before adding new ones."""
    # Step 1: Initial indexing of DOC-A (2 chunks) and DOC-B (1 chunk) -> Total = 3
    index_chunks(
        sample_chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    assert get_collection_count("kb", persist_directory=tmp_path) == 3

    # Step 2: Re-ingest DOC-A with updated content: only 1 new chunk (e.g. document was edited/shortened)
    updated_doc_a = [
        DocumentChunk(
            doc_id="DOC-A",
            chunk_id="DOC-A_v2.0_c0",
            chunk_index=0,
            title="Architecture Overview v2",
            page=1,
            text="Updated unified architecture description.",
            access_group="engineering",
            modified_at="2026-10-12",
            source_path="docs/arch.pdf",
            source_type="pdf",
            version="2.0",
            status="active",
            supersedes="DOC-A_v1.0",
        )
    ]

    index_chunks(
        updated_doc_a,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )

    # Assert total count is now 2 (1 new chunk for DOC-A + 1 chunk for DOC-B)
    assert get_collection_count("kb", persist_directory=tmp_path) == 2

    # Assert old DOC-A chunks (v1.0_c0 and v1.0_c1) were completely deleted
    assert get_chunk("DOC-A_v1.0_c0", collection_name="kb", persist_directory=tmp_path) is None
    assert get_chunk("DOC-A_v1.0_c1", collection_name="kb", persist_directory=tmp_path) is None

    # Assert new chunk exists
    new_record = get_chunk("DOC-A_v2.0_c0", collection_name="kb", persist_directory=tmp_path)
    assert new_record is not None
    assert new_record["metadata"]["version"] == "2.0"
    assert new_record["metadata"]["title"] == "Architecture Overview v2"


def test_persistence_after_reopening(
    tmp_path: Path,
    mock_emb_fn: MockEmbeddingFunction,
    sample_chunks: list[DocumentChunk],
):
    """Verify data persists on disk and is accessible after creating a fresh client."""
    # Index with first indexer instance
    indexer_1 = DocumentIndexer(
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    indexer_1.index_chunks(sample_chunks)
    assert indexer_1.get_count() == 3

    del indexer_1

    # Re-open with new indexer instance on same directory
    indexer_2 = DocumentIndexer(
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    assert indexer_2.get_count() == 3
    chunk = indexer_2.get_chunk("DOC-A_v1.0_c0")
    assert chunk is not None
    assert chunk["document"] == "The system architecture utilizes a distributed vector store."


def test_query_similar_with_where_filter(
    tmp_path: Path,
    mock_emb_fn: MockEmbeddingFunction,
    sample_chunks: list[DocumentChunk],
):
    """Verify similarity search returns matches and respects metadata filtering."""
    index_chunks(
        sample_chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )

    # Query without filter
    results = query_similar(
        query_text="vector store architecture",
        n_results=2,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    assert len(results) <= 2
    assert "chunk_id" in results[0]
    assert "distance" in results[0]

    # Query with where filter on access_group
    fin_results = query_similar(
        query_text="budget sign-off",
        n_results=5,
        where={"access_group": "finance"},
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    assert len(fin_results) == 1
    assert fin_results[0]["chunk_id"] == "DOC-B_v1.0_c0"


# ============================================================================
# Real Model Integration Test
# ============================================================================

def test_real_minilm_model_integration(tmp_path: Path):
    """Integration test using actual sentence-transformers/all-MiniLM-L6-v2."""
    real_emb_fn = MiniLMEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    chunks = [
        DocumentChunk(
            doc_id="DOC-AI",
            chunk_id="DOC-AI_c0",
            chunk_index=0,
            title="Artificial Intelligence Guide",
            page=1,
            text="Machine learning models learn patterns and representations directly from training data.",
            access_group="public",
            modified_at="2026-10-01",
            source_path="ai.txt",
            source_type="txt",
            version="1.0",
            status="active",
        ),
        DocumentChunk(
            doc_id="DOC-COOK",
            chunk_id="DOC-COOK_c0",
            chunk_index=0,
            title="Culinary Recipes",
            page=1,
            text="Bake sourdough bread at 230 degrees Celsius in a preheated Dutch oven.",
            access_group="public",
            modified_at="2026-10-01",
            source_path="recipes.txt",
            source_type="txt",
            version="1.0",
            status="active",
        ),
    ]

    index_chunks(
        chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=real_emb_fn,
    )

    # Query for ML concepts
    results = query_similar(
        query_text="deep neural network training and algorithms",
        n_results=2,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=real_emb_fn,
    )
    assert len(results) == 2
    # The AI chunk must rank higher (smaller cosine distance) than the cooking chunk
    assert results[0]["chunk_id"] == "DOC-AI_c0"
    assert results[0]["distance"] < results[1]["distance"]

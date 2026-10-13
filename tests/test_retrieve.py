"""tests/test_retrieve.py

Tests for permission-aware vector retrieval (src/retrieval/retrieve.py):
- Reuses existing collection 'kb' and sentence-transformers model
- Access group filtering via where={"access_group": {"$in": groups}}
- Empty groups returns empty results (no unfiltered search)
- Ranked chunks with similarity = 1 - cosine_distance and original passage text
- Negative testing: lab-ops user never receives quality chunks even on direct identifier searches
- Missing matches, metadata preservation, and query error handling
"""

import hashlib
from pathlib import Path
from typing import Any
import pytest

from contracts import (
    DocumentChunk,
    RetrievalRequest,
    RetrievalResponse,
    UserContext,
)
from src.retrieval.index import (
    DEFAULT_COLLECTION_NAME,
    MiniLMEmbeddingFunction,
    index_chunks,
)
from src.retrieval.retrieve import (
    PermissionAwareRetriever,
    retrieve,
    vector_retrieve,
)


# ============================================================================
# Deterministic Mock Embedding Function for Fast Retrieval Unit Tests
# ============================================================================

class MockRetrievalEmbeddingFunction:
    """Fast deterministic mock embedding producing 384-dim pseudo-embeddings."""

    @staticmethod
    def name() -> str:
        return "mock_retrieval_embedding"

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
            h = hashlib.sha256(text.encode("utf-8")).digest()
            raw_floats = [((b / 255.0) * 2.0 - 1.0) for b in (h * 12)[:384]]
            norm = sum(x * x for x in raw_floats) ** 0.5 or 1.0
            embeddings.append([x / norm for x in raw_floats])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self(texts)

    def embed_query(self, query: str) -> list[float]:
        return self([query.strip()])[0]


@pytest.fixture
def mock_retrieval_emb_fn() -> MockRetrievalEmbeddingFunction:
    return MockRetrievalEmbeddingFunction()


@pytest.fixture
def seeded_db(tmp_path: Path, mock_retrieval_emb_fn: MockRetrievalEmbeddingFunction) -> Path:
    """Seed ChromaDB with documents across two distinct access groups: 'lab-ops' and 'quality'."""
    chunks = [
        # Lab-Ops documents
        DocumentChunk(
            doc_id="DOC-LAB-01",
            chunk_id="DOC-LAB-01_c0",
            chunk_index=0,
            title="Centrifuge Calibration SOP",
            page=1,
            text="Operating speed must not exceed 15,000 RPM. Balance rotor weights within 0.1g.",
            access_group="lab-ops",
            modified_at="2026-10-01",
            source_path="sop/centrifuge.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),
        DocumentChunk(
            doc_id="DOC-LAB-02",
            chunk_id="DOC-LAB-02_c0",
            chunk_index=0,
            title="Autoclave Sterilization Manual",
            page=2,
            text="Steam sterilization cycle requires 121 degrees Celsius at 15 PSI for 30 minutes.",
            access_group="lab-ops",
            modified_at="2026-10-02",
            source_path="sop/autoclave.docx",
            source_type="docx",
            version="1.0",
            status="active",
        ),
        # Quality documents
        DocumentChunk(
            doc_id="DOC-QUAL-01",
            chunk_id="DOC-QUAL-01_c0",
            chunk_index=0,
            title="Annual Quality Audit Findings",
            page=1,
            text="Critical non-conformance identified in sample tracking batch traceability logs.",
            access_group="quality",
            modified_at="2026-10-05",
            source_path="audit/2026_quality.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),
        DocumentChunk(
            doc_id="DOC-QUAL-02",
            chunk_id="DOC-QUAL-02_c0",
            chunk_index=0,
            title="CAPA Corrective Action Plan",
            page=1,
            text="Root cause analysis requires five-whys methodology and QA director sign-off.",
            access_group="quality",
            modified_at="2026-10-06",
            source_path="audit/capa.txt",
            source_type="txt",
            version="1.0",
            status="active",
        ),
    ]

    index_chunks(
        chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_retrieval_emb_fn,
    )
    return tmp_path


# ============================================================================
# Unit Tests
# ============================================================================

def test_lab_ops_user_never_receives_quality_chunks(
    seeded_db: Path,
    mock_retrieval_emb_fn: MockRetrievalEmbeddingFunction,
):
    """Negative Security Test: A lab-ops user must never receive quality-only chunks.

    Even when explicitly searching for quality terms or direct document IDs.
    """
    user_lab = UserContext(user_id="lab_technician", roles=["lab-ops"], groups=["lab-ops"])

    # 1. Search for generic terms
    resp1 = retrieve(
        query="equipment SOP and operational procedures",
        user_context=user_lab,
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )
    for r in resp1.results:
        assert r.chunk.access_group == "lab-ops"
        assert r.chunk.access_group != "quality"

    # 2. Direct identifier search attempting to access quality document
    resp2 = retrieve(
        query="DOC-QUAL-01 Annual Quality Audit Findings non-conformance",
        user_context=user_lab,
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )
    # Even though query contains exact text of DOC-QUAL-01, quality chunks must NOT appear
    for r in resp2.results:
        assert r.chunk.access_group == "lab-ops"
        assert "quality" not in r.chunk.access_group
        assert r.chunk.doc_id != "DOC-QUAL-01"


def test_quality_user_never_receives_lab_ops_chunks(
    seeded_db: Path,
    mock_retrieval_emb_fn: MockRetrievalEmbeddingFunction,
):
    """Negative Security Test: A quality user must never receive lab-ops chunks."""
    user_qual = UserContext(user_id="qa_auditor", roles=["quality"], groups=["quality"])

    # Attempt to search for centrifuge SOP
    resp = retrieve(
        query="Centrifuge Calibration SOP 15,000 RPM DOC-LAB-01",
        user_context=user_qual,
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )
    for r in resp.results:
        assert r.chunk.access_group == "quality"
        assert r.chunk.access_group != "lab-ops"
        assert r.chunk.doc_id != "DOC-LAB-01"


def test_empty_groups_never_falls_back_to_unfiltered_search(
    seeded_db: Path,
    mock_retrieval_emb_fn: MockRetrievalEmbeddingFunction,
):
    """Security Requirement: Empty groups must return an empty result, never unfiltered search."""
    # Empty list
    resp1 = retrieve(
        query="autoclave sterilization",
        groups=[],
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )
    assert len(resp1.results) == 0
    assert resp1.total_found == 0

    # UserContext with empty roles and groups
    anonymous_user = UserContext(user_id="anon", roles=[], groups=[])
    resp2 = retrieve(
        query="autoclave sterilization",
        user_context=anonymous_user,
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )
    assert len(resp2.results) == 0


def test_retrieval_response_structure_and_similarity_scoring(
    seeded_db: Path,
    mock_retrieval_emb_fn: MockRetrievalEmbeddingFunction,
):
    """Verify response structures, metadata preservation, and similarity score formula."""
    resp = retrieve(
        query="centrifuge speed balance",
        groups=["lab-ops"],
        top_k=2,
        mode="vector",
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )

    assert len(resp) >= 1
    top_hit = resp[0]

    # Verify score = 1 - cosine_distance
    assert hasattr(top_hit, "similarity")
    assert hasattr(top_hit, "distance")
    assert hasattr(top_hit, "score")
    expected_sim = round(max(0.0, min(1.0, 1.0 - top_hit.distance)), 6)
    assert top_hit.similarity == expected_sim
    assert top_hit.score == expected_sim

    # Verify original clean passage text is returned and belongs to lab-ops
    assert top_hit.text in [
        "Operating speed must not exceed 15,000 RPM. Balance rotor weights within 0.1g.",
        "Steam sterilization cycle requires 121 degrees Celsius at 15 PSI for 30 minutes.",
    ]
    # Verify metadata preservation (all 13 fields accessible)
    assert top_hit.chunk.access_group == "lab-ops"
    assert top_hit.chunk.doc_id in ["DOC-LAB-01", "DOC-LAB-02"]
    assert top_hit.chunk.version == "1.0"
    assert top_hit.chunk.status == "active"

    # Verify dictionary conversion
    dicts = resp.to_dicts()
    assert isinstance(dicts, list)
    assert dicts[0]["chunk_id"] == top_hit.chunk.chunk_id
    assert "metadata" in dicts[0]


def test_permission_aware_retriever_class(
    seeded_db: Path,
    mock_retrieval_emb_fn: MockRetrievalEmbeddingFunction,
):
    """Verify PermissionAwareRetriever class and contract interface."""
    retriever = PermissionAwareRetriever(
        collection_name="kb",
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )

    user = UserContext(user_id="u1", roles=["lab-ops"], groups=["lab-ops"])
    req = RetrievalRequest(query="rotor weight", user_context=user, top_k=5)
    resp = retriever.retrieve(req)

    assert len(resp) > 0
    for item in resp:
        assert item.chunk.access_group == "lab-ops"


def test_empty_or_whitespace_query_returns_empty(
    seeded_db: Path,
    mock_retrieval_emb_fn: MockRetrievalEmbeddingFunction,
):
    """Verify empty query string safely returns empty result."""
    resp = retrieve(
        query="   ",
        groups=["lab-ops"],
        persist_directory=seeded_db,
        embedding_fn=mock_retrieval_emb_fn,
    )
    assert len(resp) == 0


# ============================================================================
# Real Model Integration Test
# ============================================================================

def test_real_minilm_permission_aware_retrieval(tmp_path: Path):
    """End-to-end integration test with sentence-transformers/all-MiniLM-L6-v2.

    Seeds real embeddings for 'lab-ops' and 'quality' documents and verifies
    permission filtering under real semantic search.
    """
    real_emb_fn = MiniLMEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    chunks = [
        DocumentChunk(
            doc_id="DOC-LAB-01",
            chunk_id="DOC-LAB-01_c0",
            chunk_index=0,
            title="Bio-Safety Protocols",
            page=1,
            text="Personal protective equipment including gloves and lab coats are mandatory in containment zone 3.",
            access_group="lab-ops",
            modified_at="2026-10-01",
            source_path="safety.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),
        DocumentChunk(
            doc_id="DOC-QUAL-01",
            chunk_id="DOC-QUAL-01_c0",
            chunk_index=0,
            title="Quality Management Audit",
            page=1,
            text="Annual regulatory compliance inspection report detailing ISO 9001 certification standards.",
            access_group="quality",
            modified_at="2026-10-01",
            source_path="iso.pdf",
            source_type="pdf",
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

    # User 1: Lab-Ops
    user_lab = UserContext(user_id="lab_user", roles=["lab-ops"], groups=["lab-ops"])
    resp_lab = retrieve(
        query="ISO 9001 compliance standards and regulatory inspection",
        user_context=user_lab,
        persist_directory=tmp_path,
        embedding_fn=real_emb_fn,
    )
    # The query is about ISO 9001 (quality document), but user only has lab-ops access.
    # Therefore, NO quality chunks may be returned!
    for r in resp_lab.results:
        assert r.chunk.access_group == "lab-ops"
        assert r.chunk.access_group != "quality"

    # User 2: Quality
    user_qual = UserContext(user_id="qual_user", roles=["quality"], groups=["quality"])
    resp_qual = retrieve(
        query="ISO 9001 compliance standards and regulatory inspection",
        user_context=user_qual,
        persist_directory=tmp_path,
        embedding_fn=real_emb_fn,
    )
    assert len(resp_qual.results) >= 1
    assert resp_qual.results[0].chunk.doc_id == "DOC-QUAL-01"
    assert resp_qual.results[0].chunk.access_group == "quality"
    assert resp_qual.results[0].similarity > 0.5

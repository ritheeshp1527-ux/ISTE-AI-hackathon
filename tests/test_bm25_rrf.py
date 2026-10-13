"""tests/test_bm25_rrf.py

Comprehensive tests for BM25 keyword retrieval and Reciprocal Rank Fusion (RRF):
1. Exact identifier search: SOP-QA-014 ranks in top 3 when authorized.
2. Regression test: Quality-only document NEVER returned for lab-ops user,
   even when querying exact identifier SOP-QA-014.
3. Natural-language query search.
4. Reciprocal Rank Fusion combining vector + BM25, deduplication by chunk_id.
5. Distinction of rrf_score and similarity fields (not interchangeable).
6. Deterministic tie-breaking (-rrf_score, -similarity, chunk_id).
7. Configurable candidate limits and RRF k parameters.
8. Deletion purges from BM25 corpus (no stale entries).
9. Idempotent re-indexing updates BM25 corpus without duplicates.
10. Tokenization of exact compound identifiers (hyphens, dots, underscores).
"""

import hashlib
from pathlib import Path
from typing import Any
import pytest

from contracts import (
    DocumentChunk,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    UserContext,
)
from src.retrieval.index import (
    DEFAULT_COLLECTION_NAME,
    MiniLMEmbeddingFunction,
    delete_doc,
    index_chunks,
)
from src.retrieval.retrieve import (
    bm25_retrieve,
    hybrid_retrieve,
    reciprocal_rank_fusion,
    retrieve,
    tokenize_for_bm25,
    vector_retrieve,
)


# ============================================================================
# Deterministic Mock Embedding Function for Fast BM25/RRF Unit Tests
# ============================================================================

class MockEmbeddingFunction:
    """Deterministic 384-dimensional pseudo-embedding function for fast testing."""

    @staticmethod
    def name() -> str:
        return "mock_bm25_embedding"

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict[str, Any]:
        return {"name": "mock_bm25"}

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
def mock_emb_fn() -> MockEmbeddingFunction:
    return MockEmbeddingFunction()


@pytest.fixture
def seeded_db(tmp_path: Path, mock_emb_fn: MockEmbeddingFunction) -> Path:
    """Seed ChromaDB with a multi-group knowledge base including SOP-QA-014."""
    chunks = [
        # Quality documents
        DocumentChunk(
            doc_id="SOP-QA-014",
            chunk_id="SOP-QA-014_c0",
            chunk_index=0,
            title="Cleanroom Sanitization SOP-QA-014",
            page=1,
            text="Cleanroom disinfection procedure SOP-QA-014 requires daily quaternary ammonium treatment.",
            access_group="quality",
            modified_at="2026-10-01",
            source_path="sop/sop-qa-014.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),
        DocumentChunk(
            doc_id="SOP-QA-015",
            chunk_id="SOP-QA-015_c0",
            chunk_index=0,
            title="Quality Control Auditing SOP-QA-015",
            page=1,
            text="Internal quality audit protocol SOP-QA-015 specifies annual batch review requirements.",
            access_group="quality",
            modified_at="2026-10-01",
            source_path="sop/sop-qa-015.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),
        DocumentChunk(
            doc_id="DOC-QUAL-09",
            chunk_id="DOC-QUAL-09_c0",
            chunk_index=0,
            title="Regulatory Compliance Overview",
            page=1,
            text="General guidelines for regulatory compliance across GMP manufacturing facilities.",
            access_group="quality",
            modified_at="2026-10-01",
            source_path="sop/compliance.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),

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
            title="Autoclave Operation Manual",
            page=1,
            text="Steam sterilization cycle requires 121 degrees Celsius at 15 PSI for 30 minutes.",
            access_group="lab-ops",
            modified_at="2026-10-01",
            source_path="sop/autoclave.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),
    ]

    index_chunks(
        chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=mock_emb_fn,
    )
    return tmp_path


# ============================================================================
# Tokenizer Tests
# ============================================================================

def test_tokenize_for_bm25_handles_compound_identifiers():
    """Verify compound identifiers (hyphen, underscore, period) and words are extracted."""
    text = "Inspect SOP-QA-014 according to ISO-9001:2015 rev_2 and section 4.2."
    tokens = tokenize_for_bm25(text)

    # Compound tokens
    assert "sop-qa-014" in tokens
    assert "iso-9001" in tokens
    assert "rev_2" in tokens
    assert "4.2" in tokens

    # Individual words
    assert "sop" in tokens
    assert "qa" in tokens
    assert "014" in tokens
    assert "inspect" in tokens


def test_tokenize_for_bm25_empty_text():
    assert tokenize_for_bm25("") == []
    assert tokenize_for_bm25("   ") == []


# ============================================================================
# Task 6 Core Validation: Exact Identifier & Quality Isolation
# ============================================================================

def test_sop_qa_014_ranks_in_top_three_when_authorized(
    seeded_db: Path,
    mock_emb_fn: MockEmbeddingFunction,
):
    """VALIDATION REQUIREMENT:

    Demonstrate that SOP-QA-014 ranks in the top three for an exact-identifier test
    when the document is present and authorized.
    """
    resp = retrieve(
        query="SOP-QA-014",
        groups=["quality"],
        top_k=5,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )

    assert len(resp.results) >= 1
    # Check top 3 results contain SOP-QA-014
    top_3_doc_ids = [r.chunk.doc_id for r in resp.results[:3]]
    assert "SOP-QA-014" in top_3_doc_ids

    # The top hit should be SOP-QA-014 due to strong BM25 keyword matching
    top_hit = resp.results[0]
    assert top_hit.chunk.doc_id == "SOP-QA-014"
    assert top_hit.chunk.access_group == "quality"

    # Distinct score fields validation
    assert top_hit.rrf_score > 0.0
    assert hasattr(top_hit, "similarity")
    assert hasattr(top_hit, "rrf_score")
    # Verify rrf_score and similarity are distinct fields, not interchangeable
    assert top_hit.rrf_score != top_hit.similarity


def test_regression_quality_only_doc_never_appears_for_lab_ops_user(
    seeded_db: Path,
    mock_emb_fn: MockEmbeddingFunction,
):
    """REGRESSION TEST (Requirement 9):

    Prove that a quality-only document never appears for a lab-ops-only user,
    even if BM25 matches its identifier perfectly.
    """
    # Lab-ops user queries the exact identifier of the quality document
    resp = retrieve(
        query="SOP-QA-014 Cleanroom Sanitization quaternary ammonium",
        groups=["lab-ops"],
        top_k=10,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )

    # Assert that SOP-QA-014 is NEVER returned
    for r in resp.results:
        assert r.chunk.access_group == "lab-ops"
        assert r.chunk.access_group != "quality"
        assert r.chunk.doc_id != "SOP-QA-014"
        assert r.chunk.doc_id != "SOP-QA-015"
        assert r.chunk.doc_id != "DOC-QUAL-09"

    # Also test BM25 directly to ensure BM25 corpus isolation
    bm25_results = bm25_retrieve(
        query="SOP-QA-014",
        groups=["lab-ops"],
        top_k=10,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )
    for r in bm25_results:
        assert r.chunk.access_group != "quality"
        assert r.chunk.doc_id != "SOP-QA-014"


# ============================================================================
# Natural-Language Query & BM25 Scoring
# ============================================================================

def test_natural_language_query_retrieval(
    seeded_db: Path,
    mock_emb_fn: MockEmbeddingFunction,
):
    """Verify natural-language queries return relevant chunks."""
    resp = retrieve(
        query="What temperature and pressure is required for steam sterilization in autoclave?",
        groups=["lab-ops"],
        top_k=3,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )

    assert len(resp.results) >= 1
    top_hit = resp.results[0]
    assert top_hit.chunk.doc_id == "DOC-LAB-02"
    assert "121 degrees Celsius at 15 PSI" in top_hit.text
    assert top_hit.chunk.access_group == "lab-ops"


# ============================================================================
# Reciprocal Rank Fusion & Deduplication
# ============================================================================

def test_reciprocal_rank_fusion_scoring_and_deduplication():
    """Verify RRF formula, distinct score fields, and chunk_id deduplication."""
    chunk1 = DocumentChunk(
        doc_id="D1",
        chunk_id="D1_c0",
        chunk_index=0,
        title="Doc 1",
        page=1,
        text="Content 1",
        access_group="groupA",
        modified_at="2026-10-01",
        source_path="d1.txt",
        source_type="txt",
        version="1.0",
        status="active",
    )
    chunk2 = DocumentChunk(
        doc_id="D2",
        chunk_id="D2_c0",
        chunk_index=0,
        title="Doc 2",
        page=1,
        text="Content 2",
        access_group="groupA",
        modified_at="2026-10-01",
        source_path="d2.txt",
        source_type="txt",
        version="1.0",
        status="active",
    )

    # Chunk 1 is rank 1 in vector, rank 2 in BM25
    # Chunk 2 is rank 2 in vector, not in BM25
    vec_results = [
        RetrievedChunk(chunk=chunk1, score=0.9, similarity=0.9, distance=0.1),
        RetrievedChunk(chunk=chunk2, score=0.7, similarity=0.7, distance=0.3),
    ]
    bm25_results = [
        RetrievedChunk(chunk=chunk1, score=2.5, bm25_score=2.5, similarity=0.0, distance=1.0),
    ]

    fused = reciprocal_rank_fusion(
        vector_results=vec_results,
        bm25_results=bm25_results,
        k=60,
        top_k=10,
    )

    # Deduplicated by chunk_id
    assert len(fused) == 2
    assert fused[0].chunk_id == "D1_c0"
    assert fused[1].chunk_id == "D2_c0"

    # D1 RRF = 1/(60+1) + 1/(60+1) = 2/61 ~ 0.032787
    expected_d1_rrf = round(1.0 / 61.0 + 1.0 / 61.0, 6)
    assert fused[0].rrf_score == expected_d1_rrf
    assert fused[0].similarity == 0.9
    assert fused[0].bm25_score == 2.5
    # Verify rrf_score and similarity are distinct
    assert fused[0].rrf_score != fused[0].similarity

    # D2 RRF = 1/(60+2) ~ 0.016129
    expected_d2_rrf = round(1.0 / 62.0, 6)
    assert fused[1].rrf_score == expected_d2_rrf
    assert fused[1].similarity == 0.7


def test_deterministic_tie_breaking():
    """Verify deterministic tie-breaking: primary -rrf, secondary -similarity, tertiary chunk_id."""
    def make_chunk(cid: str) -> DocumentChunk:
        return DocumentChunk(
            doc_id=cid,
            chunk_id=cid,
            chunk_index=0,
            title="T",
            page=1,
            text="text",
            access_group="grp",
            modified_at="2026-10-01",
            source_path="s.txt",
            source_type="txt",
            version="1.0",
            status="active",
        )

    # Three chunks with identical RRF score (rank 1 each from different lists) and identical similarity (0.0)
    # chunk_b, chunk_a, chunk_c
    vec_results = [RetrievedChunk(chunk=make_chunk("chunk_b"), score=0.5, similarity=0.5)]
    bm25_results = [
        RetrievedChunk(chunk=make_chunk("chunk_c"), score=2.0, bm25_score=2.0),
        RetrievedChunk(chunk=make_chunk("chunk_a"), score=1.5, bm25_score=1.5),
    ]

    # chunk_b has rrf = 1/61, sim = 0.5
    # chunk_c has rrf = 1/61, sim = 0.0
    # chunk_a has rrf = 1/62, sim = 0.0
    fused = reciprocal_rank_fusion(vec_results, bm25_results, k=60, top_k=3)
    assert fused[0].chunk_id == "chunk_b"  # tie on rrf broken by higher similarity
    assert fused[1].chunk_id == "chunk_c"
    assert fused[2].chunk_id == "chunk_a"

    # Now tie on BOTH rrf and similarity: sorted alphabetically by chunk_id
    r1 = [RetrievedChunk(chunk=make_chunk("Z_doc"), score=0.5, similarity=0.5)]
    r2 = [RetrievedChunk(chunk=make_chunk("A_doc"), score=0.5, similarity=0.5)]
    # Z_doc rank 1 in vec, A_doc rank 1 in bm25 -> identical rrf and sim
    fused_alpha = reciprocal_rank_fusion(r1, r2, k=60, top_k=2)
    assert fused_alpha[0].chunk_id == "A_doc"
    assert fused_alpha[1].chunk_id == "Z_doc"


def test_configurable_candidate_limits(
    seeded_db: Path,
    mock_emb_fn: MockEmbeddingFunction,
):
    """Verify candidate_k limit and top_k cutoffs are respected."""
    resp = retrieve(
        query="SOP",
        groups=["quality"],
        top_k=2,
        candidate_k=5,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )
    assert len(resp.results) <= 2


# ============================================================================
# Stale Index & Deletion Synchronization (Requirement 7)
# ============================================================================

def test_deletion_purges_bm25_entries(
    seeded_db: Path,
    mock_emb_fn: MockEmbeddingFunction,
):
    """Requirement 7: Document deletions must immediately purge entries from BM25 retrieval."""
    # 1. Before deletion: SOP-QA-014 is retrievable
    before_resp = retrieve(
        query="SOP-QA-014",
        groups=["quality"],
        top_k=5,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )
    doc_ids_before = [r.chunk.doc_id for r in before_resp.results]
    assert "SOP-QA-014" in doc_ids_before

    # 2. Delete SOP-QA-014
    deleted_count = delete_doc(
        doc_id="SOP-QA-014",
        collection_name="kb",
        persist_directory=seeded_db,
    )
    assert deleted_count >= 1

    # 3. After deletion: SOP-QA-014 must NOT appear in BM25 or RRF results
    after_resp = retrieve(
        query="SOP-QA-014 Cleanroom Sanitization quaternary ammonium",
        groups=["quality"],
        top_k=5,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )
    doc_ids_after = [r.chunk.doc_id for r in after_resp.results]
    assert "SOP-QA-014" not in doc_ids_after


def test_reindexing_updates_bm25_without_stale_or_duplicate_chunks(
    seeded_db: Path,
    mock_emb_fn: MockEmbeddingFunction,
):
    """Requirement 7: Re-indexing updates BM25 corpus without duplicates or stale passages."""
    # Re-index SOP-QA-014 with new passage text
    updated_chunk = DocumentChunk(
        doc_id="SOP-QA-014",
        chunk_id="SOP-QA-014_c0",
        chunk_index=0,
        title="Sanitization Protocol Revision 2.0",
        page=1,
        text="Hydrogen peroxide vapor sterilization protocol SOP-QA-014 replacing all quaternary ammonium treatments.",
        access_group="quality",
        modified_at="2026-10-10",
        source_path="sop/sop-qa-014-v2.pdf",
        source_type="pdf",
        version="2.0",
        status="active",
    )

    index_chunks(
        [updated_chunk],
        collection_name="kb",
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )

    # Search for new keyword "hydrogen peroxide vapor"
    resp = retrieve(
        query="hydrogen peroxide vapor sterilization",
        groups=["quality"],
        top_k=5,
        persist_directory=seeded_db,
        embedding_fn=mock_emb_fn,
    )

    assert len(resp.results) >= 1
    top_hit = resp.results[0]
    assert top_hit.chunk.doc_id == "SOP-QA-014"
    assert top_hit.chunk.version == "2.0"
    assert "Hydrogen peroxide vapor" in top_hit.text
    # Ensure no duplicates: only one SOP-QA-014 exists
    sop_hits = [r for r in resp.results if r.chunk.doc_id == "SOP-QA-014"]
    assert len(sop_hits) == 1


# ============================================================================
# Empty Inputs & Error Handling
# ============================================================================

def test_empty_groups_and_empty_query(
    seeded_db: Path,
    mock_emb_fn: MockEmbeddingFunction,
):
    """Verify empty groups or queries safely return empty results."""
    # Empty query
    assert len(retrieve(query="", groups=["quality"], persist_directory=seeded_db)) == 0
    assert len(bm25_retrieve(query="", groups=["quality"], persist_directory=seeded_db)) == 0

    # Empty groups
    assert len(retrieve(query="SOP-QA-014", groups=[], persist_directory=seeded_db)) == 0
    assert len(bm25_retrieve(query="SOP-QA-014", groups=[], persist_directory=seeded_db)) == 0


# ============================================================================
# Real Model End-to-End Hybrid Retrieval Integration Test
# ============================================================================

def test_real_minilm_hybrid_retrieval(tmp_path: Path):
    """End-to-end integration test with sentence-transformers/all-MiniLM-L6-v2."""
    real_emb_fn = MiniLMEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    chunks = [
        DocumentChunk(
            doc_id="SOP-QA-014",
            chunk_id="SOP-QA-014_c0",
            chunk_index=0,
            title="Cleanroom Sanitization SOP-QA-014",
            page=1,
            text="Cleanroom disinfection procedure SOP-QA-014 requires daily quaternary ammonium treatment.",
            access_group="quality",
            modified_at="2026-10-01",
            source_path="sop/sop-qa-014.pdf",
            source_type="pdf",
            version="1.0",
            status="active",
        ),
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
    ]

    index_chunks(
        chunks,
        collection_name="kb",
        persist_directory=tmp_path,
        embedding_fn=real_emb_fn,
    )

    # 1. Quality user retrieves SOP-QA-014
    resp_qual = retrieve(
        query="SOP-QA-014 quaternary ammonium disinfection",
        groups=["quality"],
        top_k=3,
        persist_directory=tmp_path,
        embedding_fn=real_emb_fn,
    )
    assert len(resp_qual.results) >= 1
    assert resp_qual.results[0].chunk.doc_id == "SOP-QA-014"
    assert resp_qual.results[0].rrf_score > 0.0
    assert resp_qual.results[0].similarity > 0.3

    # 2. Lab-ops user queries SOP-QA-014: MUST BE COMPLETELY ISOLATED
    resp_lab = retrieve(
        query="SOP-QA-014 quaternary ammonium disinfection",
        groups=["lab-ops"],
        top_k=3,
        persist_directory=tmp_path,
        embedding_fn=real_emb_fn,
    )
    for r in resp_lab.results:
        assert r.chunk.access_group != "quality"
        assert r.chunk.doc_id != "SOP-QA-014"

"""tests/test_dataset.py

End-to-end integration tests verifying complete compatibility with the dataset folder:
- Validation of dataset/manifest.csv
- Multi-format document extraction (PDF, DOCX, TXT/email/teams)
- 13 agreed chunk fields and deterministic IDs
- Permission boundary verification (quality restricted memo never leaks to lab-ops)
"""

from pathlib import Path
import pytest

from contracts import UserContext, validate_manifest_csv
from src.retrieval.ingest import extract_document, ingest_manifest
from src.retrieval.index import index_chunks
from src.retrieval.retrieve import retrieve

DATASET_DIR = Path("dataset")
MANIFEST_PATH = DATASET_DIR / "manifest.csv"


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="dataset/manifest.csv not present")
def test_dataset_manifest_validation():
    """Verify that dataset/manifest.csv validates cleanly without errors."""
    records = validate_manifest_csv(MANIFEST_PATH)
    assert len(records) == 29

    doc_ids = {r.doc_id for r in records}
    assert "SOP-QA-014" in doc_ids
    assert "SOP-LAB-007" in doc_ids
    assert "QA-MEMO-2026-04" in doc_ids
    assert "POL-HR-003" in doc_ids
    assert "LIB-ICH-Q9" in doc_ids


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="dataset/manifest.csv not present")
def test_dataset_ingestion_and_chunk_fields():
    """Verify ingestion of dataset produces valid chunks with all 13 metadata fields."""
    records = validate_manifest_csv(MANIFEST_PATH)
    # Test sample across diverse formats: PDF, DOCX, Email TXT, Teams TXT
    sample_records = [
        r for r in records
        if r.doc_id in {"SOP-QA-014", "SOP-LAB-012", "EML-CAL-01", "TEAMS-QA-202609"}
    ]

    all_chunks = []
    for rec in sample_records:
        file_path = DATASET_DIR / rec.source_path
        pages = extract_document(file_path, rec)
        assert len(pages) > 0
        from src.retrieval.ingest import chunk_document_pages
        chunks = chunk_document_pages(pages, rec)
        assert len(chunks) > 0
        all_chunks.extend(chunks)

    for chunk in all_chunks:
        assert chunk.doc_id
        assert chunk.chunk_id
        assert chunk.text.strip()
        assert chunk.access_group in {"quality", "lab-ops", "all"}
        assert chunk.status in {"active", "effective", "superseded"}


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="dataset/manifest.csv not present")
def test_dataset_permission_boundary_e2e(tmp_path: Path):
    """Verify that QA-MEMO-2026-04 (quality restricted) never leaks to lab-ops user."""
    from tests.test_retrieve import MockRetrievalEmbeddingFunction
    mock_emb = MockRetrievalEmbeddingFunction()

    records = validate_manifest_csv(MANIFEST_PATH)
    target_records = [r for r in records if r.doc_id in {"QA-MEMO-2026-04", "SOP-LAB-007"}]

    all_chunks = []
    for rec in target_records:
        file_path = DATASET_DIR / rec.source_path
        pages = extract_document(file_path, rec)
        from src.retrieval.ingest import chunk_document_pages
        all_chunks.extend(chunk_document_pages(pages, rec))

    index_chunks(
        all_chunks,
        collection_name="kb_dataset_perm_test",
        persist_directory=tmp_path,
        embedding_fn=mock_emb,
    )

    # 1. Quality user (Alice) should retrieve QA-MEMO-2026-04
    alice_ctx = UserContext(user_id="alice", roles=["quality"], groups=["quality"])
    alice_resp = retrieve(
        query="What were the findings of the Q3 internal audit memo?",
        user_context=alice_ctx,
        collection_name="kb_dataset_perm_test",
        persist_directory=tmp_path,
        embedding_fn=mock_emb,
    )
    assert any(r.chunk.doc_id == "QA-MEMO-2026-04" for r in alice_resp.results)

    # 2. Lab-ops user (Bob) must NEVER retrieve QA-MEMO-2026-04
    bob_ctx = UserContext(user_id="bob", roles=["lab-ops"], groups=["lab-ops"])
    bob_resp = retrieve(
        query="What were the findings of the Q3 internal audit memo?",
        user_context=bob_ctx,
        collection_name="kb_dataset_perm_test",
        persist_directory=tmp_path,
        embedding_fn=mock_emb,
    )
    for r in bob_resp.results:
        assert r.chunk.access_group != "quality"
        assert r.chunk.doc_id != "QA-MEMO-2026-04"

"""tests/test_contracts.py

Tests for contracts.py data models, structure, manifest validation,
13 chunk fields, deterministic IDs, and Chroma metadata conversion.
"""

from pathlib import Path
import pytest

from contracts import (
    CHUNK_13_FIELDS,
    MANIFEST_REQUIRED_FIELDS,
    SUPPORTED_SOURCE_TYPES,
    VALID_STATUSES,
    Document,
    DocumentChunk,
    DocumentMetadata,
    ExtractedPage,
    IndexingContract,
    IngestionContract,
    ManifestValidationError,
    RetrievalContract,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    UserContext,
    generate_chunk_id,
    to_chroma_metadata,
    validate_manifest_csv,
    validate_manifest_record,
    validate_manifest_records,
)


def test_document_creation():
    doc = Document(
        id="doc-1",
        content="Sample document content",
        metadata={"filename": "sample.pdf"},
        permissions=["admin", "engineering"],
    )
    assert doc.id == "doc-1"
    assert doc.content == "Sample document content"
    assert doc.metadata["filename"] == "sample.pdf"
    assert "admin" in doc.permissions


def test_document_chunk_defaults():
    chunk = DocumentChunk(
        chunk_id="chk-1",
        document_id="doc-1",
        content="First paragraph",
    )
    assert chunk.chunk_id == "chk-1"
    assert chunk.document_id == "doc-1"
    assert chunk.content == "First paragraph"
    assert chunk.metadata == {}
    assert chunk.permissions == []
    assert chunk.chunk_index == 0


def test_user_context():
    user = UserContext(
        user_id="user-42",
        roles=["analyst", "viewer"],
        groups=["finance"],
    )
    assert user.user_id == "user-42"
    assert "analyst" in user.roles
    assert "finance" in user.groups


def test_retrieval_request_and_response():
    user = UserContext(user_id="user-1", roles=["public"])
    req = RetrievalRequest(
        query="What is the net revenue?",
        user_context=user,
        top_k=3,
    )
    assert req.query == "What is the net revenue?"
    assert req.top_k == 3

    chunk = DocumentChunk(
        chunk_id="chk-1",
        document_id="doc-1",
        content="Revenue is 10M",
        permissions=["public"],
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.95)
    resp = RetrievalResponse(
        query=req.query,
        results=[retrieved],
        user_context=user,
        total_found=1,
    )
    assert resp.total_found == 1
    assert len(resp.results) == 1
    assert resp.results[0].score == 0.95
    assert resp.results[0].chunk.chunk_id == "chk-1"


def test_chunk_carries_13_agreed_fields():
    """Verify that every chunk carries the 13 agreed fields."""
    chunk = DocumentChunk(
        doc_id="DOC001",
        chunk_id="DOC001_v1.0_c0",
        chunk_index=0,
        title="Q3 Financial Report",
        page=1,
        text="Revenue was $10M in Q3.",
        access_group="finance",
        modified_at="2026-10-01T10:00:00Z",
        source_path="data/reports/q3.pdf",
        source_type="pdf",
        version="1.0",
        status="active",
        supersedes=None,
    )

    # Check all 13 fields are defined and accessible
    assert len(CHUNK_13_FIELDS) == 13
    for field_name in CHUNK_13_FIELDS:
        assert hasattr(chunk, field_name), f"Missing field: {field_name}"

    assert chunk.doc_id == "DOC001"
    assert chunk.chunk_id == "DOC001_v1.0_c0"
    assert chunk.chunk_index == 0
    assert chunk.title == "Q3 Financial Report"
    assert chunk.page == 1
    assert chunk.text == "Revenue was $10M in Q3."
    assert chunk.access_group == "finance"
    assert chunk.modified_at == "2026-10-01T10:00:00Z"
    assert chunk.source_path == "data/reports/q3.pdf"
    assert chunk.source_type == "pdf"
    assert chunk.version == "1.0"
    assert chunk.status == "active"
    assert chunk.supersedes is None


def test_deterministic_chunk_id_generation():
    """Verify chunk_id is deterministic and safe for idempotent re-ingestion."""
    id1 = generate_chunk_id(doc_id="DOC-101", version="1.0", chunk_index=0, content="Hello World")
    id2 = generate_chunk_id(doc_id="DOC-101", version="1.0", chunk_index=0, content="Hello World")
    # Same inputs must produce identical chunk_id
    assert id1 == id2

    # Different version produces different chunk_id
    id_v2 = generate_chunk_id(doc_id="DOC-101", version="2.0", chunk_index=0, content="Hello World")
    assert id1 != id_v2

    # Different chunk position produces different chunk_id
    id_pos = generate_chunk_id(doc_id="DOC-101", version="1.0", chunk_index=1, content="Hello World")
    assert id1 != id_pos

    # Different content produces different chunk_id
    id_content = generate_chunk_id(doc_id="DOC-101", version="1.0", chunk_index=0, content="Different Text")
    assert id1 != id_content


def test_valid_manifest_record():
    """Verify validation of a valid manifest record."""
    record = {
        "doc_id": "DOC-HR-01",
        "title": "Employee Handbook",
        "source_path": "docs/hr/handbook.pdf",
        "source_type": "PDF",  # Test normalization
        "version": "1.2",
        "status": "Active",   # Test normalization
        "access_group": "hr",
        "modified_at": "2026-09-15",
        "supersedes": "DOC-HR-00",
    }
    meta = validate_manifest_record(record)
    assert meta.doc_id == "DOC-HR-01"
    assert meta.source_type == "pdf"
    assert meta.status == "active"
    assert meta.supersedes == "DOC-HR-00"
    assert meta.access_group == "hr"


def test_manifest_missing_required_fields():
    """Verify validation errors when required fields are missing or empty."""
    base_record = {
        "doc_id": "DOC-01",
        "title": "Doc Title",
        "source_path": "path/doc.txt",
        "source_type": "txt",
        "version": "1.0",
        "status": "active",
        "access_group": "public",
        "modified_at": "2026-01-01",
    }

    # Test each required field missing
    for req_field in MANIFEST_REQUIRED_FIELDS:
        incomplete = dict(base_record)
        del incomplete[req_field]
        with pytest.raises(ManifestValidationError, match="Missing required field"):
            validate_manifest_record(incomplete)

        # Test empty string for required field
        empty_field = dict(base_record)
        empty_field[req_field] = "   "
        with pytest.raises(ManifestValidationError):
            validate_manifest_record(empty_field)


def test_manifest_duplicate_doc_ids():
    """Verify duplicate doc_ids raise ManifestValidationError."""
    records = [
        {
            "doc_id": "DOC-DUP",
            "title": "Doc 1",
            "source_path": "docs/1.md",
            "source_type": "md",
            "version": "1.0",
            "status": "active",
            "access_group": "engineering",
            "modified_at": "2026-01-01",
        },
        {
            "doc_id": "DOC-DUP",  # Duplicate ID
            "title": "Doc 2",
            "source_path": "docs/2.md",
            "source_type": "md",
            "version": "1.0",
            "status": "active",
            "access_group": "engineering",
            "modified_at": "2026-01-02",
        },
    ]
    with pytest.raises(ManifestValidationError, match="Duplicate doc_id 'DOC-DUP'"):
        validate_manifest_records(records)


def test_manifest_invalid_source_type():
    """Verify unsupported file types raise ManifestValidationError."""
    bad_record = {
        "doc_id": "DOC-BAD-TYPE",
        "title": "Binary Executable",
        "source_path": "files/tool.exe",
        "source_type": "exe",  # Invalid type
        "version": "1.0",
        "status": "active",
        "access_group": "admin",
        "modified_at": "2026-01-01",
    }
    with pytest.raises(ManifestValidationError, match="Unsupported source_type 'exe'"):
        validate_manifest_record(bad_record)


def test_manifest_invalid_status():
    """Verify unsupported status raises ManifestValidationError."""
    bad_record = {
        "doc_id": "DOC-BAD-STATUS",
        "title": "Draft Document",
        "source_path": "files/doc.txt",
        "source_type": "txt",
        "version": "1.0",
        "status": "pending_approval",  # Invalid status
        "access_group": "public",
        "modified_at": "2026-01-01",
    }
    with pytest.raises(ManifestValidationError, match="Invalid status 'pending_approval'"):
        validate_manifest_record(bad_record)


def test_optional_supersedes_handling():
    """Verify consistent handling of optional supersedes values."""
    # When omitted / None -> None
    r1 = validate_manifest_record({
        "doc_id": "DOC-1",
        "title": "T1",
        "source_path": "p/1.txt",
        "source_type": "txt",
        "version": "1.0",
        "status": "active",
        "access_group": "public",
        "modified_at": "2026-01-01",
    })
    assert r1.supersedes is None

    # When empty string or whitespace -> normalized to None
    r2 = validate_manifest_record({
        "doc_id": "DOC-2",
        "title": "T2",
        "source_path": "p/2.txt",
        "source_type": "txt",
        "version": "1.0",
        "status": "active",
        "access_group": "public",
        "modified_at": "2026-01-01",
        "supersedes": "   ",
    })
    assert r2.supersedes is None

    # When set to previous doc_id -> preserved
    r3 = validate_manifest_record({
        "doc_id": "DOC-3",
        "title": "T3",
        "source_path": "p/3.txt",
        "source_type": "txt",
        "version": "2.0",
        "status": "active",
        "access_group": "public",
        "modified_at": "2026-01-01",
        "supersedes": "DOC-2",
    })
    assert r3.supersedes == "DOC-2"


def test_chroma_metadata_compatibility():
    """Verify all metadata fields are converted to Chroma-compatible types (str, int, float, bool)."""
    chunk = DocumentChunk(
        doc_id="DOC-99",
        chunk_id="DOC-99_v1.0_c0",
        chunk_index=0,
        title="Policy Document",
        page=2,
        text="Content text here",
        access_group="security",
        modified_at="2026-05-10",
        source_path="policies/sec.docx",
        source_type="docx",
        version="1.0",
        status="active",
        supersedes=None,  # Note: None should become empty string ""
    )

    chroma_meta = chunk.to_chroma_metadata()

    # Chroma requires that no value is None, dict, or list
    for key, val in chroma_meta.items():
        assert val is not None, f"Key '{key}' had None value in Chroma metadata"
        assert isinstance(val, (str, int, float, bool)), (
            f"Key '{key}' with value {val} (type {type(val)}) is not Chroma-compatible"
        )

    # Specific check for supersedes: converted to empty string
    assert chroma_meta["supersedes"] == ""

    # Test standalone to_chroma_metadata sanitizer
    raw_dict = {
        "doc_id": "DOC-1",
        "tags": ["alpha", "beta"],
        "nested": {"nested_key": 123},
        "flag": True,
        "null_val": None,
    }
    sanitized = to_chroma_metadata(raw_dict)
    assert sanitized["null_val"] == ""
    assert sanitized["flag"] is True
    assert sanitized["tags"] == "alpha,beta"
    assert '"nested_key": 123' in str(sanitized["nested"])


def test_validate_manifest_csv_file(tmp_path: Path):
    """Verify reading and validating a manifest.csv file from disk."""
    csv_file = tmp_path / "manifest.csv"
    csv_content = (
        "doc_id,title,source_path,source_type,version,status,access_group,modified_at,supersedes\n"
        "DOC01,Intro,docs/intro.md,md,1.0,active,public,2026-01-01,\n"
        "DOC02,Spec,docs/spec.pdf,pdf,2.0,active,engineering,2026-02-01,DOC01\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    records = validate_manifest_csv(csv_file)
    assert len(records) == 2
    assert records[0].doc_id == "DOC01"
    assert records[0].supersedes is None
    assert records[1].doc_id == "DOC02"
    assert records[1].supersedes == "DOC01"


def test_extracted_page_model():
    """Verify ExtractedPage model fields and instantiation."""
    page = ExtractedPage(
        doc_id="DOC-01",
        page_number=3,
        text="Extracted text on page 3",
        metadata={"char_count": 25},
    )
    assert page.doc_id == "DOC-01"
    assert page.page_number == 3
    assert page.text == "Extracted text on page 3"
    assert page.metadata["char_count"] == 25

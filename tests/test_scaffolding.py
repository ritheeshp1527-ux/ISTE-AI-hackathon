"""tests/test_scaffolding.py

Validates scaffolding for Dev 1 components:
- Modules are cleanly importable.
- Classes and functions exist with expected signatures.
- Placeholders raise NotImplementedError as expected during scaffolding.
"""

import pytest
from contracts import Document, DocumentChunk, UserContext
import dev1
from dev1.ingest import DocumentIngester, chunk_document, ingest_documents, load_document
from dev1.index import DocumentIndexer, create_index, index_chunks
from dev1.retrieve import PermissionAwareRetriever, retrieve

# Also verify root imports work
import ingest
import index
import retrieve as root_retrieve


def test_dev1_imports_and_exports():
    """Verify package symbols are exported correctly."""
    assert hasattr(dev1, "load_document")
    assert hasattr(dev1, "chunk_document")
    assert hasattr(dev1, "create_index")
    assert hasattr(dev1, "index_chunks")
    assert hasattr(dev1, "retrieve")
    assert hasattr(dev1, "PermissionAwareRetriever")


def test_root_reexports():
    """Verify root modules forward to dev1 implementations."""
    assert ingest.load_document is load_document
    assert index.create_index is create_index
    assert root_retrieve.retrieve is retrieve


def test_ingest_implementation_handles_missing_file():
    """Verify ingest functions validate file existence."""
    with pytest.raises(FileNotFoundError):
        load_document("nonexistent_dummy.pdf")


def test_index_implementation_initialized(tmp_path):
    """Verify index functions are implemented."""
    indexer = DocumentIndexer(persist_directory=tmp_path)
    assert indexer.get_count() == 0


def test_retrieve_implementation_initialized(tmp_path):
    """Verify retrieve functions are implemented."""
    user = UserContext(user_id="u1", roles=["user"])
    res = retrieve(query="test query", user_context=user, persist_directory=tmp_path)
    assert len(res) == 0

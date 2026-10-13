"""contracts.py

Shared data models, manifest schemas, and contract interfaces for the document Q&A system.
Defines types exchanged across Dev 1 (Ingestion, Indexing, Retrieval)
and other system components (Dev 2 generation, Dev 3 API/UI).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# Constants & Enums
# ============================================================================

SUPPORTED_SOURCE_TYPES: set[str] = {"pdf", "docx", "txt", "md"}

VALID_STATUSES: set[str] = {"active", "archived", "deprecated", "draft"}

MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "doc_id",
    "title",
    "source_path",
    "source_type",
    "version",
    "status",
    "access_group",
    "modified_at",
)

CHUNK_13_FIELDS: tuple[str, ...] = (
    "doc_id",
    "chunk_id",
    "chunk_index",
    "title",
    "page",
    "text",
    "access_group",
    "modified_at",
    "source_path",
    "source_type",
    "version",
    "status",
    "supersedes",
)


UNKNOWN_PAGE_NUMBER: int = 0


# ============================================================================
# Exceptions
# ============================================================================

class ManifestValidationError(ValueError):
    """Raised when manifest validation fails (missing fields, duplicates, invalid types)."""
    pass


class EmptyExtractionError(ValueError):
    """Raised when text extraction from a file yields empty content (e.g. scanned PDF without OCR)."""
    pass


# ============================================================================
# Manifest & Document Metadata
# ============================================================================

class DocumentMetadata(BaseModel):
    """Represents validated document metadata from manifest.csv."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    doc_id: str = Field(description="Unique identifier for the document")
    title: str = Field(description="Document title")
    source_path: str = Field(description="Relative or absolute path to document file")
    source_type: str = Field(description="File extension/type (pdf, docx, txt, md)")
    version: str = Field(default="1.0", description="Document version string")
    status: str = Field(default="active", description="Lifecycle status (active, archived, deprecated, draft)")
    access_group: str = Field(default="public", description="Permission group required for access")
    modified_at: str = Field(description="Last modified timestamp or ISO date")
    supersedes: str | None = Field(default=None, description="Optional doc_id that this version replaces")

    @field_validator("doc_id", "title", "source_path", "modified_at", mode="before")
    @classmethod
    def validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        if v is None or not str(v).strip():
            raise ManifestValidationError(f"Field '{info.field_name}' must be a non-empty string.")
        return str(v).strip()

    @field_validator("source_type", mode="before")
    @classmethod
    def validate_source_type(cls, v: Any) -> str:
        if v is None:
            raise ManifestValidationError("Field 'source_type' cannot be None.")
        norm = str(v).strip().lower().lstrip(".")
        if norm not in SUPPORTED_SOURCE_TYPES:
            raise ManifestValidationError(
                f"Unsupported source_type '{v}'. Must be one of: {sorted(SUPPORTED_SOURCE_TYPES)}"
            )
        return norm

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> str:
        if v is None:
            raise ManifestValidationError("Field 'status' cannot be None.")
        norm = str(v).strip().lower()
        if norm not in VALID_STATUSES:
            raise ManifestValidationError(
                f"Invalid status '{v}'. Must be one of: {sorted(VALID_STATUSES)}"
            )
        return norm

    @field_validator("access_group", mode="before")
    @classmethod
    def validate_access_group(cls, v: Any) -> str:
        if v is None or not str(v).strip():
            raise ManifestValidationError("Field 'access_group' must be a non-empty string.")
        return str(v).strip()

    @field_validator("supersedes", mode="before")
    @classmethod
    def normalize_supersedes(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


ManifestRecord = DocumentMetadata


# ============================================================================
# Document & Extracted Page Models
# ============================================================================

class Document(BaseModel):
    """Represents a raw or loaded document prior to chunking."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    id: str
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)


class ExtractedPage(BaseModel):
    """Represents a single extracted page from a document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    page_number: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Document Chunk (The 13 Agreed Fields + Chroma Metadata Compatibility)
# ============================================================================

def generate_chunk_id(
    doc_id: str,
    version: str,
    chunk_index: int,
    content: str | None = None,
) -> str:
    """Generate a deterministic chunk_id based on doc_id, version, chunk_index, and content.

    Ensures that re-ingesting the exact same document version and chunk position/content
    yields the identical chunk_id, enabling idempotent replacements in vector stores.
    """
    clean_doc = str(doc_id).strip()
    clean_ver = str(version).strip().replace(" ", "_")
    if content:
        content_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:8]
        return f"{clean_doc}_v{clean_ver}_c{chunk_index}_{content_hash}"
    return f"{clean_doc}_v{clean_ver}_c{chunk_index}"


class DocumentChunk(BaseModel):
    """Represents a discrete text chunk carrying all 13 agreed metadata fields.

    Agreed Fields:
      1. doc_id
      2. chunk_id
      3. chunk_index
      4. title
      5. page
      6. text
      7. access_group
      8. modified_at
      9. source_path
      10. source_type
      11. version
      12. status
      13. supersedes
    """

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    # 13 Agreed Fields
    doc_id: str = Field(default="", description="Parent document identifier")
    chunk_id: str = Field(default="", description="Deterministic chunk identifier")
    chunk_index: int = Field(default=0, description="0-indexed position within document")
    title: str = Field(default="", description="Document title")
    page: int = Field(default=1, description="Source page number (1-indexed)")
    text: str = Field(default="", description="Extracted text chunk content")
    access_group: str = Field(default="", description="Access control group")
    modified_at: str = Field(default="", description="Modification timestamp or date")
    source_path: str = Field(default="", description="Source file path")
    source_type: str = Field(default="", description="Source file extension/type")
    version: str = Field(default="1.0", description="Document version")
    status: str = Field(default="active", description="Document lifecycle status")
    supersedes: str | None = Field(default=None, description="Optional doc_id superseded by parent document")

    # Compatibility fields with Task 1 signatures
    document_id: str = Field(default="")
    content: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sync_compatibility_and_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync doc_id <-> document_id
            if "document_id" in data and not data.get("doc_id"):
                data["doc_id"] = data["document_id"]
            elif "doc_id" in data and not data.get("document_id"):
                data["document_id"] = data["doc_id"]

            # Sync text <-> content
            if "content" in data and not data.get("text"):
                data["text"] = data["content"]
            elif "text" in data and not data.get("content"):
                data["content"] = data["text"]

            # Sync access_group <-> permissions
            if "access_group" in data and data["access_group"] and not data.get("permissions"):
                data["permissions"] = [data["access_group"]]
            elif "permissions" in data and data["permissions"] and not data.get("access_group"):
                data["access_group"] = data["permissions"][0]

            # Auto-generate chunk_id if missing and doc_id present
            if not data.get("chunk_id") and data.get("doc_id"):
                data["chunk_id"] = generate_chunk_id(
                    doc_id=data["doc_id"],
                    version=data.get("version", "1.0"),
                    chunk_index=data.get("chunk_index", 0),
                    content=data.get("text") or data.get("content"),
                )

            # Normalize supersedes
            if "supersedes" in data:
                val = data["supersedes"]
                if val is not None and not str(val).strip():
                    data["supersedes"] = None
        return data

    def to_chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Convert chunk fields into strictly Chroma-compatible primitive metadata types.

        ChromaDB only accepts str, int, float, and bool metadata values.
        None values (such as optional supersedes) are converted to empty strings.
        """
        return {
            "doc_id": str(self.doc_id),
            "chunk_id": str(self.chunk_id),
            "chunk_index": int(self.chunk_index),
            "title": str(self.title),
            "page": int(self.page),
            "access_group": str(self.access_group),
            "modified_at": str(self.modified_at),
            "source_path": str(self.source_path),
            "source_type": str(self.source_type),
            "version": str(self.version),
            "status": str(self.status),
            "supersedes": str(self.supersedes or ""),
        }

    @property
    def embedding_text(self) -> str:
        """Text used for vector embeddings (prepends document title to passage text).

        The .text attribute retains clean original passage text, while .embedding_text
        combines title and passage to give embeddings richer document context.
        """
        if self.title:
            return f"{self.title}\n\n{self.text}"
        return self.text


Chunk = DocumentChunk


def get_embedding_text(chunk: DocumentChunk) -> str:
    """Return text for embedding generation (prepends title to passage text)."""
    return chunk.embedding_text


def to_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Sanitize any metadata dictionary to guarantee Chroma-compatible primitive types."""
    sanitized: dict[str, str | int | float | bool] = {}
    for k, v in metadata.items():
        if v is None:
            sanitized[k] = ""
        elif isinstance(v, bool):
            sanitized[k] = v
        elif isinstance(v, (int, float, str)):
            sanitized[k] = v
        elif isinstance(v, (list, tuple, set)):
            sanitized[k] = ",".join(str(item) for item in v)
        elif isinstance(v, dict):
            sanitized[k] = json.dumps(v)
        else:
            sanitized[k] = str(v)
    return sanitized


# ============================================================================
# User Context & Retrieval Models
# ============================================================================

class UserContext(BaseModel):
    """Represents the requesting user's identity and permissions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """A document chunk retrieved in response to a query, paired with its relevance score."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: DocumentChunk
    score: float = 0.0
    similarity: float = 0.0
    distance: float = 0.0
    rrf_score: float = 0.0
    bm25_score: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def sync_scores(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Only sync score -> similarity if neither bm25_score nor rrf_score is specified
            if (
                "score" in data
                and "similarity" not in data
                and "bm25_score" not in data
                and "rrf_score" not in data
            ):
                data["similarity"] = data["score"]
            elif "similarity" in data and "score" not in data:
                data["score"] = data.get("rrf_score", data["similarity"])
        return data

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def access_group(self) -> str:
        return self.chunk.access_group

    def __getitem__(self, item: str) -> Any:
        if item in ("chunk_id", "id"):
            return self.chunk.chunk_id
        if item == "text":
            return self.chunk.text
        if item == "score":
            return self.score
        if item == "similarity":
            return self.similarity
        if item == "distance":
            return self.distance
        if item == "rrf_score":
            return self.rrf_score
        if item == "bm25_score":
            return self.bm25_score
        if item == "chunk":
            return self.chunk
        if item == "metadata":
            return self.chunk.to_chroma_metadata()
        if hasattr(self.chunk, item):
            return getattr(self.chunk, item)
        raise KeyError(item)


RetrievalResult = RetrievedChunk


class RetrievalRequest(BaseModel):
    """Parameters for a permission-aware document retrieval query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    user_context: UserContext
    top_k: int = 5


class RetrievalResponse(BaseModel):
    """Results returned from a permission-aware document retrieval query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    results: list[RetrievedChunk] = Field(default_factory=list)
    user_context: UserContext | None = None
    total_found: int = 0

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> RetrievedChunk:
        return self.results[idx]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "text": r.chunk.text,
                "score": r.score,
                "similarity": r.similarity,
                "distance": r.distance,
                "rrf_score": r.rrf_score,
                "bm25_score": r.bm25_score,
                "metadata": r.chunk.to_chroma_metadata(),
            }
            for r in self.results
        ]


# ============================================================================
# Manifest Validation Functions
# ============================================================================

def validate_manifest_record(record: dict[str, Any] | DocumentMetadata) -> DocumentMetadata:
    """Validate a single manifest record dictionary against DocumentMetadata rules."""
    if isinstance(record, DocumentMetadata):
        return record

    if not isinstance(record, dict):
        raise ManifestValidationError(f"Manifest record must be a dict, got {type(record).__name__}.")

    # Check for required fields present in record
    for field_name in MANIFEST_REQUIRED_FIELDS:
        if field_name not in record or record[field_name] is None or not str(record[field_name]).strip():
            raise ManifestValidationError(f"Missing required field: '{field_name}'.")

    try:
        return DocumentMetadata(**record)
    except Exception as e:
        if isinstance(e, ManifestValidationError):
            raise
        raise ManifestValidationError(str(e)) from e


def validate_manifest_records(
    records: list[dict[str, Any]] | list[DocumentMetadata],
) -> list[DocumentMetadata]:
    """Validate a list of manifest records, enforcing unique doc_ids and field schemas."""
    validated: list[DocumentMetadata] = []
    seen_doc_ids: set[str] = set()

    for idx, rec in enumerate(records, start=1):
        validated_rec = validate_manifest_record(rec)
        doc_id = validated_rec.doc_id
        if doc_id in seen_doc_ids:
            raise ManifestValidationError(
                f"Duplicate doc_id '{doc_id}' found in manifest at entry #{idx}."
            )
        seen_doc_ids.add(doc_id)
        validated.append(validated_rec)

    return validated


def validate_manifest_csv(csv_path: str | Path) -> list[DocumentMetadata]:
    """Parse and validate a manifest.csv file from disk."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {csv_path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ManifestValidationError(f"Manifest CSV is empty: {csv_path}")

        # Check required columns present in header
        missing_columns = [col for col in MANIFEST_REQUIRED_FIELDS if col not in reader.fieldnames]
        if missing_columns:
            raise ManifestValidationError(
                f"Manifest CSV missing required header columns: {missing_columns}"
            )

        rows = list(reader)

    return validate_manifest_records(rows)


# ============================================================================
# Protocol Contracts
# ============================================================================

class IngestionContract(Protocol):
    """Interface contract for document ingestion operations."""

    def load_document(self, file_path: str) -> Document:
        """Load a document from disk or raw source."""
        ...

    def chunk_document(
        self,
        document: Document,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[DocumentChunk]:
        """Split a document into discrete chunks."""
        ...


class IndexingContract(Protocol):
    """Interface contract for vector indexing operations."""

    def index_chunks(
        self,
        chunks: list[DocumentChunk],
        collection_name: str = "documents",
    ) -> int:
        """Index chunks into vector storage. Returns count of chunks indexed."""
        ...


class RetrievalContract(Protocol):
    """Interface contract for permission-aware retrieval operations."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Execute a permission-filtered vector retrieval query."""
        ...

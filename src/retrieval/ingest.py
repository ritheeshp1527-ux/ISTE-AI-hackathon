"""src/retrieval/ingest.py

Document extraction and chunking pipeline for the Document Q&A System.
Reads documents validated against manifest.csv and produces discrete chunks
with all 13 agreed metadata fields, 1-based PDF page numbers, and unknown-page
conventions for DOCX and TXT.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import docx
import pymupdf as fitz

from contracts import (
    CHUNK_13_FIELDS,
    UNKNOWN_PAGE_NUMBER,
    Document,
    DocumentChunk,
    DocumentMetadata,
    EmptyExtractionError,
    ExtractedPage,
    IngestionContract,
    generate_chunk_id,
    get_embedding_text,
    validate_manifest_csv,
    validate_manifest_record,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Format-Specific Extractors
# ============================================================================

def extract_pdf(file_path: str | Path, doc_id: str) -> list[ExtractedPage]:
    """Extract text page-by-page from a PDF using PyMuPDF.

    Preserves 1-based source page numbers.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    pages: list[ExtractedPage] = []
    doc = fitz.open(str(path))
    try:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append(
                ExtractedPage(
                    doc_id=doc_id,
                    page_number=idx,  # 1-based page index
                    text=text,
                    metadata={"page_index": idx - 1, "source": str(path)},
                )
            )
    finally:
        doc.close()

    return pages


def extract_docx(file_path: str | Path, doc_id: str) -> list[ExtractedPage]:
    """Extract text, paragraphs, headings, and tables from a DOCX file using python-docx.

    Uses UNKNOWN_PAGE_NUMBER (0) to denote unpaged document flow.
    Preserves heading prefixes and table row/column structure with units.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    doc = docx.Document(str(path))
    content_blocks: list[str] = []

    # Iterate over block-level body elements (paragraphs and tables)
    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if tag == "p":
            p = docx.text.paragraph.Paragraph(element, doc)
            text = p.text.strip()
            if not text:
                continue
            style_name = p.style.name if p.style else ""
            if "Heading 1" in style_name:
                content_blocks.append(f"# {text}")
            elif "Heading 2" in style_name:
                content_blocks.append(f"## {text}")
            elif "Heading 3" in style_name:
                content_blocks.append(f"### {text}")
            elif "List" in style_name or "Bullet" in style_name:
                content_blocks.append(f"- {text}")
            else:
                content_blocks.append(text)
        elif tag == "tbl":
            tbl = docx.table.Table(element, doc)
            table_lines: list[str] = []
            for row in tbl.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(row_cells):
                    table_lines.append("| " + " | ".join(row_cells) + " |")
            if table_lines:
                content_blocks.append("\n".join(table_lines))

    combined_text = "\n\n".join(content_blocks).strip()
    return [
        ExtractedPage(
            doc_id=doc_id,
            page_number=UNKNOWN_PAGE_NUMBER,
            text=combined_text,
            metadata={"source": str(path), "unpaged": True},
        )
    ]


def extract_txt(file_path: str | Path, doc_id: str) -> list[ExtractedPage]:
    """Extract plain text from TXT or MD files.

    Uses UNKNOWN_PAGE_NUMBER (0) to denote unpaged document flow.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    content = ""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            content = path.read_text(encoding=enc).strip()
            break
        except UnicodeDecodeError:
            continue

    return [
        ExtractedPage(
            doc_id=doc_id,
            page_number=UNKNOWN_PAGE_NUMBER,
            text=content,
            metadata={"source": str(path), "unpaged": True},
        )
    ]


def extract_document(
    file_path: str | Path,
    metadata: DocumentMetadata,
    raise_on_empty: bool = False,
) -> list[ExtractedPage]:
    """Extract pages/text from a document based on its source_type in metadata.

    Reports empty extraction clearly and avoids silently returning empty content.
    """
    path = Path(file_path)
    source_type = metadata.source_type.lower().lstrip(".")

    if source_type == "pdf":
        pages = extract_pdf(path, metadata.doc_id)
    elif source_type == "docx":
        pages = extract_docx(path, metadata.doc_id)
    elif source_type in ("txt", "md"):
        pages = extract_txt(path, metadata.doc_id)
    else:
        raise ValueError(f"Unsupported source type '{source_type}' for {path}")

    total_chars = sum(len(p.text.strip()) for p in pages)
    if total_chars == 0:
        msg = (
            f"Empty text extracted from {path} (doc_id='{metadata.doc_id}'). "
            f"The file is empty or is a scanned PDF requiring OCR."
        )
        logger.warning(msg)
        if raise_on_empty:
            raise EmptyExtractionError(msg)

    return pages


# ============================================================================
# Chunking Logic (150–250 words, ~30 words overlap)
# ============================================================================

def chunk_text(
    text: str,
    target_min_words: int = 150,
    target_max_words: int = 250,
    overlap_words: int = 30,
) -> list[str]:
    """Split text into chunks of approximately 150–250 words with ~30 words of overlap.

    Prefers paragraph, table, and heading boundaries before splitting on sentences.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    words_in_text = clean_text.split()
    if len(words_in_text) <= target_max_words:
        return [clean_text]

    # Split into structural blocks by double newlines (paragraphs, headings, tables)
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", clean_text) if b.strip()]

    # If any single block exceeds max words, break it down by sentences
    units: list[str] = []
    for block in raw_blocks:
        b_words = block.split()
        if len(b_words) <= target_max_words:
            units.append(block)
        else:
            # Split block on sentence boundaries
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
            sub_accum: list[str] = []
            sub_count = 0
            for s in sentences:
                s_count = len(s.split())
                if sub_count + s_count > target_max_words and sub_accum:
                    units.append(" ".join(sub_accum))
                    sub_accum = [s]
                    sub_count = s_count
                else:
                    sub_accum.append(s)
                    sub_count += s_count
            if sub_accum:
                units.append(" ".join(sub_accum))

    # Assemble units into chunks of target size with overlap
    chunks: list[str] = []
    current_units: list[str] = []
    current_word_count = 0

    for unit in units:
        unit_words = len(unit.split())
        if current_word_count + unit_words <= target_max_words:
            current_units.append(unit)
            current_word_count += unit_words
        else:
            # Chunk is ready
            if current_word_count >= target_min_words or current_word_count > 0:
                chunk_str = "\n\n".join(current_units).strip()
                chunks.append(chunk_str)

                # Extract overlap from end of current chunk
                all_words = chunk_str.split()
                if len(all_words) > overlap_words:
                    overlap_phrase = " ".join(all_words[-overlap_words:])
                else:
                    overlap_phrase = " ".join(all_words)

                current_units = [overlap_phrase, unit] if overlap_phrase else [unit]
                current_word_count = len(overlap_phrase.split()) + unit_words
            else:
                current_units.append(unit)
                current_word_count += unit_words

    if current_units:
        final_chunk = "\n\n".join(current_units).strip()
        # Avoid duplicating an identical overlap-only chunk
        if chunks and final_chunk == chunks[-1]:
            pass
        elif final_chunk:
            chunks.append(final_chunk)

    return chunks


def chunk_document_pages(
    pages: list[ExtractedPage],
    metadata: DocumentMetadata,
    target_min_words: int = 150,
    target_max_words: int = 250,
    overlap_words: int = 30,
) -> list[DocumentChunk]:
    """Chunk extracted pages and attach all 13 agreed metadata fields to each chunk."""
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        passages = chunk_text(
            page_text,
            target_min_words=target_min_words,
            target_max_words=target_max_words,
            overlap_words=overlap_words,
        )

        for passage in passages:
            passage_clean = passage.strip()
            if not passage_clean:
                continue

            chunk_id = generate_chunk_id(
                doc_id=metadata.doc_id,
                version=metadata.version,
                chunk_index=chunk_index,
                content=passage_clean,
            )

            chunk = DocumentChunk(
                doc_id=metadata.doc_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                title=metadata.title,
                page=page.page_number,  # 1-based for PDF, 0 for DOCX/TXT
                text=passage_clean,     # Clean original passage text
                access_group=metadata.access_group,
                modified_at=metadata.modified_at,
                source_path=metadata.source_path,
                source_type=metadata.source_type,
                version=metadata.version,
                status=metadata.status,
                supersedes=metadata.supersedes,
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks


# ============================================================================
# High-Level Ingestion Entry Points
# ============================================================================

def resolve_file_path(
    source_path: str | Path,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve document file path against base directory, env, or working directory."""
    path = Path(source_path)
    if path.is_absolute() and path.exists():
        return path

    candidates: list[Path] = []
    if base_dir:
        candidates.append(Path(base_dir) / path)

    env_dir = os.getenv("DOCUMENTS_DIRECTORY")
    if env_dir:
        candidates.append(Path(env_dir) / path)

    candidates.append(Path.cwd() / path)
    candidates.append(path)

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    return path


def ingest_document(
    metadata: DocumentMetadata | dict[str, Any],
    base_dir: str | Path | None = None,
    raise_on_empty: bool = False,
    target_min_words: int = 150,
    target_max_words: int = 250,
    overlap_words: int = 30,
) -> list[DocumentChunk]:
    """Ingest, extract, and chunk a single document from validated metadata."""
    meta = validate_manifest_record(metadata)
    file_path = resolve_file_path(meta.source_path, base_dir=base_dir)

    pages = extract_document(file_path, meta, raise_on_empty=raise_on_empty)
    return chunk_document_pages(
        pages,
        meta,
        target_min_words=target_min_words,
        target_max_words=target_max_words,
        overlap_words=overlap_words,
    )


def ingest_manifest(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
    raise_on_empty: bool = False,
) -> list[DocumentChunk]:
    """Ingest all documents listed in a validated manifest.csv file."""
    manifest_p = Path(manifest_path)
    manifest_records = validate_manifest_csv(manifest_p)

    resolved_base = base_dir if base_dir else manifest_p.parent

    all_chunks: list[DocumentChunk] = []
    for record in manifest_records:
        chunks = ingest_document(
            record,
            base_dir=resolved_base,
            raise_on_empty=raise_on_empty,
        )
        all_chunks.extend(chunks)

    return all_chunks


class DocumentIngester(IngestionContract):
    """Encapsulates document loading and chunking configuration and operations."""

    def __init__(
        self,
        target_min_words: int = 150,
        target_max_words: int = 250,
        overlap_words: int = 30,
        base_dir: str | Path | None = None,
    ) -> None:
        self.target_min_words = target_min_words
        self.target_max_words = target_max_words
        self.overlap_words = overlap_words
        self.base_dir = base_dir

    def load_document(self, file_path: str) -> Document:
        path = resolve_file_path(file_path, base_dir=self.base_dir)
        source_type = path.suffix.lower().lstrip(".")
        meta = DocumentMetadata(
            doc_id=path.stem,
            title=path.name,
            source_path=str(path),
            source_type=source_type if source_type in ("pdf", "docx", "txt", "md") else "txt",
            version="1.0",
            status="active",
            access_group="public",
            modified_at="unknown",
        )
        pages = extract_document(path, meta, raise_on_empty=False)
        combined = "\n\n".join(p.text for p in pages if p.text)
        return Document(
            id=meta.doc_id,
            content=combined,
            metadata={"source_path": str(path), "pages_count": len(pages)},
            permissions=[meta.access_group],
        )

    def chunk_document(
        self,
        document: Document,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[DocumentChunk]:
        meta = DocumentMetadata(
            doc_id=document.id,
            title=document.metadata.get("title", document.id),
            source_path=document.metadata.get("source_path", document.id),
            source_type=document.metadata.get("source_type", "txt"),
            version=document.metadata.get("version", "1.0"),
            status=document.metadata.get("status", "active"),
            access_group=document.permissions[0] if document.permissions else "public",
            modified_at=document.metadata.get("modified_at", "unknown"),
        )
        page = ExtractedPage(
            doc_id=document.id,
            page_number=UNKNOWN_PAGE_NUMBER,
            text=document.content,
        )
        return chunk_document_pages(
            [page],
            meta,
            target_min_words=self.target_min_words,
            target_max_words=self.target_max_words,
            overlap_words=self.overlap_words,
        )

"""src/retrieval/retrieve.py

Permission-aware retrieval layer for the Document Q&A System.
Combines dense semantic vector retrieval (ChromaDB) and sparse keyword retrieval (BM25)
using Reciprocal Rank Fusion (RRF), while guaranteeing zero data leakage across
access boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

from rank_bm25 import BM25Okapi

from contracts import (
    DocumentChunk,
    RetrievalContract,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    UserContext,
)
from src.retrieval.index import (
    DEFAULT_COLLECTION_NAME,
    get_default_embedding_function,
    get_or_create_collection,
    prepare_query_text_for_embedding,
)

logger = logging.getLogger(__name__)


# ============================================================================
# BM25 Tokenization
# ============================================================================

def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 keyword search.

    Preserves exact compound identifiers like 'SOP-QA-014', 'DOC-LAB-01', 'ISO-9001',
    as well as emitting lowercase alphanumeric words for natural-language queries.

    Args:
        text: Query or document text string.

    Returns:
        List of extracted string tokens.
    """
    if not text:
        return []
    text_lower = text.lower()

    # 1. Hyphenated / dotted / underscored compound tokens (e.g. 'sop-qa-014', 'iso-9001')
    compounds = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)+", text_lower)

    # 2. Individual alphanumeric words (e.g. 'sop', 'qa', '014', 'centrifuge')
    words = re.findall(r"[a-z0-9]+", text_lower)

    return compounds + words


# ============================================================================
# Core Permission-Aware Vector Retrieval
# ============================================================================

def vector_retrieve(
    query: str,
    groups: list[str],
    top_k: int = 10,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
) -> list[RetrievedChunk]:
    """Execute a permission-filtered vector similarity search in ChromaDB.

    Enforces:
    1. Query embedding with sentence-transformers/all-MiniLM-L6-v2 (or custom embedding_fn).
    2. Strict access_group filtering: where={"access_group": {"$in": groups}}.
    3. If groups is empty, immediately returns [] (never falls back to unfiltered search).
    4. Computes similarity score = 1 - cosine_distance.

    Args:
        query: User question or search query string.
        groups: Authorized access groups for the requesting user.
        top_k: Maximum number of relevant chunks to return.
        collection_name: Target Chroma collection name.
        persist_directory: Storage directory path.
        embedding_fn: Embedding callable (defaults to all-MiniLM-L6-v2).

    Returns:
        List of ranked RetrievedChunk objects.
    """
    clean_query = prepare_query_text_for_embedding(query)
    if not clean_query:
        return []

    # Security check: if groups is empty, reject query to prevent leakage
    valid_groups = [str(g).strip() for g in groups if str(g).strip()]
    if not valid_groups:
        logger.warning(
            "Access denied: empty access groups provided for query '%s'. Zero chunks returned.",
            clean_query,
        )
        return []

    emb_function = embedding_fn if embedding_fn is not None else get_default_embedding_function()
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_fn=emb_function,
    )

    # Compute query embedding using the same model and preprocessing
    query_vector = emb_function([clean_query])[0]

    # Enforce access_group filtering inside ChromaDB
    where_filter: dict[str, Any] = {"access_group": {"$in": valid_groups}}

    res = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    results: list[RetrievedChunk] = []
    if res.get("ids") and res["ids"][0]:
        ids = res["ids"][0]
        docs = res["documents"][0] if res.get("documents") else [""] * len(ids)
        metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(ids)
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)

        for idx in range(len(ids)):
            chunk_id = ids[idx]
            passage_text = docs[idx]
            raw_meta = metas[idx] or {}
            distance = float(dists[idx])

            # Similarity score = 1 - cosine_distance
            similarity = max(0.0, min(1.0, 1.0 - distance))

            # Reconstruct typed DocumentChunk preserving all 13 metadata fields
            chunk = DocumentChunk(
                doc_id=raw_meta.get("doc_id", ""),
                chunk_id=chunk_id,
                chunk_index=int(raw_meta.get("chunk_index", 0)),
                title=raw_meta.get("title", ""),
                page=int(raw_meta.get("page", 0)),
                text=passage_text,  # Clean original passage text
                access_group=raw_meta.get("access_group", ""),
                modified_at=raw_meta.get("modified_at", ""),
                source_path=raw_meta.get("source_path", ""),
                source_type=raw_meta.get("source_type", ""),
                version=raw_meta.get("version", "1.0"),
                status=raw_meta.get("status", "active"),
                supersedes=raw_meta.get("supersedes") or None,
            )

            retrieved = RetrievedChunk(
                chunk=chunk,
                score=round(similarity, 6),
                similarity=round(similarity, 6),
                distance=round(distance, 6),
                rrf_score=0.0,
                bm25_score=0.0,
            )
            results.append(retrieved)

    # Sort by similarity score descending
    results.sort(key=lambda r: r.similarity, reverse=True)
    return results


# ============================================================================
# Core Permission-Aware BM25 Retrieval
# ============================================================================

def bm25_retrieve(
    query: str,
    groups: list[str],
    top_k: int = 10,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
) -> list[RetrievedChunk]:
    """Execute a permission-filtered BM25 keyword search over authorized document chunks.

    Security & Freshness Guarantees:
    1. The BM25 corpus is constructed strictly from permitted chunks in ChromaDB:
       `where={"access_group": {"$in": groups}}`. Unauthorized chunks never enter
       the candidate set or BM25 index.
    2. If groups is empty, immediately returns [] (never falls back to unfiltered search).
    3. Document deletions and index updates in ChromaDB are reflected immediately without
       any stale BM25 entries.
    4. Exact identifiers (such as SOP-QA-014) and natural language terms are both indexed.

    Args:
        query: Search query containing keywords or exact identifiers.
        groups: Authorized access groups for the requesting user.
        top_k: Maximum number of relevant chunks to return.
        collection_name: ChromaDB collection name.
        persist_directory: Storage directory path.
        embedding_fn: Embedding callable (defaults to all-MiniLM-L6-v2).

    Returns:
        List of ranked RetrievedChunk objects with BM25 score.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    valid_groups = [str(g).strip() for g in groups if str(g).strip()]
    if not valid_groups:
        logger.warning(
            "Access denied: empty access groups provided for BM25 query '%s'. Zero chunks returned.",
            clean_query,
        )
        return []

    query_tokens = tokenize_for_bm25(clean_query)
    if not query_tokens:
        return []

    emb_function = embedding_fn if embedding_fn is not None else get_default_embedding_function()
    collection = get_or_create_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_fn=emb_function,
    )

    # Fetch ONLY permitted chunks directly from live Chroma storage
    where_filter: dict[str, Any] = {"access_group": {"$in": valid_groups}}
    res = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )

    if not res or not res.get("ids") or len(res["ids"]) == 0:
        return []

    ids = res["ids"]
    docs = res.get("documents") or [""] * len(ids)
    metas = res.get("metadatas") or [{}] * len(ids)

    chunks: list[DocumentChunk] = []
    corpus_texts: list[str] = []

    for cid, doc_text, raw_meta in zip(ids, docs, metas):
        raw_meta = raw_meta or {}
        chunk = DocumentChunk(
            doc_id=raw_meta.get("doc_id", ""),
            chunk_id=cid,
            chunk_index=int(raw_meta.get("chunk_index", 0)),
            title=raw_meta.get("title", ""),
            page=int(raw_meta.get("page", 0)),
            text=doc_text,
            access_group=raw_meta.get("access_group", ""),
            modified_at=raw_meta.get("modified_at", ""),
            source_path=raw_meta.get("source_path", ""),
            source_type=raw_meta.get("source_type", ""),
            version=raw_meta.get("version", "1.0"),
            status=raw_meta.get("status", "active"),
            supersedes=raw_meta.get("supersedes") or None,
        )
        chunks.append(chunk)

        # Index doc_id, title, and passage text to ensure exact identifiers match
        bm25_text = f"{chunk.doc_id} {chunk.title} {chunk.text}"
        corpus_texts.append(bm25_text)

    # Build BM25 index on permitted corpus
    tokenized_corpus = [tokenize_for_bm25(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    results: list[RetrievedChunk] = []
    for chunk, raw_score in zip(chunks, scores):
        score_val = float(raw_score)
        # Only retain candidates with positive keyword match
        if score_val <= 0.0:
            continue

        retrieved = RetrievedChunk(
            chunk=chunk,
            score=round(score_val, 6),
            bm25_score=round(score_val, 6),
            similarity=0.0,
            distance=1.0,
            rrf_score=0.0,
        )
        results.append(retrieved)

    # Deterministic tie-breaking: primary (-bm25_score), secondary (chunk_id ascending)
    results.sort(key=lambda r: (-r.bm25_score, r.chunk.chunk_id))
    return results[:top_k]


# ============================================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================================

def reciprocal_rank_fusion(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Fuse vector and BM25 ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(c) = sum(1.0 / (k + rank_m(c))) for m in [vector, bm25]
        where rank_m(c) is 1-based rank in candidate list m.

    Features & Invariants:
    - Deduplicates candidates strictly by chunk_id.
    - Preserves distinct RRF ranking score (`rrf_score`) and vector similarity (`similarity`).
    - Deterministic tie-breaking: sorts by (-rrf_score, -similarity, chunk_id ascending).
    - Returns up to top_k fused candidates.

    Args:
        vector_results: Candidate chunks ranked by vector similarity.
        bm25_results: Candidate chunks ranked by BM25 keyword score.
        k: Smoothing constant (standard default is 60).
        top_k: Maximum number of fused results to return.

    Returns:
        List of fused RetrievedChunk instances sorted deterministically.
    """
    fused_map: dict[str, dict[str, Any]] = {}

    # 1. Process vector candidates
    for rank_idx, item in enumerate(vector_results, start=1):
        cid = item.chunk.chunk_id
        rrf_contrib = 1.0 / (k + rank_idx)
        if cid not in fused_map:
            fused_map[cid] = {
                "chunk": item.chunk,
                "similarity": item.similarity,
                "distance": item.distance,
                "bm25_score": 0.0,
                "rrf_score": 0.0,
            }
        fused_map[cid]["rrf_score"] += rrf_contrib

    # 2. Process BM25 candidates
    for rank_idx, item in enumerate(bm25_results, start=1):
        cid = item.chunk.chunk_id
        rrf_contrib = 1.0 / (k + rank_idx)
        if cid not in fused_map:
            fused_map[cid] = {
                "chunk": item.chunk,
                "similarity": item.similarity or 0.0,
                "distance": item.distance if item.distance is not None else 1.0,
                "bm25_score": item.bm25_score or item.score,
                "rrf_score": 0.0,
            }
        else:
            if not fused_map[cid]["similarity"] and item.similarity:
                fused_map[cid]["similarity"] = item.similarity
            fused_map[cid]["bm25_score"] = item.bm25_score or item.score
        fused_map[cid]["rrf_score"] += rrf_contrib

    # 3. Deterministic tie-breaking sort:
    # Primary: -rrf_score (descending)
    # Secondary: -similarity (descending)
    # Tertiary: chunk_id (alphabetical ascending)
    sorted_items = sorted(
        fused_map.values(),
        key=lambda x: (-x["rrf_score"], -x["similarity"], x["chunk"].chunk_id),
    )

    # 4. Construct final typed RetrievedChunk list
    fused_results: list[RetrievedChunk] = []
    for entry in sorted_items[:top_k]:
        rrf_val = round(entry["rrf_score"], 6)
        sim_val = round(entry["similarity"], 6)
        dist_val = round(entry["distance"], 6)
        bm25_val = round(entry["bm25_score"], 6)

        rc = RetrievedChunk(
            chunk=entry["chunk"],
            score=rrf_val,
            rrf_score=rrf_val,
            similarity=sim_val,
            distance=dist_val,
            bm25_score=bm25_val,
        )
        fused_results.append(rc)

    return fused_results


# ============================================================================
# Hybrid Retrieval
# ============================================================================

def hybrid_retrieve(
    query: str,
    groups: list[str],
    top_k: int = 10,
    candidate_k: int | None = None,
    rrf_k: int = 60,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
) -> list[RetrievedChunk]:
    """Execute permission-aware hybrid retrieval combining Vector and BM25 search via RRF.

    Args:
        query: User search query or exact identifier.
        groups: Authorized access groups for the requesting user.
        top_k: Number of final fused chunks to return.
        candidate_k: Candidate limit per retriever before fusion (default: max(top_k * 3, 30)).
        rrf_k: Reciprocal Rank Fusion smoothing parameter (default: 60).
        collection_name: Target Chroma collection name.
        persist_directory: Storage directory path.
        embedding_fn: Embedding callable.

    Returns:
        List of fused, deduplicated RetrievedChunk instances.
    """
    valid_groups = [str(g).strip() for g in groups if str(g).strip()]
    if not valid_groups or not query.strip():
        return []

    num_candidates = candidate_k if candidate_k is not None else max(top_k * 3, 30)

    # 1. Fetch vector candidates
    vector_candidates = vector_retrieve(
        query=query,
        groups=valid_groups,
        top_k=num_candidates,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_fn=embedding_fn,
    )

    # 2. Fetch BM25 candidates from strictly authorized corpus
    bm25_candidates = bm25_retrieve(
        query=query,
        groups=valid_groups,
        top_k=num_candidates,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_fn=embedding_fn,
    )

    # 3. Fuse with RRF
    return reciprocal_rank_fusion(
        vector_results=vector_candidates,
        bm25_results=bm25_candidates,
        k=rrf_k,
        top_k=top_k,
    )


# ============================================================================
# High-Level Retrieval Entry Point
# ============================================================================

def retrieve(
    query: str | RetrievalRequest,
    groups: list[str] | str | UserContext | None = None,
    top_k: int = 10,
    mode: str = "rrf",
    candidate_k: int | None = None,
    rrf_k: int = 60,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | Path | None = None,
    embedding_fn: Any = None,
    user_context: UserContext | None = None,
    **kwargs: Any,
) -> RetrievalResponse:
    """Execute permission-aware document retrieval.

    Supports:
    - Hybrid RRF retrieval (`mode='rrf'` or `mode='hybrid'`)
    - Vector-only retrieval (`mode='vector'`)
    - Keyword-only retrieval (`mode='bm25'`)
    - Both functional arguments `retrieve(query, groups, top_k)`
      and structured request objects `retrieve(request: RetrievalRequest)`.

    Args:
        query: Query string or structured RetrievalRequest.
        groups: List of authorized access groups, single group string, or UserContext.
        top_k: Number of chunks to return.
        mode: Retrieval strategy ('rrf', 'hybrid', 'vector', or 'bm25').
        candidate_k: Configurable candidate limit for RRF before fusion.
        rrf_k: Reciprocal Rank Fusion smoothing parameter (default: 60).
        collection_name: Target Chroma collection name.
        persist_directory: Storage directory path.
        embedding_fn: Custom embedding callable.
        user_context: Requesting UserContext.

    Returns:
        Structured RetrievalResponse containing ranked results.
    """
    # 1. Resolve query, user_context, top_k, and groups
    if isinstance(query, RetrievalRequest):
        req = query
        target_query = req.query
        target_top_k = req.top_k or top_k
        user_ctx = req.user_context
        target_groups = list(set(user_ctx.groups + user_ctx.roles))
    else:
        target_query = str(query)
        target_top_k = top_k

        if isinstance(groups, UserContext):
            user_ctx = groups
            target_groups = list(set(user_ctx.groups + user_ctx.roles))
        elif user_context is not None:
            user_ctx = user_context
            target_groups = list(set(user_ctx.groups + user_ctx.roles))
            if isinstance(groups, list):
                target_groups.extend(groups)
        elif isinstance(groups, str):
            target_groups = [groups.strip()] if groups.strip() else []
            user_ctx = UserContext(user_id="user", roles=target_groups, groups=target_groups)
        elif isinstance(groups, (list, tuple, set)):
            target_groups = [str(g).strip() for g in groups if str(g).strip()]
            user_ctx = UserContext(user_id="user", roles=target_groups, groups=target_groups)
        else:
            target_groups = []
            user_ctx = UserContext(user_id="anonymous", roles=[], groups=[])

    # Deduplicate groups
    unique_groups = sorted(set(target_groups))

    # 2. Execute retrieval strategy according to mode
    norm_mode = mode.lower() if mode else "rrf"
    if norm_mode in ("rrf", "hybrid"):
        chunks = hybrid_retrieve(
            query=target_query,
            groups=unique_groups,
            top_k=target_top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_fn=embedding_fn,
        )
    elif norm_mode == "vector":
        chunks = vector_retrieve(
            query=target_query,
            groups=unique_groups,
            top_k=target_top_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_fn=embedding_fn,
        )
    elif norm_mode == "bm25":
        chunks = bm25_retrieve(
            query=target_query,
            groups=unique_groups,
            top_k=target_top_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_fn=embedding_fn,
        )
    else:
        chunks = hybrid_retrieve(
            query=target_query,
            groups=unique_groups,
            top_k=target_top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_fn=embedding_fn,
        )

    return RetrievalResponse(
        query=target_query,
        results=chunks,
        user_context=user_ctx,
        total_found=len(chunks),
    )


# ============================================================================
# PermissionAwareRetriever Class Implementation
# ============================================================================

class PermissionAwareRetriever(RetrievalContract):
    """Enforces access control and permission filtering during retrieval."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | Path | None = None,
        default_top_k: int = 10,
        embedding_fn: Any = None,
        mode: str = "rrf",
        candidate_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.default_top_k = default_top_k
        self.embedding_fn = embedding_fn
        self.mode = mode
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return retrieve(
            query=request,
            top_k=request.top_k or self.default_top_k,
            mode=self.mode,
            candidate_k=self.candidate_k,
            rrf_k=self.rrf_k,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_fn=self.embedding_fn,
        )

    def retrieve_for_user(
        self,
        query: str,
        user: UserContext,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        return retrieve(
            query=query,
            user_context=user,
            top_k=top_k or self.default_top_k,
            mode=self.mode,
            candidate_k=self.candidate_k,
            rrf_k=self.rrf_k,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_fn=self.embedding_fn,
        )

    def retrieve_for_groups(
        self,
        query: str,
        groups: list[str],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        resp = retrieve(
            query=query,
            groups=groups,
            top_k=top_k or self.default_top_k,
            mode=self.mode,
            candidate_k=self.candidate_k,
            rrf_k=self.rrf_k,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_fn=self.embedding_fn,
        )
        return resp.results

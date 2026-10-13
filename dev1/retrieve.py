"""dev1/retrieve.py

Dev 1 permission-aware retrieval module connected to src.retrieval.retrieve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contracts import (
    RetrievalContract,
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    UserContext,
)
from src.retrieval.retrieve import (
    DEFAULT_COLLECTION_NAME,
    PermissionAwareRetriever,
    bm25_retrieve,
    hybrid_retrieve,
    reciprocal_rank_fusion,
    retrieve,
    tokenize_for_bm25,
    vector_retrieve,
)

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "retrieve",
    "vector_retrieve",
    "bm25_retrieve",
    "reciprocal_rank_fusion",
    "hybrid_retrieve",
    "tokenize_for_bm25",
    "PermissionAwareRetriever",
]

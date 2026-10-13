"""Document chunking and mock vector retrieval interface (T06).

The package exposes a provider-neutral contract: ``Chunker`` produces typed
``Chunk`` records with exact source offsets; ``EmbeddingProvider`` turns text
into ``EmbeddingRecord`` objects; ``VectorIndex`` stores them and answers
nearest-neighbour queries with scores and source offsets.

Only deterministic mock providers are bundled.  Real embedding models are an
approval-gated follow-up task (T13).
"""
from __future__ import annotations

from hermes_bot.retrieval.chunker import Chunker, chunk_id_from
from hermes_bot.retrieval.errors import (
    DimensionMismatch,
    IncompatibleIndex,
    RetrievalError,
    ZeroVectorRejected,
)
from hermes_bot.retrieval.index import SearchResult, VectorIndex
from hermes_bot.retrieval.provider import (
    TEST_MODEL_ID,
    EmbeddingProvider,
    TestEmbeddingProvider,
    available_providers,
)

__all__ = [
    "TEST_MODEL_ID",
    "Chunker",
    "chunk_id_from",
    "EmbeddingProvider",
    "TestEmbeddingProvider",
    "available_providers",
    "VectorIndex",
    "SearchResult",
    "RetrievalError",
    "DimensionMismatch",
    "IncompatibleIndex",
    "ZeroVectorRejected",
]
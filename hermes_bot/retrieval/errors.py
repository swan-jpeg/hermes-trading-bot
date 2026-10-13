"""Typed errors for the retrieval layer."""
from __future__ import annotations


class RetrievalError(Exception):
    """Base class for retrieval-layer errors."""


class DimensionMismatch(RetrievalError):
    """Vector dimension does not match the provider or index dimension."""


class IncompatibleIndex(RetrievalError):
    """Vectors are incompatible with the index partition (task/model/revision/dim)."""


class ZeroVectorRejected(RetrievalError):
    """A zero or non-finite vector was rejected by the index."""

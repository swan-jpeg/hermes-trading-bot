"""Data contracts for the weighted multi-method correlation graph (T18).

All models are immutable (frozen pydantic), consistent with the rest of the
repository.  Everything is offline and deterministic: no network, no trained
models, no real market data.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CorrelationNode(BaseModel):
    """A node in the correlation graph with a systematic embedding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
    node_type: str = Field(min_length=1)
    as_of: datetime


class CorrelationMethodResult(BaseModel):
    """Output of one correlation method on a pair of return series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)
    sample_size: int = Field(ge=0)


class CorrelationEdge(BaseModel):
    """A weighted multi-method correlation between two nodes.

    ``method_weights`` are derived deterministically from the two node
    embeddings and always sum to 1.0.  ``correlation_score`` is the weighted
    combination of the per-method scores.  ``relation_embedding`` embeds the
    *way* the two nodes are connected (e.g. "supplier").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str = Field(min_length=1)
    node_a: str = Field(min_length=1)
    node_b: str = Field(min_length=1)
    as_of: datetime
    window: int = Field(ge=1)
    method_scores: dict[str, float]
    method_weights: dict[str, float]
    correlation_score: float = Field(ge=-1.0, le=1.0)
    certainty: float = Field(ge=0.0, le=1.0)
    relation_type: str | None = None
    relation_embedding: list[float] = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)

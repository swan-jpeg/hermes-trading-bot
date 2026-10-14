"""Weighted multi-method correlation graph infrastructure (T18).

Offline and deterministic: no network, no trained models, no real data.
Method weights are derived from node embeddings (softmax over deterministic
projection vectors), each node holds at most 50 edges, and correlations are
never double-counted: ``effective_correlation`` is direct-edge-only, while
``partial_correlation`` removes the linear effect of shared neighbours.

Persistence uses migration v3 in ``hermes_bot.data.migrations``.
"""
from __future__ import annotations

from hermes_bot.correlation.graph import (
    MAX_EDGES_PER_NODE,
    CorrelationGraph,
    edge_strength,
)
from hermes_bot.correlation.methods import (
    CorrelationMethod,
    DynamicConditionalCorrelation,
    KendallTau,
    RollingPearson,
    Spearman,
    default_methods,
)
from hermes_bot.correlation.repository import CorrelationRepository
from hermes_bot.correlation.schemas import (
    CorrelationEdge,
    CorrelationMethodResult,
    CorrelationNode,
)
from hermes_bot.correlation.weighting import (
    KNOWN_RELATION_TYPES,
    build_edge,
    compute_certainty,
    method_weights,
    projection_vector,
    relation_embedding,
)

__all__ = [
    "MAX_EDGES_PER_NODE",
    "KNOWN_RELATION_TYPES",
    "CorrelationEdge",
    "CorrelationGraph",
    "CorrelationMethod",
    "CorrelationMethodResult",
    "CorrelationNode",
    "CorrelationRepository",
    "DynamicConditionalCorrelation",
    "KendallTau",
    "RollingPearson",
    "Spearman",
    "build_edge",
    "compute_certainty",
    "default_methods",
    "edge_strength",
    "method_weights",
    "projection_vector",
    "relation_embedding",
]

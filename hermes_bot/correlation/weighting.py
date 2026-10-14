"""Deterministic method weighting from node embeddings (T18).

No trained parameters: the projection vector ``w_m`` per method is derived
from a SHA-256 hash of the method name (same hash-to-float scheme as the
T06 ``TestEmbeddingProvider``).

Weight formula (exact)
----------------------
For an edge between nodes A and B with embeddings ``eA`` and ``eB``
(equal dimension d):

    pair[i] = eA[i] * eB[i]                      (elementwise product)
    logit_m = dot(pair, w_m)                     (one logit per method)
    weight_m = exp(logit_m - max_k logit_k) / sum_j exp(logit_j - max_k logit_k)

so the weights form a softmax over the methods and always sum to 1.0.
Different node embeddings produce different pairwise products and therefore
different weights.

Weighted score:  ``correlation_score = sum_m weight_m * score_m`` (clipped
to [-1, 1]).

Certainty formula (exact)
-------------------------
Let ``scores`` be the per-method scores and ``n`` the smallest method
``sample_size`` (conservative) with ``min_samples`` (default 20):

    agreement  = clip(1 - std_pop(scores) / 2, 0, 1)   (method consensus;
                 max population std of values in [-1,1] is 1, halved for a
                 gentler slope)
    size_factor = min(1, n / min_samples)
    certainty  = agreement * size_factor               (in [0, 1])

Relation embedding
------------------
``relation_embedding = embedding_provider.embed([label]).vector`` where the
label is ``"relation:<relation_type>"``.  A relation type outside
``KNOWN_RELATION_TYPES`` (or ``None``) maps to the deterministic default
label ``"relation:unknown"`` and sets ``relation_embedding_default=True``
in the edge evidence.
"""
from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from hermes_bot.correlation.methods import CorrelationMethod
from hermes_bot.correlation.schemas import CorrelationEdge, CorrelationMethodResult, CorrelationNode
from hermes_bot.retrieval.provider import EmbeddingProvider

DEFAULT_MIN_SAMPLES = 20
UNKNOWN_RELATION_LABEL = "relation:unknown"
KNOWN_RELATION_TYPES = frozenset(
    {"supplier", "customer", "competitor", "sector", "ownership", "correlation"}
)


def _sha256_floats(seed: bytes, dimension: int) -> list[float]:
    """Deterministic finite floats in [-1, 1] from a seed (T06 scheme)."""
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for i in range(0, len(digest), 4):
            if len(values) >= dimension:
                break
            u32 = struct.unpack(">I", digest[i : i + 4])[0]
            values.append(u32 / 0xFFFFFFFF * 2 - 1)
        counter += 1
    return values


def projection_vector(method_name: str, dimension: int) -> list[float]:
    """Fixed deterministic projection vector ``w_m`` for a method."""
    seed = hashlib.sha256(f"correlation-method:{method_name}".encode()).digest()
    return _sha256_floats(seed, dimension)


def method_weights(
    embedding_a: Sequence[float],
    embedding_b: Sequence[float],
    method_names: Sequence[str],
) -> dict[str, float]:
    """Softmax method weights from two node embeddings. Sums to 1.0."""
    if len(embedding_a) != len(embedding_b):
        raise ValueError("node embeddings must have equal dimension")
    if not method_names:
        raise ValueError("at least one method is required")
    pair = [a * b for a, b in zip(embedding_a, embedding_b, strict=True)]
    logits: dict[str, float] = {}
    for name in method_names:
        w_m = projection_vector(name, len(pair))
        logits[name] = sum(p * w for p, w in zip(pair, w_m, strict=True))
    max_logit = max(logits.values())
    exps = {name: math.exp(logit - max_logit) for name, logit in logits.items()}
    total = sum(exps.values())
    return {name: value / total for name, value in exps.items()}


def compute_certainty(
    scores: Sequence[float],
    sample_size: int,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> float:
    """Certainty in [0, 1]: method agreement times sample-size factor."""
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)
    agreement = max(0.0, min(1.0, 1.0 - std / 2.0))
    size_factor = min(1.0, sample_size / min_samples) if min_samples > 0 else 1.0
    return max(0.0, min(1.0, agreement * size_factor))


def relation_embedding(
    provider: EmbeddingProvider,
    relation_type: str | None,
) -> tuple[list[float], bool]:
    """Embed the connection label; returns (vector, used_default)."""
    if relation_type is not None and relation_type in KNOWN_RELATION_TYPES:
        label = f"relation:{relation_type}"
        used_default = False
    else:
        label = UNKNOWN_RELATION_LABEL
        used_default = True
    record = provider.embed([label], {"object_id": label})[0]
    return list(record.vector), used_default


def build_edge(
    *,
    edge_id: str,
    node_a: CorrelationNode,
    node_b: CorrelationNode,
    series_a: Sequence[float],
    series_b: Sequence[float],
    methods: dict[str, CorrelationMethod],
    provider: EmbeddingProvider,
    as_of: datetime,
    window: int = 20,
    relation_type: str | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> CorrelationEdge:
    """Build a fully weighted correlation edge from two nodes and series."""
    results: list[CorrelationMethodResult] = [
        methods[name].score(series_a, series_b) for name in sorted(methods)
    ]
    scores = {r.method: r.score for r in results}
    weights = method_weights(node_a.embedding, node_b.embedding, sorted(methods))
    correlation_score = sum(weights[m] * scores[m] for m in scores)
    correlation_score = max(-1.0, min(1.0, correlation_score))
    sample_size = min(r.sample_size for r in results)
    certainty = compute_certainty(
        list(scores.values()), sample_size, min_samples=min_samples
    )
    embedding, used_default = relation_embedding(provider, relation_type)
    evidence: dict[str, Any] = {
        "sample_size": sample_size,
        "min_samples": min_samples,
        "method_sample_sizes": {r.method: r.sample_size for r in results},
        "relation_embedding_default": used_default,
    }
    return CorrelationEdge(
        edge_id=edge_id,
        node_a=node_a.node_id,
        node_b=node_b.node_id,
        as_of=as_of,
        window=window,
        method_scores=scores,
        method_weights=weights,
        correlation_score=correlation_score,
        certainty=certainty,
        relation_type=relation_type,
        relation_embedding=embedding,
        evidence=evidence,
    )

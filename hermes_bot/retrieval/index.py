"""In-memory exact cosine vector index for tiny corpora (T06).

This index is intentionally simple: it keeps records in memory, normalizes
vectors, and computes exact cosine similarity.  It is partitioned by
``(task, model_id, model_revision, dimension)`` so incompatible vectors cannot
share an index.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hermes_bot.retrieval.errors import (
    DimensionMismatch,
    IncompatibleIndex,
    ZeroVectorRejected,
)
from hermes_bot.schemas import EmbeddingRecord, SimilarityMetric


@dataclass(frozen=True)
class SearchResult:
    """One retrieval hit with score and source metadata."""

    embedding_id: str
    object_id: str
    object_type: str
    score: float
    source_offsets: tuple[int, int] | None
    available_at: datetime
    metadata: dict[str, Any]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        raise ZeroVectorRejected("zero vector cannot be indexed")
    return [v / norm for v in vector]


def _is_finite(vector: list[float]) -> bool:
    return all(math.isfinite(v) for v in vector)


def _partition_key(record: EmbeddingRecord) -> tuple[str, str, str, int]:
    return (record.task.value, record.model_id, record.model_revision, record.dimension)


class VectorIndex:
    """In-memory exact cosine index partitioned by model metadata.

    Parameters
    ----------
    task:
        Retrieval task partition.
    model_id:
        Model identifier partition.
    model_revision:
        Immutable model revision partition.
    dimension:
        Expected vector dimension.
    """

    def __init__(self, task: str, model_id: str, model_revision: str, dimension: int) -> None:
        self.task = task
        self.model_id = model_id
        self.model_revision = model_revision
        self.dimension = dimension
        self._records: dict[str, EmbeddingRecord] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, records: list[EmbeddingRecord]) -> None:
        """Add or replace records in the index.

        Duplicate ``embedding_id`` values are overwritten deterministically.
        """
        expected_key = (self.task, self.model_id, self.model_revision, self.dimension)
        for record in records:
            if record.dimension != self.dimension:
                raise DimensionMismatch(
                    f"dimension {record.dimension} != index dimension {self.dimension}"
                )
            key = _partition_key(record)
            if key != expected_key:
                raise IncompatibleIndex(
                    f"record partition {key} does not match index {expected_key}"
                )
            if record.similarity_metric != SimilarityMetric.COSINE:
                raise IncompatibleIndex(
                    f"metric {record.similarity_metric.value} not supported by this index"
                )
            if not _is_finite(record.vector):
                raise ZeroVectorRejected("vector contains non-finite values")

            normalized = _normalize(record.vector)
            self._records[record.embedding_id] = record
            self._vectors[record.embedding_id] = normalized

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        as_of: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return top-k results by exact cosine similarity.

        Parameters
        ----------
        query_vector:
            Query vector; must match the index dimension.
        top_k:
            Maximum number of results.
        as_of:
            Only return records with ``available_at <= as_of``.
        metadata_filter:
            Only return records whose ``metadata`` contains all key/value pairs.
        """
        if len(query_vector) != self.dimension:
            raise DimensionMismatch(
                f"query dimension {len(query_vector)} != index dimension {self.dimension}"
            )
        if not _is_finite(query_vector):
            raise ZeroVectorRejected("query vector contains non-finite values")

        query_norm = _normalize(query_vector)
        scores: list[tuple[float, str]] = []
        for embedding_id, vector in self._vectors.items():
            record = self._records[embedding_id]
            if as_of is not None and record.available_at > as_of:
                continue
            if metadata_filter and not all(
                record.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            score = sum(a * b for a, b in zip(query_norm, vector, strict=False))
            scores.append((score, embedding_id))

        scores.sort(key=lambda x: (-x[0], x[1]))
        results: list[SearchResult] = []
        for score, embedding_id in scores[:top_k]:
            record = self._records[embedding_id]
            score = max(-1.0, min(1.0, score))
            source_offsets: tuple[int, int] | None = None
            start_off = record.metadata.get("start_offset")
            end_off = record.metadata.get("end_offset")
            if isinstance(start_off, int) and isinstance(end_off, int):
                source_offsets = (start_off, end_off)
            results.append(
                SearchResult(
                    embedding_id=embedding_id,
                    object_id=record.object_id,
                    object_type=record.object_type.value,
                    score=score,
                    source_offsets=source_offsets,
                    available_at=record.available_at,
                    metadata=dict(record.metadata),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._records)

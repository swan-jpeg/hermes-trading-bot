"""Embedding provider protocol and deterministic mock provider (T06).

Real embedding models are explicitly excluded from this task.  The bundled
``TestEmbeddingProvider`` produces deterministic 384-dimensional vectors from a
lexical hash and is marked ``test_only`` so it cannot be selected by
application configuration.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from hermes_bot.schemas import EmbeddingRecord, EmbeddingTask, ObjectType, SimilarityMetric

TEST_MODEL_ID = "test_lexical_hash"
TEST_DIMENSION = 384


def _sha256_floats(seed: bytes, dimension: int) -> list[float]:
    """Deterministic finite floats in [-1, 1] from a seed."""
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for i in range(0, len(digest), 4):
            if len(values) >= dimension:
                break
            # Map 32-bit unsigned integer to a float in [-1, 1].
            u32 = struct.unpack(">I", digest[i : i + 4])[0]
            values.append(u32 / 0xFFFFFFFF * 2 - 1)
        counter += 1
    return values


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _vector_checksum(vector: list[float]) -> str:
    """Stable checksum over a canonical JSON serialization of the vector."""
    canonical = json.dumps(vector, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EmbeddingProvider(ABC):
    """Provider-neutral embedding contract.

    Subclasses declare their immutable model metadata as class attributes and
    implement ``embed`` to return one ``EmbeddingRecord`` per input text.
    """

    model_id: str
    model_revision: str
    tokenizer_revision: str
    dimension: int
    preprocessing_version: str
    normalized: bool = False
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    test_only: bool = False

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        metadata: dict[str, Any] | list[dict[str, Any]],
        *,
        object_type: ObjectType = ObjectType.CHUNK,
        task: EmbeddingTask = EmbeddingTask.DOCUMENT_RETRIEVAL,
        available_at: datetime | None = None,
    ) -> list[EmbeddingRecord]:
        """Return an ``EmbeddingRecord`` for every text in ``texts``.

        ``metadata`` may be a single dict applied to every text, or a list of
        dicts with length ``len(texts)`` for per-text metadata.  The provider
        must validate that every produced vector has the declared dimension
        and contains only finite values.
        """

    def _build_record(
        self,
        text: str,
        object_id: str,
        vector: list[float],
        *,
        object_type: ObjectType,
        task: EmbeddingTask,
        available_at: datetime,
        metadata: dict[str, Any],
    ) -> EmbeddingRecord:
        now = datetime.now(UTC)
        final_vector = _normalize(vector) if self.normalized else vector
        return EmbeddingRecord(
            embedding_id="emb_" + str(uuid.uuid4()),
            object_id=object_id,
            object_type=object_type,
            task=task,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            dimension=self.dimension,
            preprocessing_version=self.preprocessing_version,
            normalized=self.normalized,
            similarity_metric=self.similarity_metric,
            created_at=now,
            vector_checksum=_vector_checksum(final_vector),
            vector=final_vector,
            available_at=available_at,
            metadata=metadata,
        )


class TestEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock provider for offline tests.

    Vectors are derived from a lexical hash of the input text.  The provider is
    normalized and produces 384-dimensional cosine-compatible vectors.  It is
    marked ``test_only`` and therefore excluded from ``available_providers()``.
    """

    model_id = TEST_MODEL_ID
    model_revision = "1"
    tokenizer_revision = "1"
    dimension = TEST_DIMENSION
    preprocessing_version = "whitespace_lower_v1"
    normalized = True
    similarity_metric = SimilarityMetric.COSINE
    test_only = True

    def embed(
        self,
        texts: list[str],
        metadata: dict[str, Any] | list[dict[str, Any]],
        *,
        object_type: ObjectType = ObjectType.CHUNK,
        task: EmbeddingTask = EmbeddingTask.DOCUMENT_RETRIEVAL,
        available_at: datetime | None = None,
    ) -> list[EmbeddingRecord]:
        if available_at is None:
            available_at = datetime.now(UTC)
        meta_list: list[dict[str, Any]]
        if isinstance(metadata, dict):
            meta_list = [metadata] * len(texts)
        else:
            if len(metadata) != len(texts):
                raise ValueError("metadata list length must equal texts length")
            meta_list = metadata

        records: list[EmbeddingRecord] = []
        for text, meta in zip(texts, meta_list, strict=True):
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            vector = _normalize(_sha256_floats(seed, self.dimension))
            records.append(
                self._build_record(
                    text=text,
                    object_id=meta.get("object_id", "obj_test"),
                    vector=vector,
                    object_type=object_type,
                    task=task,
                    available_at=available_at,
                    metadata=meta,
                )
            )
        return records


def available_providers() -> dict[str, EmbeddingProvider]:
    """Return application-selectable providers (test providers excluded)."""
    return {}

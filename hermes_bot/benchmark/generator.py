"""Seeded, deterministic synthetic dataset generator for local benchmarking.

The generator is bounded and never touches the network or persistent storage.
All identifiers are derived from the seed so the same ``(n, seed)`` pair always
produces the same logical dataset.
"""
from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from hermes_bot.retrieval.provider import TestEmbeddingProvider
from hermes_bot.schemas import (
    CanonicalEntity,
    Document,
    EntityType,
    EpistemicType,
    Event,
    RelationEdge,
    RelationType,
    SourceKind,
    SourceRecord,
)

# Fixed wall-clock moment for the entire synthetic history.  Using a constant
# value keeps the benchmark reproducible and avoids accidental dependence on
# the current machine time.
_EPOCH = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_RELATIONS = tuple(RelationType)
_ENTITY_TYPES = (
    EntityType.COMPANY,
    EntityType.ORGANIZATION,
    EntityType.INDUSTRY,
    EntityType.COUNTRY,
    EntityType.CURRENCY,
)


def _uuid(seed: str, suffix: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{seed}.{suffix}"))


def _entity_id(seed: str, suffix: str) -> str:
    return f"ent_{_uuid(seed, suffix)}"


def _edge_id(seed: str, suffix: str) -> str:
    return f"edg_{_uuid(seed, suffix)}"


def _doc_id(seed: str, suffix: str) -> str:
    digest = hashlib.sha256(f"{seed}.{suffix}".encode()).hexdigest()
    return f"doc_{digest}"


def _source_id(seed: str, suffix: str) -> str:
    return f"src_{_uuid(seed, suffix)}"


@dataclass(frozen=True)
class SyntheticDataset:
    """A deterministic synthetic workload."""

    n: int
    seed: int
    entities: list[CanonicalEntity]
    sources: list[SourceRecord]
    documents: list[Document]
    events: list[Event]
    edges: list[RelationEdge]
    embeddings: list[Any]  # hermes_bot.schemas.EmbeddingRecord
    web_inputs: list[dict[str, Any]]
    reports: list[dict[str, Any]]

    @property
    def entity_ids(self) -> list[str]:
        return [e.entity_id for e in self.entities]


def _generate_text(rng: random.Random, topic: str, index: int, length: int = 80) -> str:
    words = [
        "market", "growth", "demand", "supply", "policy", "rates", "inflation",
        "revenue", "cost", "margin", "forecast", "expansion", "contraction",
        "AI", "chips", "semiconductor", "energy", "finance", "trade", "labour",
    ]
    rng.shuffle(words)
    sentence = " ".join(words[: max(5, length // 7)])
    return f"{topic} {index}: {sentence}."


def generate_dataset(n: int, seed: int = 42) -> SyntheticDataset:
    """Create a deterministic synthetic dataset of size ``n``.

    Parameters
    ----------
    n:
        Number of entities/documents/events/edges/chunks to generate.
    seed:
        Random seed used for all stochastic choices.

    Returns
    -------
    A ``SyntheticDataset`` with ``n`` primary records plus derived records.
    """
    if n < 1:
        raise ValueError("dataset size must be >= 1")

    rng = random.Random(seed)
    provider = TestEmbeddingProvider()

    seed_token = f"benchmark-seed-{seed}"

    entities: list[CanonicalEntity] = []
    for i in range(n):
        entity = CanonicalEntity(
            entity_id=_entity_id(seed_token, f"entity-{i}"),
            entity_type=rng.choice(_ENTITY_TYPES),
            canonical_name=f"Synthetic Entity {i}",
            created_at=_EPOCH,
            valid_from=_EPOCH,
            metadata={"seed_index": i},
        )
        entities.append(entity)

    sources: list[SourceRecord] = []
    documents: list[Document] = []
    texts: list[str] = []
    for i in range(n):
        source = SourceRecord(
            source_id=_source_id(seed_token, f"source-{i}"),
            source_uri=f"https://example.test/source/{i}",
            source_kind=rng.choice(list(SourceKind)),
            published_time=_EPOCH,
            ingested_at=_EPOCH,
            available_at=_EPOCH,
            metadata={"seed_index": i},
        )
        sources.append(source)

        text = _generate_text(rng, "report", i)
        texts.append(text)
        doc = Document(
            document_id=_doc_id(seed_token, f"doc-{i}"),
            source_id=source.source_id,
            source_uri=source.source_uri,
            raw_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            language="en",
            document_type="report",
            published_time=_EPOCH,
            ingested_at=_EPOCH,
            available_at=_EPOCH,
            metadata={"seed_index": i},
        )
        documents.append(doc)

    events: list[Event] = []
    for i in range(n):
        events.append(
            Event(
                event_id=f"evt_{_uuid(seed_token, f'event-{i}')}",
                event_type="macro",
                normalized_claim_text=texts[i][:60],
                event_time=_EPOCH,
                published_time=_EPOCH,
                available_at=_EPOCH,
                epistemic_type=EpistemicType.FORECAST,
                novelty=round(rng.uniform(0.0, 1.0), 4),
                extractor_version="benchmark/v1",
                source_document_ids=[documents[i % len(documents)].document_id],
                affected_entity_ids=[entities[i % len(entities)].entity_id],
                created_at=_EPOCH,
            )
        )

    edges: list[RelationEdge] = []
    relation_cycle = rng.sample(_RELATIONS, min(n, len(_RELATIONS))) if n > 1 else _RELATIONS
    for i in range(n):
        from_idx = i % len(entities)
        to_idx = (i + 1) % len(entities)
        edges.append(
            RelationEdge(
                edge_id=_edge_id(seed_token, f"edge-{i}"),
                from_entity_id=entities[from_idx].entity_id,
                relation_type=relation_cycle[i % len(relation_cycle)],
                to_entity_id=entities[to_idx].entity_id,
                available_at=_EPOCH,
                evidence_ids=[documents[i % len(documents)].document_id],
                source_ids=[sources[i % len(sources)].source_id],
                method_version="benchmark/v1",
                created_at=_EPOCH,
            )
        )

    # Add a few longer random edges so the graph is not a single cycle.
    extra_edges = min(n // 5, n)
    for i in range(extra_edges):
        from_idx = rng.randrange(len(entities))
        to_idx = rng.randrange(len(entities))
        if from_idx == to_idx:
            continue
        edges.append(
            RelationEdge(
                edge_id=_edge_id(seed_token, f"edge-extra-{i}"),
                from_entity_id=entities[from_idx].entity_id,
                relation_type=rng.choice(_RELATIONS),
                to_entity_id=entities[to_idx].entity_id,
                available_at=_EPOCH,
                evidence_ids=[documents[from_idx % len(documents)].document_id],
                source_ids=[sources[from_idx % len(sources)].source_id],
                method_version="benchmark/v1",
                created_at=_EPOCH,
            )
        )

    meta = [
        {
            "object_id": doc.document_id,
            "start_offset": 0,
            "end_offset": len(text),
            "section": "body",
        }
        for doc, text in zip(documents, texts, strict=True)
    ]
    raw_embeddings = provider.embed(texts, meta, available_at=_EPOCH)
    # The T06 provider stamps every record with a random uuid4 id and the
    # current wall-clock time.  Rebuild each record with a seed-derived id and
    # the fixed epoch so the dataset is reproducible as a whole.  Vector
    # content and all other fields are left untouched.
    embeddings = [
        record.model_copy(
            update={
                "embedding_id": f"emb_{_uuid(seed_token, f'embedding-{i}')}",
                "created_at": _EPOCH,
            }
        )
        for i, record in enumerate(raw_embeddings)
    ]

    web_inputs: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for i in range(min(n, 20)):
        inp = {
            "source": rng.choice(["speech", "report", "alert"]),
            "entity_id": entities[i % len(entities)].entity_id,
            "headline": f"Headline {i}",
            "body": texts[i % len(texts)],
            "sentiment": round(rng.uniform(-1.0, 1.0), 4),
            "confidence": round(rng.uniform(0.5, 1.0), 4),
        }
        web_inputs.append(inp)
        if inp["source"] == "report":
            reports.append(inp)

    return SyntheticDataset(
        n=n,
        seed=seed,
        entities=entities,
        sources=sources,
        documents=documents,
        events=events,
        edges=edges,
        embeddings=embeddings,
        web_inputs=web_inputs,
        reports=reports,
    )

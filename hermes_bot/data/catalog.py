"""Typed repositories and validation for the canonical SQLite catalog (T04).

Decimal representation
----------------------
Prices, quantities and weights are stored as ``REAL`` values in the named
currency.  Returns and portfolio weights are NAV fractions (``0.10`` = 10%).
JSON metadata/evidence lists are serialized to TEXT.  Timestamps are
timezone-aware UTC ISO-8601 strings at the storage boundary.

Validation
----------
Pydantic models in ``hermes_bot.schemas`` enforce ID formats and field ranges.
Repositories add temporal-interval validation and relation-evidence checks.
Foreign-key and uniqueness constraints are enforced by SQLite with
``PRAGMA foreign_keys = ON``.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any, TypeVar

from hermes_bot.schemas import (
    Alias,
    AssetClass,
    CanonicalEntity,
    DecisionRecord,
    Document,
    EntityType,
    EpistemicType,
    Event,
    Instrument,
    InstrumentType,
    RelationEdge,
    RelationType,
    ResolutionStatus,
    SourceKind,
    SourceRecord,
    StrengthSemantics,
)


class CatalogValidationError(ValueError):
    """Raised when a catalog record violates a temporal or evidence rule."""


T = TypeVar("T")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _parse_dt_required(value: str | None) -> datetime:
    if value is None:
        raise CatalogValidationError("expected non-null timestamp")
    return datetime.fromisoformat(value)


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def _parse_json(value: str | None) -> dict[str, object] | list[object]:
    if value is None:
        return {}
    return json.loads(value)


def _parse_metadata(value: str | None) -> dict[str, object]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise CatalogValidationError("metadata field must be a JSON object")
    return parsed


def _parse_str_list(value: str | None) -> list[str]:
    if value is None:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise CatalogValidationError("expected JSON list of strings")
    return [str(item) for item in parsed]


def validate_interval(valid_from: datetime | None, valid_to: datetime | None) -> None:
    """Ensure a temporal interval is well-formed.

    Both bounds are optional, but if both are present ``valid_to`` must be
    strictly after ``valid_from``.
    """
    if valid_from is not None and valid_to is not None and valid_to <= valid_from:
        raise CatalogValidationError(
            f"valid_to ({valid_to}) must be after valid_from ({valid_from})"
        )


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class _Repository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn


class EntityRepository(_Repository):
    def insert(self, entity: CanonicalEntity) -> None:
        validate_interval(entity.valid_from, entity.valid_to)
        self.conn.execute(
            """
            INSERT INTO entity
                (entity_id, entity_type, canonical_name, valid_from, valid_to,
                 metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.entity_id,
                entity.entity_type.value,
                entity.canonical_name,
                _iso(entity.valid_from),
                _iso(entity.valid_to),
                _json(entity.metadata),
                _iso(entity.created_at),
            ),
        )

    def get(self, entity_id: str) -> CanonicalEntity | None:
        row = self.conn.execute(
            "SELECT * FROM entity WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return None
        return CanonicalEntity(
            entity_id=row["entity_id"],
            entity_type=EntityType(row["entity_type"]),
            canonical_name=row["canonical_name"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_to=_parse_dt(row["valid_to"]),
            metadata=_parse_metadata(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
        )


class InstrumentRepository(_Repository):
    def insert(self, instrument: Instrument) -> None:
        validate_interval(instrument.valid_from, instrument.valid_to)
        self.conn.execute(
            """
            INSERT INTO instrument
                (instrument_id, issuer_entity_id, asset_class, exchange_mic, currency,
                 instrument_type, canonical_name, valid_from, valid_to, metadata,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                instrument.instrument_id,
                instrument.issuer_entity_id,
                instrument.asset_class.value,
                instrument.exchange_mic,
                instrument.currency,
                instrument.instrument_type.value,
                instrument.canonical_name,
                _iso(instrument.valid_from),
                _iso(instrument.valid_to),
                _json(instrument.metadata),
                _iso(instrument.created_at),
            ),
        )

    def get(self, instrument_id: str) -> Instrument | None:
        row = self.conn.execute(
            "SELECT * FROM instrument WHERE instrument_id = ?", (instrument_id,)
        ).fetchone()
        if row is None:
            return None
        return Instrument(
            instrument_id=row["instrument_id"],
            issuer_entity_id=row["issuer_entity_id"],
            asset_class=AssetClass(row["asset_class"]),
            exchange_mic=row["exchange_mic"],
            currency=row["currency"],
            instrument_type=InstrumentType(row["instrument_type"]),
            canonical_name=row["canonical_name"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_to=_parse_dt(row["valid_to"]),
            metadata=_parse_metadata(row["metadata"]),
            created_at=_parse_dt_required(row["created_at"]),
        )


class AliasRepository(_Repository):
    def insert(self, alias: Alias) -> None:
        validate_interval(alias.valid_from, alias.valid_to)
        self.conn.execute(
            """
            INSERT INTO alias
                (alias, namespace, entity_id, valid_from, valid_to,
                 resolution_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alias.alias,
                alias.namespace,
                alias.entity_id,
                _iso(alias.valid_from),
                _iso(alias.valid_to),
                alias.resolution_status.value,
                _iso(alias.created_at),
            ),
        )

    def get_by_alias(self, alias: str, namespace: str) -> list[Alias]:
        rows = self.conn.execute(
            "SELECT * FROM alias WHERE alias = ? AND namespace = ?",
            (alias, namespace),
        ).fetchall()
        return [
            Alias(
                alias=row["alias"],
                namespace=row["namespace"],
                entity_id=row["entity_id"],
                valid_from=_parse_dt(row["valid_from"]),
                valid_to=_parse_dt(row["valid_to"]),
                resolution_status=ResolutionStatus(row["resolution_status"]),
                created_at=_parse_dt_required(row["created_at"]),
            )
            for row in rows
        ]


class SourceRecordRepository(_Repository):
    def insert(self, source: SourceRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO source_record
                (source_id, source_uri, source_kind, license_id, raw_object_path,
                 published_time, ingested_at, available_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source.source_id,
                source.source_uri,
                source.source_kind.value,
                source.license_id,
                source.raw_object_path,
                _iso(source.published_time),
                _iso(source.ingested_at),
                _iso(source.available_at),
                _json(source.metadata),
            ),
        )

    def get(self, source_id: str) -> SourceRecord | None:
        row = self.conn.execute(
            "SELECT * FROM source_record WHERE source_id = ?", (source_id,)
        ).fetchone()
        if row is None:
            return None
        return SourceRecord(
            source_id=row["source_id"],
            source_uri=row["source_uri"],
            source_kind=SourceKind(row["source_kind"]),
            license_id=row["license_id"],
            raw_object_path=row["raw_object_path"],
            published_time=_parse_dt(row["published_time"]),
            ingested_at=_parse_dt_required(row["ingested_at"]),
            available_at=_parse_dt_required(row["available_at"]),
            metadata=_parse_metadata(row["metadata"]),
        )


class DocumentRepository(_Repository):
    def insert(self, document: Document) -> None:
        self.conn.execute(
            """
            INSERT INTO document
                (document_id, source_id, source_uri, raw_sha256, language,
                 document_type, published_time, ingested_at, available_at,
                 license_id, raw_object_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.document_id,
                document.source_id,
                document.source_uri,
                document.raw_sha256,
                document.language,
                document.document_type,
                _iso(document.published_time),
                _iso(document.ingested_at),
                _iso(document.available_at),
                document.license_id,
                document.raw_object_path,
                _json(document.metadata),
            ),
        )

    def get(self, document_id: str) -> Document | None:
        row = self.conn.execute(
            "SELECT * FROM document WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        return Document(
            document_id=row["document_id"],
            source_id=row["source_id"],
            source_uri=row["source_uri"],
            raw_sha256=row["raw_sha256"],
            language=row["language"],
            document_type=row["document_type"],
            published_time=_parse_dt(row["published_time"]),
            ingested_at=_parse_dt_required(row["ingested_at"]),
            available_at=_parse_dt_required(row["available_at"]),
            license_id=row["license_id"],
            raw_object_path=row["raw_object_path"],
            metadata=_parse_metadata(row["metadata"]),
        )


class EventRepository(_Repository):
    def insert(self, event: Event) -> None:
        self.conn.execute(
            """
            INSERT INTO event
                (event_id, event_type, normalized_claim_text, stance_direction,
                 event_time, published_time, available_at, epistemic_type, novelty,
                 extractor_version, source_document_ids, source_chunk_ids,
                 affected_entity_ids, uncertainty_fields, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                event.normalized_claim_text,
                event.stance_direction,
                _iso(event.event_time),
                _iso(event.published_time),
                _iso(event.available_at),
                event.epistemic_type.value,
                event.novelty,
                event.extractor_version,
                _json(event.source_document_ids),
                _json(event.source_chunk_ids),
                _json(event.affected_entity_ids),
                _json(event.uncertainty_fields),
                _iso(event.created_at),
            ),
        )

    def get(self, event_id: str) -> Event | None:
        row = self.conn.execute(
            "SELECT * FROM event WHERE event_id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        return Event(
            event_id=row["event_id"],
            event_type=row["event_type"],
            normalized_claim_text=row["normalized_claim_text"],
            stance_direction=row["stance_direction"],
            event_time=_parse_dt(row["event_time"]),
            published_time=_parse_dt(row["published_time"]),
            available_at=_parse_dt_required(row["available_at"]),
            epistemic_type=EpistemicType(row["epistemic_type"]),
            novelty=row["novelty"],
            extractor_version=row["extractor_version"],
            source_document_ids=_parse_str_list(row["source_document_ids"]),
            source_chunk_ids=_parse_str_list(row["source_chunk_ids"]),
            affected_entity_ids=_parse_str_list(row["affected_entity_ids"]),
            uncertainty_fields=_parse_metadata(row["uncertainty_fields"]),
            created_at=_parse_dt_required(row["created_at"]),
        )


class RelationEdgeRepository(_Repository):
    def insert(self, edge: RelationEdge) -> None:
        validate_interval(edge.valid_from, edge.valid_to)
        if not edge.evidence_ids:
            raise CatalogValidationError("relation_edge.evidence_ids must not be empty")
        self.conn.execute(
            """
            INSERT INTO relation_edge
                (edge_id, from_entity_id, relation_type, to_entity_id, valid_from,
                 valid_to, observed_at, available_at, evidence_ids, source_ids,
                 strength_value, strength_semantics, confidence, method_version,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.from_entity_id,
                edge.relation_type.value,
                edge.to_entity_id,
                _iso(edge.valid_from),
                _iso(edge.valid_to),
                _iso(edge.observed_at),
                _iso(edge.available_at),
                _json(edge.evidence_ids),
                _json(edge.source_ids),
                edge.strength_value,
                edge.strength_semantics.value,
                edge.confidence,
                edge.method_version,
                _iso(edge.created_at),
            ),
        )

    def get(self, edge_id: str) -> RelationEdge | None:
        row = self.conn.execute(
            "SELECT * FROM relation_edge WHERE edge_id = ?", (edge_id,)
        ).fetchone()
        if row is None:
            return None
        return RelationEdge(
            edge_id=row["edge_id"],
            from_entity_id=row["from_entity_id"],
            relation_type=RelationType(row["relation_type"]),
            to_entity_id=row["to_entity_id"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_to=_parse_dt(row["valid_to"]),
            observed_at=_parse_dt(row["observed_at"]),
            available_at=_parse_dt_required(row["available_at"]),
            evidence_ids=_parse_str_list(row["evidence_ids"]),
            source_ids=_parse_str_list(row["source_ids"]),
            strength_value=row["strength_value"],
            strength_semantics=StrengthSemantics(row["strength_semantics"]),
            confidence=row["confidence"],
            method_version=row["method_version"],
            created_at=_parse_dt_required(row["created_at"]),
        )


class DecisionRecordRepository(_Repository):
    def insert(self, decision: DecisionRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO decision_record
                (decision_id, instrument_id, target_weight, decision_time,
                 strategy_id, strategy_version, input_record_ids, reason_codes,
                 approval_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.instrument_id,
                decision.target_weight,
                _iso(decision.decision_time),
                decision.strategy_id,
                decision.strategy_version,
                _json(decision.input_record_ids),
                _json(decision.reason_codes),
                decision.approval_id,
                _iso(decision.created_at),
            ),
        )

    def get(self, decision_id: str) -> DecisionRecord | None:
        row = self.conn.execute(
            "SELECT * FROM decision_record WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            return None
        return DecisionRecord(
            decision_id=row["decision_id"],
            instrument_id=row["instrument_id"],
            target_weight=row["target_weight"],
            decision_time=_parse_dt_required(row["decision_time"]),
            strategy_id=row["strategy_id"],
            strategy_version=row["strategy_version"],
            input_record_ids=_parse_str_list(row["input_record_ids"]),
            reason_codes=_parse_str_list(row["reason_codes"]),
            approval_id=row["approval_id"],
            created_at=_parse_dt_required(row["created_at"]),
        )


class Catalog:
    """Bundle of all canonical repositories for convenience."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.entity = EntityRepository(conn)
        self.instrument = InstrumentRepository(conn)
        self.alias = AliasRepository(conn)
        self.source = SourceRecordRepository(conn)
        self.document = DocumentRepository(conn)
        self.event = EventRepository(conn)
        self.relation = RelationEdgeRepository(conn)
        self.decision = DecisionRecordRepository(conn)

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

"""Tests for the typed temporal economic graph (T07)."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hermes_bot.data.catalog import Catalog
from hermes_bot.data.graph import EconomicGraph, GraphQueryError
from hermes_bot.data.migrations import MigrationRunner
from hermes_bot.schemas import (
    CanonicalEntity,
    EntityType,
    RelationEdge,
    RelationType,
    StrengthSemantics,
)


def _entity_id(i: int) -> str:
    return f"ent_{uuid.uuid5(uuid.NAMESPACE_DNS, f'entity-{i}.test')}"


@pytest.fixture
def graph_db(tmp_path):
    """Fresh catalog DB pre-populated with 10 entities."""
    db_path = tmp_path / "graph.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    runner = MigrationRunner(conn)
    runner.migrate()

    catalog = Catalog(conn)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    entities = []
    for i in range(10):
        entity = CanonicalEntity(
            entity_id=_entity_id(i),
            entity_type=EntityType.COMPANY,
            canonical_name=f"Entity {i}",
            created_at=now,
            valid_from=now - timedelta(days=30),
        )
        entities.append(entity)
        catalog.entity.insert(entity)
    catalog.commit()
    yield conn, entities, now
    conn.close()


def _edge_id() -> str:
    return f"edg_{uuid.uuid4()}"


def _make_edge(
    from_id: str,
    to_id: str,
    relation: RelationType,
    evidence: list[str],
    available_at: datetime,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    strength_semantics: StrengthSemantics = StrengthSemantics.UNKNOWN,
    confidence: float | None = None,
) -> RelationEdge:
    return RelationEdge(
        edge_id=_edge_id(),
        from_entity_id=from_id,
        relation_type=relation,
        to_entity_id=to_id,
        available_at=available_at,
        valid_from=valid_from,
        valid_to=valid_to,
        evidence_ids=evidence,
        source_ids=[],
        strength_semantics=strength_semantics,
        confidence=confidence,
        method_version="t07/v1",
        created_at=available_at,
    )


def test_relation_type_rejects_unknown_string(graph_db):
    conn, entities, now = graph_db
    graph = EconomicGraph(conn)
    with pytest.raises(GraphQueryError):
        graph.traverse(
            entities[0].entity_id,
            as_of=now,
            max_hops=1,
            allowed_types=["not_a_relation"],  # type: ignore[list-item]
        )


def test_allowed_types_filter(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0, e1, e2 = entities[0].entity_id, entities[1].entity_id, entities[2].entity_id

    catalog.relation.insert(_make_edge(e0, e1, RelationType.ISSUER_OF, ["ev1"], now))
    catalog.relation.insert(
        _make_edge(e0, e2, RelationType.SUBSIDIARY_OF, ["ev2"], now)
    )
    catalog.commit()

    graph = EconomicGraph(conn)
    paths = graph.traverse(e0, as_of=now, max_hops=1, allowed_types={RelationType.ISSUER_OF})
    assert len(paths) == 2  # zero-hop + one match
    hop = paths[1].hops[0]
    assert hop.relation_type == RelationType.ISSUER_OF
    assert hop.to_entity_id == e1


def test_temporal_valid_expired_future(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0, e1, e2, e3 = (
        entities[0].entity_id,
        entities[1].entity_id,
        entities[2].entity_id,
        entities[3].entity_id,
    )

    valid = now - timedelta(days=1)
    expired = now - timedelta(days=10)
    future = now + timedelta(days=1)

    # Valid at as_of
    catalog.relation.insert(
        _make_edge(e0, e1, RelationType.LOCATED_IN, ["ev_valid"], now, valid_from=valid)
    )
    # Expired before as_of
    catalog.relation.insert(
        _make_edge(
            e0,
            e2,
            RelationType.LOCATED_IN,
            ["ev_expired"],
            now,
            valid_from=expired,
            valid_to=valid,
        )
    )
    # Future valid_from
    catalog.relation.insert(
        _make_edge(e0, e3, RelationType.LOCATED_IN, ["ev_future"], now, valid_from=future)
    )
    catalog.commit()

    graph = EconomicGraph(conn)
    paths = graph.traverse(e0, as_of=now, max_hops=1)
    targets = {p.path[-1] for p in paths if len(p.path) > 1}
    assert targets == {e1}
    assert all("ev_valid" in p.evidence_ids for p in paths if len(p.path) > 1)


def test_availability_filter(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0, e1 = entities[0].entity_id, entities[1].entity_id

    # Not available until after as_of
    catalog.relation.insert(
        _make_edge(e0, e1, RelationType.LOCATED_IN, ["ev_late"], now + timedelta(hours=1))
    )
    catalog.commit()

    graph = EconomicGraph(conn)
    paths = graph.traverse(e0, as_of=now, max_hops=1)
    assert all(len(p.path) == 1 for p in paths)


def test_cycle_prevention(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0 = entities[0].entity_id
    e1 = entities[1].entity_id
    e2 = entities[2].entity_id

    # A -> B -> C -> A cycle
    catalog.relation.insert(_make_edge(e0, e1, RelationType.CUSTOMER_OF, ["e1"], now))
    catalog.relation.insert(_make_edge(e1, e2, RelationType.CUSTOMER_OF, ["e2"], now))
    catalog.relation.insert(_make_edge(e2, e0, RelationType.CUSTOMER_OF, ["e3"], now))
    catalog.commit()

    graph = EconomicGraph(conn)
    paths = graph.traverse(e0, as_of=now, max_hops=3)
    # Zero-hop + A->B + A->B->C + A->B->C->A blocked by cycle
    for p in paths:
        # No repeated entity ids in any returned path
        assert len(p.path) == len(set(p.path)), p
    # With max_hops=3 we should still get A->B->C (3 nodes, 2 hops)
    assert any(p.path == (e0, e1, e2) for p in paths)
    # Should not contain the cyclic path back to A
    assert not any(p.path[-1] == e0 and len(p.path) > 1 for p in paths)


def test_evidence_accumulated_in_path(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0, e1, e2 = entities[0].entity_id, entities[1].entity_id, entities[2].entity_id

    catalog.relation.insert(_make_edge(e0, e1, RelationType.SUPPLIES, ["ev_a"], now))
    catalog.relation.insert(_make_edge(e1, e2, RelationType.SUPPLIES, ["ev_b"], now))
    catalog.commit()

    graph = EconomicGraph(conn)
    paths = graph.traverse(e0, as_of=now, max_hops=2)
    long = next(p for p in paths if p.path == (e0, e1, e2))
    assert long.evidence_ids == ("ev_a", "ev_b")
    assert [h.relation_type for h in long.hops] == [RelationType.SUPPLIES, RelationType.SUPPLIES]


def test_max_hops_boundaries(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0, e1, e2, e3 = (
        entities[0].entity_id,
        entities[1].entity_id,
        entities[2].entity_id,
        entities[3].entity_id,
    )
    catalog.relation.insert(_make_edge(e0, e1, RelationType.INDICATOR_MEASURES, ["1"], now))
    catalog.relation.insert(_make_edge(e1, e2, RelationType.INDICATOR_MEASURES, ["2"], now))
    catalog.relation.insert(_make_edge(e2, e3, RelationType.INDICATOR_MEASURES, ["3"], now))
    catalog.commit()

    graph = EconomicGraph(conn)

    for hops, expected_depths in [
        (0, {1}),
        (1, {1, 2}),
        (2, {1, 2, 3}),
        (3, {1, 2, 3, 4}),
    ]:
        paths = graph.traverse(e0, as_of=now, max_hops=hops)
        assert {len(p.path) for p in paths} == expected_depths

    with pytest.raises(GraphQueryError):
        graph.traverse(e0, as_of=now, max_hops=4)
    with pytest.raises(GraphQueryError):
        graph.traverse(e0, as_of=now, max_hops=-1)


def test_direction(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0, e1 = entities[0].entity_id, entities[1].entity_id

    catalog.relation.insert(_make_edge(e0, e1, RelationType.MEMBER_OF_SECTOR, ["ev_dir"], now))
    catalog.commit()

    graph = EconomicGraph(conn)
    out_paths = graph.traverse(e0, as_of=now, max_hops=1, direction="out")
    assert any(p.path == (e0, e1) for p in out_paths)

    in_paths = graph.traverse(e1, as_of=now, max_hops=1, direction="in")
    assert any(p.path == (e1, e0) for p in in_paths)

    both_paths = graph.traverse(e0, as_of=now, max_hops=1, direction="both")
    assert any(p.path == (e0, e1) for p in both_paths)


def test_missing_start_entity(graph_db):
    conn, entities, now = graph_db
    graph = EconomicGraph(conn)
    missing_id = _entity_id(99)
    assert missing_id not in {e.entity_id for e in entities}

    paths = graph.traverse(missing_id, as_of=now, max_hops=1)
    assert len(paths) == 1
    assert paths[0].path == (missing_id,)
    assert paths[0].hops == ()


def test_deterministic_ordering(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0 = entities[0].entity_id
    for i in range(1, 6):
        catalog.relation.insert(
            _make_edge(e0, entities[i].entity_id, RelationType.CURRENCY_OF, [f"e{i}"], now)
        )
    catalog.commit()

    graph = EconomicGraph(conn)
    first = graph.traverse(e0, as_of=now, max_hops=1)
    for _ in range(5):
        assert graph.traverse(e0, as_of=now, max_hops=1) == first


def test_unknown_strength_is_null(graph_db):
    conn, entities, now = graph_db
    catalog = Catalog(conn)
    e0, e1 = entities[0].entity_id, entities[1].entity_id

    edge = _make_edge(e0, e1, RelationType.EXPOSED_TO, ["ev_s"], now)
    catalog.relation.insert(edge)
    catalog.commit()

    fetched = catalog.relation.get(edge.edge_id)
    assert fetched is not None
    assert fetched.strength_value is None
    assert fetched.strength_semantics == StrengthSemantics.UNKNOWN

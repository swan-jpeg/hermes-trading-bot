"""Tests for the T18 weighted multi-method correlation graph.

Fully offline and deterministic: 10 synthetic nodes, the T06
``TestEmbeddingProvider`` (hash-based mock, no model download), seeded
synthetic return series, temporary SQLite databases only.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from hermes_bot.correlation import (
    MAX_EDGES_PER_NODE,
    CorrelationEdge,
    CorrelationGraph,
    CorrelationNode,
    CorrelationRepository,
    DynamicConditionalCorrelation,
    KendallTau,
    RollingPearson,
    Spearman,
    build_edge,
    default_methods,
    method_weights,
)
from hermes_bot.data.migrations import MIGRATIONS, MigrationRunner
from hermes_bot.retrieval.provider import TestEmbeddingProvider

AS_OF = datetime(2026, 1, 1, tzinfo=UTC)
N_OBS = 60


def _provider() -> TestEmbeddingProvider:
    return TestEmbeddingProvider()


def _make_nodes(n: int, provider: TestEmbeddingProvider) -> list[CorrelationNode]:
    nodes = []
    for i in range(n):
        record = provider.embed([f"node:{i}"], {"object_id": f"node:{i}"})[0]
        nodes.append(
            CorrelationNode(
                node_id=f"n{i}", embedding=list(record.vector), node_type="asset", as_of=AS_OF
            )
        )
    return nodes


def _make_series(n: int, n_obs: int = N_OBS) -> dict[str, list[float]]:
    """Seeded synthetic return series; n0/n1 strongly correlated."""
    rng = np.random.default_rng(42)
    series: dict[str, list[float]] = {}
    base = rng.normal(0.0, 1.0, size=n_obs)
    for i in range(n):
        noise = rng.normal(0.0, 1.0, size=n_obs)
        if i == 1:
            values = 0.9 * base + 0.1 * noise
        elif i == 0:
            values = base
        else:
            values = noise
        series[f"n{i}"] = [float(v) for v in values]
    return series


def _manual_edge(
    edge_id: str,
    a: str,
    b: str,
    score: float,
    certainty: float = 1.0,
    dim: int = 4,
) -> CorrelationEdge:
    """A handcrafted edge with an exact score (for the anti-double-count tests)."""
    weights = {"rolling_pearson": 1.0}
    return CorrelationEdge(
        edge_id=edge_id,
        node_a=a,
        node_b=b,
        as_of=AS_OF,
        window=20,
        method_scores={"rolling_pearson": score},
        method_weights=weights,
        correlation_score=score,
        certainty=certainty,
        relation_type="correlation",
        relation_embedding=[0.0] * dim,
        evidence={"sample_size": 60},
    )


# --------------------------------------------------------------------- 1. base
def test_base_graph_ten_nodes() -> None:
    provider = _provider()
    nodes = _make_nodes(10, provider)
    series = _make_series(10)
    methods = default_methods()
    graph = CorrelationGraph()
    for node in nodes:
        graph.add_node(node)
        graph.set_series(node.node_id, series[node.node_id])
    edges = [
        build_edge(
            edge_id=f"e0{i}",
            node_a=nodes[0],
            node_b=nodes[i],
            series_a=series["n0"],
            series_b=series[f"n{i}"],
            methods=methods,
            provider=provider,
            as_of=AS_OF,
            relation_type="sector",
        )
        for i in range(1, 10)
    ]
    for edge in edges:
        graph.add_edge(edge)
    assert len(graph.edges_for("n0")) == 9
    for edge in edges:
        assert -1.0 <= edge.correlation_score <= 1.0
        assert 0.0 <= edge.certainty <= 1.0
        assert math.isclose(sum(edge.method_weights.values()), 1.0, abs_tol=1e-9)
        assert len(edge.relation_embedding) == provider.dimension
        assert set(edge.method_scores) == set(methods)
        assert edge.evidence["relation_embedding_default"] is False
    # The strongly correlated pair n0-n1 must score high on every method.
    e01 = graph.get_edge("n0", "n1")
    assert e01 is not None
    assert e01.correlation_score > 0.8


# ----------------------------------------------------------- 2. embedding weights
def test_weights_depend_on_embeddings() -> None:
    provider = _provider()
    nodes = _make_nodes(3, provider)
    names = sorted(default_methods())
    w1 = method_weights(nodes[0].embedding, nodes[1].embedding, names)
    w2 = method_weights(nodes[0].embedding, nodes[2].embedding, names)
    assert w1 != w2, "changing node B's embedding must change the method weights"
    assert math.isclose(sum(w1.values()), 1.0, abs_tol=1e-12)
    # Determinism: identical inputs -> identical weights.
    w1_again = method_weights(nodes[0].embedding, nodes[1].embedding, names)
    assert w1 == w1_again

    # The full edge recomputes its weighted score from the new weights.
    series = _make_series(3)
    methods = default_methods()
    edge_b1 = build_edge(
        edge_id="e",
        node_a=nodes[0],
        node_b=nodes[1],
        series_a=series["n0"],
        series_b=series["n1"],
        methods=methods,
        provider=provider,
        as_of=AS_OF,
    )
    edge_b2 = build_edge(
        edge_id="e",
        node_a=nodes[0],
        node_b=nodes[2],
        series_a=series["n0"],
        series_b=series["n1"],
        methods=methods,
        provider=provider,
        as_of=AS_OF,
    )
    assert edge_b1.method_weights != edge_b2.method_weights
    assert edge_b1.correlation_score != edge_b2.correlation_score


# ------------------------------------------------------------------ 3. cap of 50
def test_cap_50_edges_per_node() -> None:
    hub = CorrelationNode(
        node_id="hub",
        embedding=[1.0, 0.0],
        node_type="asset",
        as_of=AS_OF,
    )
    graph = CorrelationGraph()
    graph.add_node(hub)
    n_candidates = MAX_EDGES_PER_NODE + 5
    for i in range(n_candidates):
        node = CorrelationNode(
            node_id=f"spoke{i}",
            embedding=[0.0, float(i + 1)],
            node_type="asset",
            as_of=AS_OF,
        )
        graph.add_node(node)
        # Strength increases monotonically with i (abs(score) * certainty),
        # so the weakest 5 must be dropped.
        score = (i + 1) / (n_candidates + 1)
        graph.add_edge(_manual_edge(f"e{i:03d}", "hub", f"spoke{i}", score))

    hub_edges = graph.edges_for("hub")
    assert len(hub_edges) == MAX_EDGES_PER_NODE
    remaining = {e.edge_id for e in hub_edges}
    dropped = {f"e{i:03d}" for i in range(n_candidates)} - remaining
    assert dropped == {"e000", "e001", "e002", "e003", "e004"}
    # Sorted by strength descending.
    strengths = [abs(e.correlation_score) * e.certainty for e in hub_edges]
    assert strengths == sorted(strengths, reverse=True)


# ----------------------------------------------------- 4. anti-double-counting
def test_effective_correlation_is_direct_only() -> None:
    graph = CorrelationGraph()
    for name in ("A", "B", "C"):
        graph.add_node(
            CorrelationNode(node_id=name, embedding=[1.0], node_type="asset", as_of=AS_OF)
        )
    graph.add_edge(_manual_edge("eAB", "A", "B", 0.9))
    graph.add_edge(_manual_edge("eBC", "B", "C", 0.9))

    # No direct A-C edge: NOT 1.8, NOT 0.81, just 0.0.
    assert graph.effective_correlation("A", "C") == 0.0
    assert graph.effective_correlation("A", "B") == 0.9
    assert graph.effective_correlation("B", "C") == 0.9

    # An explicit A-C edge returns exactly its own score.
    graph.add_edge(_manual_edge("eAC", "A", "C", 0.3))
    assert graph.effective_correlation("A", "C") == 0.3


def test_partial_correlation_removes_shared_neighbour() -> None:
    rng = np.random.default_rng(7)
    n_obs = 300
    b = rng.normal(0.0, 1.0, size=n_obs)
    a = 0.8 * b + 0.2 * rng.normal(0.0, 1.0, size=n_obs)
    c = 0.8 * b + 0.2 * rng.normal(0.0, 1.0, size=n_obs)

    graph = CorrelationGraph()
    for name in ("A", "B", "C"):
        graph.add_node(
            CorrelationNode(node_id=name, embedding=[1.0], node_type="asset", as_of=AS_OF)
        )
    graph.set_series("A", [float(v) for v in a])
    graph.set_series("B", [float(v) for v in b])
    graph.set_series("C", [float(v) for v in c])

    raw = graph.partial_correlation("A", "C", control=["D_missing"])
    assert raw == 0.0  # missing series -> explicit 0.0

    # Raw Pearson(A, C) is high through the shared driver B...
    mean_a, mean_c = float(a.mean()), float(c.mean())
    cov = float(((a - mean_a) * (c - mean_c)).sum())
    pearson = cov / math.sqrt(float(((a - mean_a) ** 2).sum() * ((c - mean_c) ** 2).sum()))
    assert pearson > 0.5
    # ...but after controlling for B the correlation collapses.
    partial = graph.partial_correlation("A", "C", control=["B"])
    assert abs(partial) < 0.2
    assert abs(partial - pearson) > 0.3


# ------------------------------------------------------------------- 5. methods
@pytest.mark.parametrize(
    "method",
    [RollingPearson(), Spearman(), KendallTau(), DynamicConditionalCorrelation()],
)
def test_methods_score_range_and_size(method) -> None:
    rng = np.random.default_rng(1)
    xs = [float(v) for v in rng.normal(0.0, 1.0, size=40)]
    ys = [float(x + 0.1 * e) for x, e in zip(xs, rng.normal(0.0, 1.0, size=40), strict=True)]
    result = method.score(xs, ys)
    assert -1.0 <= result.score <= 1.0
    assert result.sample_size >= method.min_observations
    # Strongly dependent series must be picked up by every method.
    assert result.score > 0.5


@pytest.mark.parametrize(
    "method",
    [RollingPearson(), Spearman(), KendallTau(), DynamicConditionalCorrelation()],
)
def test_methods_insufficient_data_is_explicit(method) -> None:
    result = method.score([1.0, 2.0], [1.0, 2.0])
    assert result.score == 0.0
    assert result.sample_size < method.min_observations


def test_rolling_pearson_respects_window() -> None:
    xs = [float(i) for i in range(50)]
    ys = [float(50 - i) for i in range(50)]
    result = RollingPearson(window=10).score(xs, ys)
    assert result.sample_size == 10
    assert result.score == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------- 6. persistence
def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row[0] for row in rows}


def test_migration_v3_fresh_database(tmp_path) -> None:
    conn = sqlite3.connect(str(tmp_path / "fresh.db"))
    try:
        applied = MigrationRunner(conn).migrate()
        assert applied == [0, 1, 2, 3]
        tables = _table_names(conn)
        assert {"correlation_node", "correlation_edge"} <= tables
        assert MigrationRunner(conn).current_version() == 3
    finally:
        conn.close()


def test_migration_v3_upgrade_preserves_earlier_tables(tmp_path) -> None:
    conn = sqlite3.connect(str(tmp_path / "upgrade.db"))
    try:
        early = [m for m in MIGRATIONS if m.version <= 2]
        assert MigrationRunner(conn, migrations=early).migrate() == [0, 1, 2]
        assert "entity" in _table_names(conn)
        applied = MigrationRunner(conn).migrate()
        assert applied == [3]
        tables = _table_names(conn)
        assert "entity" in tables  # earlier schema preserved
        assert "ledger_entry" in tables
        assert {"correlation_node", "correlation_edge"} <= tables
    finally:
        conn.close()


def test_repository_roundtrip_and_foreign_keys(tmp_path) -> None:
    provider = _provider()
    nodes = _make_nodes(2, provider)
    series = _make_series(2)
    conn = sqlite3.connect(str(tmp_path / "roundtrip.db"))
    try:
        MigrationRunner(conn).migrate()
        repo = CorrelationRepository(conn)
        edge = build_edge(
            edge_id="e01",
            node_a=nodes[0],
            node_b=nodes[1],
            series_a=series["n0"],
            series_b=series["n1"],
            methods=default_methods(),
            provider=provider,
            as_of=AS_OF,
            relation_type="supplier",
        )
        # FK enforcement: an edge without its nodes must fail.
        with pytest.raises(sqlite3.IntegrityError):
            repo.save_edge(edge)
        repo.save_node(nodes[0])
        repo.save_node(nodes[1])
        repo.save_edge(edge)
        conn.commit()

        loaded_node = repo.get_node("n0")
        assert loaded_node == nodes[0]
        loaded_edge = repo.get_edge("e01")
        assert loaded_edge == edge
        assert [e.edge_id for e in repo.edges_for_node("n1")] == ["e01"]
        assert repo.nodes() == ["n0", "n1"]

        # Upsert: saving the node again keeps a single row.
        repo.save_node(nodes[0])
        conn.commit()
        assert repo.nodes() == ["n0", "n1"]
    finally:
        conn.close()


# --------------------------------------------------------------------- 7. offline
def test_fully_offline_no_network(monkeypatch, tmp_path) -> None:
    """The whole flow must work with all network access hard-disabled."""

    def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("network access attempted")

    monkeypatch.setattr("socket.create_connection", _blocked)
    monkeypatch.setattr("urllib.request.urlopen", _blocked)

    provider = _provider()
    nodes = _make_nodes(10, provider)
    series = _make_series(10)
    methods = default_methods()
    graph = CorrelationGraph()
    for node in nodes:
        graph.add_node(node)
    for i in range(1, 10):
        graph.add_edge(
            build_edge(
                edge_id=f"e0{i}",
                node_a=nodes[0],
                node_b=nodes[i],
                series_a=series["n0"],
                series_b=series[f"n{i}"],
                methods=methods,
                provider=provider,
                as_of=AS_OF,
            )
        )
    conn = sqlite3.connect(str(tmp_path / "offline.db"))
    try:
        MigrationRunner(conn).migrate()
        repo = CorrelationRepository(conn)
        for node in nodes:
            repo.save_node(node)
        for edge in graph.edges_for("n0"):
            repo.save_edge(edge)
        conn.commit()
        assert len(repo.edges_for_node("n0")) == 9
    finally:
        conn.close()

"""SQLite persistence for the correlation graph (T18).

Uses the T04 ``MigrationRunner``; the schema lives in migration v3
(``correlation_node`` / ``correlation_edge``).  Upsert semantics: saving a
node or edge with an existing primary key replaces the row.  Tests only use
temporary databases (``tmp_path``); the production database is never touched.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from hermes_bot.correlation.schemas import CorrelationEdge, CorrelationNode


def _ts_now() -> str:
    return datetime.now(UTC).isoformat()


class CorrelationRepository:
    """Upsert/query repository over the v3 correlation tables."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------ nodes
    def save_node(self, node: CorrelationNode) -> None:
        self.conn.execute(
            """
            INSERT INTO correlation_node (node_id, node_type, embedding, as_of, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                node_type = excluded.node_type,
                embedding = excluded.embedding,
                as_of = excluded.as_of
            """,
            (
                node.node_id,
                node.node_type,
                json.dumps(node.embedding),
                node.as_of.isoformat(),
                _ts_now(),
            ),
        )

    def get_node(self, node_id: str) -> CorrelationNode | None:
        row = self.conn.execute(
            "SELECT node_id, node_type, embedding, as_of FROM correlation_node WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return CorrelationNode(
            node_id=row[0],
            node_type=row[1],
            embedding=json.loads(row[2]),
            as_of=datetime.fromisoformat(row[3]),
        )

    def nodes(self) -> list[str]:
        rows = self.conn.execute("SELECT node_id FROM correlation_node ORDER BY node_id").fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------ edges
    def save_edge(self, edge: CorrelationEdge) -> None:
        self.conn.execute(
            """
            INSERT INTO correlation_edge (
                edge_id, node_a, node_b, as_of, window,
                method_scores, method_weights, correlation_score, certainty,
                relation_type, relation_embedding, evidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET
                node_a = excluded.node_a,
                node_b = excluded.node_b,
                as_of = excluded.as_of,
                window = excluded.window,
                method_scores = excluded.method_scores,
                method_weights = excluded.method_weights,
                correlation_score = excluded.correlation_score,
                certainty = excluded.certainty,
                relation_type = excluded.relation_type,
                relation_embedding = excluded.relation_embedding,
                evidence = excluded.evidence
            """,
            (
                edge.edge_id,
                edge.node_a,
                edge.node_b,
                edge.as_of.isoformat(),
                edge.window,
                json.dumps(edge.method_scores),
                json.dumps(edge.method_weights),
                edge.correlation_score,
                edge.certainty,
                edge.relation_type,
                json.dumps(edge.relation_embedding),
                json.dumps(edge.evidence),
                _ts_now(),
            ),
        )

    def _row_to_edge(self, row: tuple) -> CorrelationEdge:
        return CorrelationEdge(
            edge_id=row[0],
            node_a=row[1],
            node_b=row[2],
            as_of=datetime.fromisoformat(row[3]),
            window=row[4],
            method_scores=json.loads(row[5]),
            method_weights=json.loads(row[6]),
            correlation_score=row[7],
            certainty=row[8],
            relation_type=row[9],
            relation_embedding=json.loads(row[10]),
            evidence=json.loads(row[11]),
        )

    _EDGE_COLUMNS = (
        "edge_id, node_a, node_b, as_of, window, method_scores, method_weights,"
        " correlation_score, certainty, relation_type, relation_embedding, evidence"
    )

    def get_edge(self, edge_id: str) -> CorrelationEdge | None:
        row = self.conn.execute(
            f"SELECT {self._EDGE_COLUMNS} FROM correlation_edge WHERE edge_id = ?",
            (edge_id,),
        ).fetchone()
        return self._row_to_edge(row) if row is not None else None

    def edges_for_node(self, node_id: str) -> list[CorrelationEdge]:
        """All edges touching the node (either side), sorted deterministically."""
        rows = self.conn.execute(
            f"SELECT {self._EDGE_COLUMNS} FROM correlation_edge"
            " WHERE node_a = ? OR node_b = ? ORDER BY edge_id",
            (node_id, node_id),
        ).fetchall()
        return [self._row_to_edge(row) for row in rows]

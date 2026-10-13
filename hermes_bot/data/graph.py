"""Typed temporal economic graph traversal over the canonical catalog (T07).

The implementation deliberately lives next to the T04 catalog.  All graph
state is stored in the ``relation_edge`` table; this module adds only query
logic and does not create edges by itself.  Traversal is deterministic,
evidence-linked and bounded.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from hermes_bot.schemas import RelationType

MAX_TRAVERSAL_HOPS = 3


class GraphQueryError(ValueError):
    """Raised when a graph query violates T07 boundaries or type constraints."""


@dataclass(frozen=True)
class GraphHop:
    """One directed hop as part of a traversal result path."""

    edge_id: str
    relation_type: RelationType
    from_entity_id: str
    to_entity_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class GraphPath:
    """A bounded traversal result: entities visited and evidence gathered."""

    path: tuple[str, ...]
    hops: tuple[GraphHop, ...]
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)


class EconomicGraph:
    """Query-time view of the typed temporal relation graph.

    The graph is read-only and point-in-time: every traversal must supply an
    explicit ``as_of`` timestamp, and only edges whose validity and availability
    include that moment are considered.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    def _normalize_types(
        self, allowed: Iterable[RelationType] | None
    ) -> tuple[RelationType, ...]:
        if allowed is None:
            return tuple(RelationType)
        allowed_set = tuple(allowed)
        if not allowed_set:
            raise GraphQueryError("allowed relation types must not be empty")
        for rt in allowed_set:
            if not isinstance(rt, RelationType):
                raise GraphQueryError(
                    f"invalid relation type {rt!r}; use RelationType enum"
                )
        return allowed_set

    @staticmethod
    def _iso(ts: datetime) -> str:
        return ts.isoformat()

    def traverse(
        self,
        from_entity_id: str,
        as_of: datetime,
        max_hops: int,
        direction: str = "out",
        allowed_types: Iterable[RelationType] | None = None,
    ) -> list[GraphPath]:
        """Return all bounded paths starting at ``from_entity_id``.

        Parameters
        ----------
        from_entity_id:
            Entity where the traversal starts.
        as_of:
            Point-in-time used to filter evidence availability and validity.
        max_hops:
            Hard cap on path length; must be between 0 and 3 inclusive.
        direction:
            ``out`` follows ``from_entity_id -> to_entity_id``,
            ``in`` follows ``to_entity_id -> from_entity_id``,
            ``both`` follows both.
        allowed_types:
            Subset of ``RelationType`` to traverse; defaults to all ten
            economic relation types.

        Returns
        -------
        List of ``GraphPath`` objects, sorted deterministically.  The first
        path is always the zero-hop path containing only ``from_entity_id``.
        For multi-hop paths ``evidence_ids`` aggregates evidence from every
        edge along the path.
        """
        if not isinstance(max_hops, int):
            raise GraphQueryError("max_hops must be an integer")
        if not 0 <= max_hops <= MAX_TRAVERSAL_HOPS:
            raise GraphQueryError(
                f"max_hops must be between 0 and {MAX_TRAVERSAL_HOPS}, got {max_hops}"
            )
        if direction not in ("out", "in", "both"):
            raise GraphQueryError(f"direction must be out/in/both, got {direction!r}")

        types = self._normalize_types(allowed_types)

        start_path = GraphPath(path=(from_entity_id,), hops=(), evidence_ids=())
        if max_hops == 0:
            return [start_path]

        results: list[GraphPath] = [start_path]
        # Stack entries carry accumulated evidence/hops for the current prefix.
        stack: list[GraphPath] = [start_path]
        while stack:
            current = stack.pop()
            tail = current.path[-1]
            current_hops = len(current.path) - 1
            if current_hops >= max_hops:
                continue
            for hop in self._neighbors(
                tail,
                as_of=as_of,
                direction=direction,
                allowed_types=types,
            ):
                next_entity = hop.to_entity_id
                if next_entity in current.path:
                    continue
                new_evidence = current.evidence_ids + hop.evidence_ids
                new_hops = current.hops + (hop,)
                new_path = GraphPath(
                    path=current.path + (next_entity,),
                    hops=new_hops,
                    evidence_ids=new_evidence,
                )
                results.append(new_path)
                stack.append(new_path)

        return _sort_paths(results)

    def _neighbors(
        self,
        entity_id: str,
        as_of: datetime,
        direction: str,
        allowed_types: tuple[RelationType, ...],
    ) -> list[GraphHop]:
        """Return all admissible hops adjacent to ``entity_id``.

        The query is point-in-time:

        * ``available_at <= as_of`` (evidence is known by ``as_of``);
        * ``valid_from IS NULL OR valid_from <= as_of``;
        * ``valid_to IS NULL OR valid_to > as_of``.

        Deterministic ordering is by relation type, then target entity id, then
        edge id so repeated runs produce identical output.
        """
        as_of_iso = self._iso(as_of)
        type_literals = [rt.value for rt in allowed_types]

        if direction == "out":
            where = "from_entity_id = ?"
            params: list[object] = [entity_id]
        elif direction == "in":
            where = "to_entity_id = ?"
            params = [entity_id]
        else:
            where = "(from_entity_id = ? OR to_entity_id = ?)"
            params = [entity_id, entity_id]

        placeholders = ",".join("?" for _ in type_literals)
        sql = f"""
            SELECT
                edge_id, relation_type, from_entity_id, to_entity_id, evidence_ids
            FROM relation_edge
            WHERE {where}
              AND available_at <= ?
              AND relation_type IN ({placeholders})
              AND (valid_from IS NULL OR valid_from <= ?)
              AND (valid_to IS NULL OR valid_to > ?)
            ORDER BY relation_type ASC, to_entity_id ASC, edge_id ASC
        """
        params.extend([as_of_iso, *type_literals, as_of_iso, as_of_iso])
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_hop(row, entity_id) for row in rows]


def _row_to_hop(row: sqlite3.Row, start_entity_id: str) -> GraphHop:
    evidence_ids = tuple(str(item) for item in json.loads(row["evidence_ids"]))
    if row["from_entity_id"] == start_entity_id:
        return GraphHop(
            edge_id=row["edge_id"],
            relation_type=RelationType(row["relation_type"]),
            from_entity_id=row["from_entity_id"],
            to_entity_id=row["to_entity_id"],
            evidence_ids=evidence_ids,
        )
    # Traversing an incoming edge backwards: to -> from.
    return GraphHop(
        edge_id=row["edge_id"],
        relation_type=RelationType(row["relation_type"]),
        from_entity_id=row["to_entity_id"],
        to_entity_id=row["from_entity_id"],
        evidence_ids=evidence_ids,
    )


def _sort_paths(paths: list[GraphPath]) -> list[GraphPath]:
    """Sort paths for deterministic test output."""
    return sorted(
        paths,
        key=lambda p: (
            p.path,
            tuple(h.edge_id for h in p.hops),
            p.evidence_ids,
        ),
    )

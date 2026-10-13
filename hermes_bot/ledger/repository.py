"""Append-only ledger repository and health state persistence."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from hermes_bot.ledger.schemas import HealthState, HealthStatus, LedgerEntry, LedgerEntryType


class LedgerRepository:
    """Append-only ledger storage.

    Only INSERT operations are performed. Updates and deletes are intentionally
    not provided so the ledger remains append-only.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._ensure_row_factory()

    def _ensure_row_factory(self) -> None:
        if self.conn.row_factory is None:
            self.conn.row_factory = sqlite3.Row

    def _iso(self, dt: datetime | None) -> str | None:
        return dt.isoformat() if dt is not None else None

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _next_sequence(self, cycle_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM ledger_entry WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def insert_cycle(
        self,
        cycle_id: str,
        decision_time: datetime,
        strategy_id: str | None = None,
        strategy_version: str | None = None,
        proposal_hash: str | None = None,
    ) -> None:
        """Create a cycle header row."""
        self.conn.execute(
            """
            INSERT INTO ledger_cycle (
                cycle_id, decision_time, strategy_id, strategy_version, proposal_hash, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cycle_id,
                self._iso(decision_time),
                strategy_id,
                strategy_version,
                proposal_hash,
                self._iso(self._now()),
            ),
        )

    def _entry_to_row(self, entry: LedgerEntry) -> tuple[Any, ...]:
        return (
            entry.entry_id,
            entry.cycle_id,
            entry.sequence,
            entry.entry_type.value,
            self._iso(entry.decision_time),
            json.dumps(entry.source_document_ids),
            entry.dataset_version,
            entry.config_version,
            entry.strategy_id,
            entry.strategy_version,
            entry.proposal_hash,
            entry.approval_id,
            entry.request_hash,
            entry.order_intent_id,
            entry.idempotency_key,
            entry.broker_order_id,
            entry.broker_status,
            entry.instrument_id,
            entry.fill_quantity,
            entry.fill_price,
            entry.fill_fees,
            entry.portfolio_snapshot,
            entry.error_code,
            entry.recovery_action,
            entry.payload,
            self._iso(entry.created_at),
        )

    def insert(self, entry: LedgerEntry) -> None:
        """Append one ledger entry."""
        self.conn.execute(
            """
            INSERT INTO ledger_entry (
                entry_id, cycle_id, sequence, entry_type, decision_time, source_document_ids,
                dataset_version, config_version, strategy_id, strategy_version, proposal_hash,
                approval_id, request_hash, order_intent_id, idempotency_key, broker_order_id,
                broker_status, instrument_id, fill_quantity, fill_price, fill_fees,
                portfolio_snapshot, error_code, recovery_action, payload, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._entry_to_row(entry),
        )

    def get_cycle(self, cycle_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM ledger_cycle WHERE cycle_id = ?", (cycle_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_entries(self, cycle_id: str) -> list[LedgerEntry]:
        rows = self.conn.execute(
            """
            SELECT * FROM ledger_entry
            WHERE cycle_id = ?
            ORDER BY sequence ASC, created_at ASC
            """,
            (cycle_id,),
        ).fetchall()
        return [self._parse_row(row) for row in rows]

    def get_entry(self, entry_id: str) -> LedgerEntry | None:
        row = self.conn.execute(
            "SELECT * FROM ledger_entry WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._parse_row(row)

    def outstanding_orders(self) -> list[dict[str, Any]]:
        """Return open or unknown broker orders that may need reconciliation."""
        rows = self.conn.execute(
            """
            SELECT
                le.idempotency_key,
                le.order_intent_id,
                le.broker_order_id,
                le.broker_status,
                le.instrument_id,
                le.cycle_id,
                le.sequence,
                le.payload
            FROM ledger_entry le
            INNER JOIN (
                SELECT idempotency_key, MAX(sequence) AS sequence
                FROM ledger_entry
                WHERE idempotency_key IS NOT NULL
                GROUP BY idempotency_key
            ) latest ON (
                le.idempotency_key = latest.idempotency_key
                AND le.sequence = latest.sequence
            )
            WHERE le.broker_status IN ('accepted', 'partially_filled', 'unknown')
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def _parse_row(self, row: sqlite3.Row) -> LedgerEntry:
        decision_time = row["decision_time"]
        created_at = row["created_at"]
        return LedgerEntry(
            entry_id=row["entry_id"],
            cycle_id=row["cycle_id"],
            sequence=row["sequence"],
            entry_type=LedgerEntryType(row["entry_type"]),
            decision_time=datetime.fromisoformat(decision_time) if decision_time else None,
            source_document_ids=json.loads(row["source_document_ids"]),
            dataset_version=row["dataset_version"],
            config_version=row["config_version"],
            strategy_id=row["strategy_id"],
            strategy_version=row["strategy_version"],
            proposal_hash=row["proposal_hash"],
            approval_id=row["approval_id"],
            request_hash=row["request_hash"],
            order_intent_id=row["order_intent_id"],
            idempotency_key=row["idempotency_key"],
            broker_order_id=row["broker_order_id"],
            broker_status=row["broker_status"],
            instrument_id=row["instrument_id"],
            fill_quantity=row["fill_quantity"],
            fill_price=row["fill_price"],
            fill_fees=row["fill_fees"],
            portfolio_snapshot=row["portfolio_snapshot"],
            error_code=row["error_code"],
            recovery_action=row["recovery_action"],
            payload=row["payload"],
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
        )


class HealthRepository:
    """Singleton health state persistence."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._ensure_row_factory()

    def _ensure_row_factory(self) -> None:
        if self.conn.row_factory is None:
            self.conn.row_factory = sqlite3.Row

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def get(self) -> HealthState | None:
        row = self.conn.execute("SELECT * FROM health_state WHERE id = 1").fetchone()
        if row is None:
            return None
        return HealthState(
            state=HealthStatus(row["state"]),
            reason=row["reason"] or "",
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def set(self, state: HealthState) -> None:
        now = self._now()
        updated_at = state.updated_at or now
        self.conn.execute(
            """
            INSERT INTO health_state (id, state, reason, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (state.state.value, state.reason, updated_at.isoformat()),
        )

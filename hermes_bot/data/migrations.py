"""Versioned SQLite schema migrations for the canonical catalog (T04).

All schema changes are applied transactionally through ``MigrationRunner``.
Migrations are identified by a monotonically increasing integer version.
The ``_migrations`` table records which versions have been applied.

Decimal representation
----------------------
Prices, quantities and weights are stored as REAL values in named currency
units.  Returns and weights are NAV fractions (``0.10`` = 10%).  Timestamps
are timezone-aware UTC ISO-8601 strings at storage boundaries.

Rollback limits
---------------
A failed migration is rolled back as a single transaction, but SQLite does
not support rolling back already-committed DDL from earlier migrations.
Downgrade/migration-undo scripts are not implemented; restoring an older
schema requires restoring a database backup taken before the migration.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

MigrationFunc = Callable[[sqlite3.Connection], None]


class MigrationError(Exception):
    """Raised when a migration cannot be applied."""


@dataclass(frozen=True)
class Migration:
    """One schema migration."""

    version: int
    name: str
    apply: MigrationFunc


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def _ts_now() -> str:
    return datetime.now(UTC).isoformat()


def _create_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _create_legacy_market_data(conn: sqlite3.Connection) -> None:
    """Baseline v0: legacy OHLCV table, preserved for compatibility."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timestamp)
        )
        """
    )


def _create_catalog_schema(conn: sqlite3.Connection) -> None:
    """v1: canonical entity, instrument, document, event and relation tables."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entity (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instrument (
            instrument_id TEXT PRIMARY KEY,
            issuer_entity_id TEXT REFERENCES entity(entity_id),
            asset_class TEXT NOT NULL,
            exchange_mic TEXT,
            currency TEXT NOT NULL,
            instrument_type TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alias (
            alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT NOT NULL,
            namespace TEXT NOT NULL,
            entity_id TEXT NOT NULL REFERENCES entity(entity_id),
            valid_from TEXT,
            valid_to TEXT,
            resolution_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (alias, namespace, entity_id, valid_from)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_record (
            source_id TEXT PRIMARY KEY,
            source_uri TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            license_id TEXT,
            raw_object_path TEXT,
            published_time TEXT,
            ingested_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document (
            document_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES source_record(source_id),
            source_uri TEXT NOT NULL,
            raw_sha256 TEXT NOT NULL,
            language TEXT,
            document_type TEXT NOT NULL,
            published_time TEXT,
            ingested_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            license_id TEXT,
            raw_object_path TEXT,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS event (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            normalized_claim_text TEXT NOT NULL,
            stance_direction TEXT,
            event_time TEXT,
            published_time TEXT,
            available_at TEXT NOT NULL,
            epistemic_type TEXT NOT NULL,
            novelty REAL,
            extractor_version TEXT NOT NULL,
            source_document_ids TEXT NOT NULL DEFAULT '[]',
            source_chunk_ids TEXT NOT NULL DEFAULT '[]',
            affected_entity_ids TEXT NOT NULL DEFAULT '[]',
            uncertainty_fields TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_edge (
            edge_id TEXT PRIMARY KEY,
            from_entity_id TEXT NOT NULL REFERENCES entity(entity_id),
            relation_type TEXT NOT NULL,
            to_entity_id TEXT NOT NULL REFERENCES entity(entity_id),
            valid_from TEXT,
            valid_to TEXT,
            observed_at TEXT,
            available_at TEXT NOT NULL,
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            source_ids TEXT NOT NULL DEFAULT '[]',
            strength_value REAL,
            strength_semantics TEXT NOT NULL,
            confidence REAL,
            method_version TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_record (
            decision_id TEXT PRIMARY KEY,
            instrument_id TEXT NOT NULL REFERENCES instrument(instrument_id),
            target_weight REAL NOT NULL,
            decision_time TEXT NOT NULL,
            strategy_id TEXT,
            strategy_version TEXT,
            input_record_ids TEXT NOT NULL DEFAULT '[]',
            reason_codes TEXT NOT NULL DEFAULT '[]',
            approval_id TEXT,
            created_at TEXT NOT NULL
        )
        """
    )


def _create_ledger_schema(conn: sqlite3.Connection) -> None:
    """v2: append-only decision ledger, cycle header and health state tables."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger_cycle (
            cycle_id TEXT PRIMARY KEY,
            decision_time TEXT NOT NULL,
            strategy_id TEXT,
            strategy_version TEXT,
            proposal_hash TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger_entry (
            entry_id TEXT PRIMARY KEY,
            cycle_id TEXT NOT NULL REFERENCES ledger_cycle(cycle_id),
            sequence INTEGER NOT NULL,
            entry_type TEXT NOT NULL,
            decision_time TEXT,
            source_document_ids TEXT NOT NULL DEFAULT '[]',
            dataset_version TEXT,
            config_version TEXT,
            strategy_id TEXT,
            strategy_version TEXT,
            proposal_hash TEXT,
            approval_id TEXT,
            request_hash TEXT,
            order_intent_id TEXT,
            idempotency_key TEXT,
            broker_order_id TEXT,
            broker_status TEXT,
            instrument_id TEXT,
            fill_quantity REAL,
            fill_price REAL,
            fill_fees REAL,
            portfolio_snapshot TEXT,
            error_code TEXT,
            recovery_action TEXT,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (cycle_id, sequence)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ledger_entry_cycle_seq
        ON ledger_entry(cycle_id, sequence)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ledger_entry_approval
        ON ledger_entry(approval_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ledger_entry_intent
        ON ledger_entry(order_intent_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ledger_entry_idempotency
        ON ledger_entry(idempotency_key)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS health_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            state TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        )
        """
    )


# ---------------------------------------------------------------------------
# Public migration registry
# ---------------------------------------------------------------------------

MIGRATIONS: list[Migration] = [
    Migration(version=0, name="legacy_market_data", apply=_create_legacy_market_data),
    Migration(version=1, name="canonical_catalog", apply=_create_catalog_schema),
    Migration(version=2, name="decision_ledger", apply=_create_ledger_schema),
]


class MigrationRunner:
    """Apply pending migrations transactionally."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        migrations: list[Migration] | None = None,
    ) -> None:
        self.conn = conn
        self.migrations = sorted(migrations or MIGRATIONS, key=lambda m: m.version)

    def current_version(self) -> int:
        """Return the highest applied migration version, or -1 if none."""
        _create_migrations_table(self.conn)
        row = self.conn.execute("SELECT MAX(version) FROM _migrations").fetchone()
        return row[0] if row[0] is not None else -1

    def migrate(self) -> list[int]:
        """Apply all pending migrations in one transaction.

        Returns the list of versions that were applied.  Raises
        ``MigrationError`` if any migration fails; the transaction is rolled
        back and no version is recorded.
        """
        self.conn.execute("PRAGMA foreign_keys = ON")

        explicit_begin = False
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
            explicit_begin = True

        # Legacy detection: market_data exists but no migration history.
        has_market_data = (
            self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_data'"
            ).fetchone()
            is not None
        )
        _create_migrations_table(self.conn)
        current = self.current_version()
        if current == -1 and has_market_data:
            self.conn.execute(
                "INSERT INTO _migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (0, "legacy_baseline", _ts_now()),
            )
            current = 0

        pending = [m for m in self.migrations if m.version > current]
        if not pending:
            if explicit_begin:
                self.conn.execute("COMMIT")
            return []

        current_migration: Migration | None = None
        try:
            for migration in pending:
                current_migration = migration
                migration.apply(self.conn)
                self.conn.execute(
                    "INSERT INTO _migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.name, _ts_now()),
                )
            if explicit_begin:
                self.conn.execute("COMMIT")
        except Exception as exc:
            if explicit_begin:
                self.conn.execute("ROLLBACK")
            name = current_migration.name if current_migration else "unknown"
            version = current_migration.version if current_migration else -1
            raise MigrationError(
                f"Migration failed at version {version} ({name})"
            ) from exc
        return [m.version for m in pending]


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Convenience entry point using the default migration registry."""
    return MigrationRunner(conn).migrate()

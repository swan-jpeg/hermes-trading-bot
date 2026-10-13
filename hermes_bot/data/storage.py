"""SQLite storage for dev (no external DB needed to start).

The store now keeps the legacy ``market_data`` table and applies versioned
migrations to add the canonical catalog tables defined in the execution
package §4.  All schema creation goes through ``hermes_bot.data.migrations``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_bot.data.migrations import MigrationRunner, migrate


class SQLiteStore:
    """Local SQLite store with versioned migrations.

    Parameters
    ----------
    path:
        Database file path.  The parent directory is created on init.
    auto_migrate:
        If True (default), pending migrations are applied on init.
    """

    def __init__(self, path: str | Path = "var/tradingbot.db", auto_migrate: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if auto_migrate:
            self.migrate()

    @property
    def conn(self) -> sqlite3.Connection:
        """Underlying SQLite connection."""
        return self._conn

    def migrate(self) -> list[int]:
        """Apply pending migrations transactionally."""
        return migrate(self._conn)

    def migration_runner(self) -> MigrationRunner:
        """Return a runner for inspection/testing."""
        return MigrationRunner(self._conn)

    def current_schema_version(self) -> int:
        """Return the currently applied migration version."""
        return self.migration_runner().current_version()

    def insert_many(self, records: list[dict]) -> int:
        """Insert or replace OHLCV rows in the legacy ``market_data`` table."""
        rows = []
        for r in records:
            ts = r["timestamp"]
            ts = ts.isoformat() if hasattr(ts, "isoformat") else ts
            rows.append((r["symbol"], ts, r["open"], r["high"], r["low"], r["close"], r["volume"]))
        self._conn.executemany(
            "INSERT OR REPLACE INTO market_data VALUES (?,?,?,?,?,?,?)", rows
        )
        self._conn.commit()
        return len(rows)

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

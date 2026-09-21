"""SQLite-opslag voor dev (geen externe DB nodig om te starten)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteStore:
    """Minimale opslag: tabel `market_data` voor OHLCV-records."""

    def __init__(self, path: str | Path = "var/tradingbot.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT, timestamp TEXT, open REAL, high REAL,
                low REAL, close REAL, volume REAL,
                PRIMARY KEY (symbol, timestamp)
            )
            """
        )
        self._conn.commit()

    def insert_many(self, records: list[dict]) -> int:
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
        self._conn.close()

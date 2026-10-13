"""Marktdata-collector: koersen/volumes via yfinance (gratis)."""
from __future__ import annotations

from hermes_bot.data import BaseCollector, CollectorConfig, DataMode
from hermes_bot.data.adapters import AdapterRequest, MarketDataAdapter


class MarketDataCollector(BaseCollector):
    """Fetches OHLCV + fundamentals for the universe (yfinance to start).

    ``OFFLINE_MOCK`` is the default and returns deterministic synthetic bars
    without any network call.  The legacy yfinance path is isolated behind
    ``mode=DataMode.NETWORK``.
    """

    name = "market"

    def __init__(
        self,
        cfg: CollectorConfig | None = None,
        symbols: list[str] | None = None,
        mode: DataMode = DataMode.OFFLINE_MOCK,
    ) -> None:
        super().__init__(cfg)
        self.symbols = symbols or ["AAPL", "MSFT", "SPY"]
        self.mode = mode

    def collect(self) -> list[dict]:
        """Fetch daily OHLCV.

        Records: {symbol, timestamp, open, high, low, close, volume}.
        ``timestamp`` is kept as the session date (ISO-8601 date string) to
        avoid faking a precise UTC timestamp for an end-of-day bar.
        """
        adapter = MarketDataAdapter(mode=self.mode, symbols=self.symbols)
        request = AdapterRequest(
            mode=self.mode,
            source_config={"symbols": self.symbols},
            max_records=100,
        )
        records = adapter.fetch(request)
        def _fmt(ts):
            return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        return [
            {
                "symbol": rec.payload["symbol"],
                "timestamp": _fmt(rec.payload["session_date"]),
                "open": float(rec.payload["open"]),
                "high": float(rec.payload["high"]),
                "low": float(rec.payload["low"]),
                "close": float(rec.payload["close"]),
                "volume": float(rec.payload["volume"]),
            }
            for rec in records
        ]

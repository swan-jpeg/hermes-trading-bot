"""Marktdata-collector: koersen/volumes via yfinance (gratis)."""
from __future__ import annotations

from datetime import UTC

from hermes_bot.data import BaseCollector, CollectorConfig


class MarketDataCollector(BaseCollector):
    """Fetches OHLCV + fundamentals for the universe (yfinance to start)."""

    name = "market"

    def __init__(
        self,
        cfg: CollectorConfig | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.symbols = symbols or ["AAPL", "MSFT", "SPY"]

    def collect(self) -> list[dict]:
        """Fetch daily OHLCV via yfinance.

        Records: {symbol, timestamp, open, high, low, close, volume}.
        """
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("yfinance niet geïnstalleerd: `uv sync --extra data`") from e

        records: list[dict] = []
        for sym in self.symbols:
            try:
                df = yf.download(sym, period="1mo", interval="1d", progress=False, auto_adjust=True)
                if df is None or df.empty:
                    continue
                df = df.reset_index()
                for _, row in df.iterrows():
                    ts = row.get("Date") or row.get("Datetime")
                    records.append(
                        {
                            "symbol": sym,
                            "timestamp": ts.to_pydatetime().replace(tzinfo=UTC)
                            if hasattr(ts, "to_pydatetime")
                            else ts,
                            "open": float(row["Open"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "volume": float(row["Volume"]),
                        }
                    )
            except Exception as e:  # pragma: no cover
                print(f"[market] {sym} mislukt: {e}")
        return records

"""Marktdata-collector: koersen/volumes/fundamentals."""
from __future__ import annotations

from hermes_bot.data import BaseCollector, CollectorConfig


class MarketDataCollector(BaseCollector):
    """Haalt OHLCV + fundamentals op voor de universe (yfinance start)."""

    name = "market"

    def __init__(
        self,
        cfg: CollectorConfig | None = None,
        symbols: list[str] | None = None,
    ) -> None:
        super().__init__(cfg)
        self.symbols = symbols or ["AAPL", "MSFT", "SPY"]

    def collect(self) -> list[dict]:
        """TODO: yfinance/polygon ophalen en als gestructureerde records teruggeven.
        Records: {symbol, timestamp, open, high, low, close, volume}.
        """
        raise NotImplementedError("Implementeer met yfinance of polygon (optionele dep).")

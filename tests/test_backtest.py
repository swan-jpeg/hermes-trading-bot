"""Tests voor backtest-framework (T5)."""
from __future__ import annotations

import pandas as pd

from hermes_bot.backtest import Backtester, BuyAndHoldStrategy, VolTargetStrategy
from hermes_bot.backtest.metrics import BacktestResult


def _prices(close: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=len(close), freq="B")
    return pd.DataFrame(
        {"open": close, "high": [c * 1.01 for c in close],
         "low": [c * 0.99 for c in close], "close": close,
         "volume": [1000.0] * len(close)},
        index=dates,
    )


def test_backtester_basic() -> None:
    """Test dat de backtester werkt zonder fouten."""
    prices = _prices([100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 110.0])
    result = Backtester().run(prices, BuyAndHoldStrategy())
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == 10
    assert result.initial_capital == 100_000.0


def test_vol_target_never_fails() -> None:
    """Test dat de vol-target strategie draait en resultaat levert."""
    prices = _prices([100.0, 102.0, 99.0, 103.0, 105.0, 108.0, 106.0, 110.0, 112.0, 115.0])
    result = Backtester().run(prices, VolTargetStrategy({"entity": "asset"}))
    assert isinstance(result, BacktestResult)
    assert result.max_drawdown <= 0.0  # drawdown is nooit positief

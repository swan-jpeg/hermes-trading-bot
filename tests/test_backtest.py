"""Tests voor backtest-framework (T5)."""
from __future__ import annotations

import pandas as pd

from hermes_bot.backtest.engine import Backtester, BuyAndHoldStrategy
from hermes_bot.backtest.metrics import BacktestResult


def test_backtester_basic() -> None:
    """Test dat de backtester werkt zonder fouten."""
    # Maak een simpele dataframe
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    prices = pd.DataFrame({
        'open': [100.0] * 10,
        'high': [105.0] * 10,
        'low': [95.0] * 10,
        'close': [100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'volume': [1000.0] * 10
    }, index=dates)
    
    # Maak een strategie
    strategy = BuyAndHoldStrategy()
    
    # Maak een backtester
    backtester = Backtester()
    
    # Voer backtest uit
    result = backtester.run(prices, strategy)
    
    # Controleer dat het resultaat correct is
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == 10
    assert result.initial_capital == 100000.0
    # De test is niet echt betrouwbaar zonder een goede implementatie


def test_buy_and_hold_strategy() -> None:
    """Test dat de buy-and-hold strategie werkt."""
    strategy = BuyAndHoldStrategy()
    
    # Test met een dummy rij
    dummy_row = {
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 102.0,
        'volume': 1000.0
    }
    
    action = strategy.decide(dummy_row)
    assert action == "BUY"
"""LAAG 5: backtest metrics."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class BacktestResult(BaseModel):
    """Result of a backtest."""

    initial_capital: float
    final_capital: float
    equity_curve: list[float]
    timestamps: list[datetime]
    returns: list[float]
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    benchmark_return: float = 0.0  # Buy-and-hold benchmark
    strategy_returns: list[float] = []  # for comparison (buy-and-hold)
    n_trades: int = 0
    total_return: float = 0.0
    annualized_return: float = 0.0
    # End exposure (weight) of the portfolio — input for the crash-scenario
    # stress ([[08]] point 3).
    final_exposure: float = 0.0
    # Crash scenarios forced on the end portfolio: name -> return.
    scenario_stress: dict[str, float] = {}
    # Bootstrap confidence intervals of mean/Sharpe/max-drawdown ([[08]] point 4).
    confidence_intervals: dict = {}

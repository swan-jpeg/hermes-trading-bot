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
    strategy_returns: list[float] = []  # Voor vergelijking
    n_trades: int = 0
    total_return: float = 0.0
    annualized_return: float = 0.0

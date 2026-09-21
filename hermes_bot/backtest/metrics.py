"""LAAG 5: backtest metrics."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class BacktestResult(BaseModel):
    """Resultaat van een backtest."""

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

    @property
    def total_return(self) -> float:
        """Totale rendement van de strategie."""
        if self.initial_capital == 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def annualized_return(self) -> float:
        """Jaarlijkse gerenormaliseerde return."""
        if not self.timestamps:
            return 0.0
        # Simpele berekening
        days = (self.timestamps[-1] - self.timestamps[0]).days if len(self.timestamps) > 1 else 1
        return (1 + self.total_return) ** (365.0 / days) - 1 if days > 0 else 0.0

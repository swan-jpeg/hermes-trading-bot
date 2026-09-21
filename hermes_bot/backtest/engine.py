"""LAAG 5: backtest-engine voor strategieën (buy-and-hold benchmark)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from hermes_bot.backtest.metrics import BacktestResult


class Backtester:
    """Event-driven backtester voor trading-strategieën."""

    def __init__(self) -> None:
        pass

    def run(self, prices: pd.DataFrame, strategy) -> BacktestResult:
        """Voer een backtest uit op de gegeven prijzen met de opgegeven strategie.

        Args:
            prices: DataFrame met kolommen ['open', 'high', 'low', 'close', 'volume']
            strategy: Strategie object dat een actie retourneert voor elke timestep

        Returns:
            BacktestResult met metrics en equity curve
        """
        # Initialisatie
        positions = {}  # {asset: aantal}
        cash = 100000.0  # Startkapitaal
        equity_curve = []
        timestamps = []

        # Voor elke tijdstap
        for idx, row in prices.iterrows():
            timestamp = idx
            timestamps.append(timestamp)
            
            # Voer strategie uit
            action = strategy.decide(row)  # Simpele strategie interface
            
            # Voer actie uit (voorbeeldimplementatie)
            if action == "BUY":
                # Koop alle beschikbare cash in de asset
                asset_price = row['close']
                if asset_price > 0:
                    quantity = cash / asset_price
                    positions['asset'] = quantity
                    cash = 0.0
            elif action == "SELL":
                # Verkoop alle posities
                if 'asset' in positions:
                    asset_price = row['close']
                    cash = positions['asset'] * asset_price
                    positions = {}
            
            # Bereken equity op dit moment
            total_value = cash
            if 'asset' in positions:
                asset_price = row['close']
                total_value += positions['asset'] * asset_price
                
            equity_curve.append(total_value)

        # Bereken metrics
        from hermes_bot.backtest.metrics import BacktestResult
        result = BacktestResult(
            initial_capital=100000.0,
            final_capital=equity_curve[-1] if equity_curve else 0.0,
            equity_curve=equity_curve,
            timestamps=timestamps,
            returns=[],
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
        )
        
        return result


class BuyAndHoldStrategy:
    """Simple buy-and-hold strategie als benchmark."""

    def decide(self, price_row) -> str:
        """Always buy at the first opportunity."""
        # Voor T5 is dit een dummy implementatie
        # In een echte implementatie zou dit een logica bevatten
        return "BUY"  # Altijd kopen bij eerste kans
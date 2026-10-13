"""Tests for the backtest framework (T5)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from hermes_bot.backtest import Backtester, BuyAndHoldStrategy, Strategy, VolTargetStrategy
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
    """Test that the backtester works without errors."""
    prices = _prices([100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 110.0])
    result = Backtester().run(prices, BuyAndHoldStrategy())
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == 10
    assert result.initial_capital == 100_000.0


def test_vol_target_never_fails() -> None:
    """Test that the vol-target strategy runs and produces a result."""
    prices = _prices([100.0, 102.0, 99.0, 103.0, 105.0, 108.0, 106.0, 110.0, 112.0, 115.0])
    result = Backtester().run(prices, VolTargetStrategy({"entity": "asset"}))
    assert isinstance(result, BacktestResult)
    assert result.max_drawdown <= 0.0  # drawdown is nooit positief


class AlwaysLongStrategy(Strategy):
    """Test strategy that always wants to be 100% invested."""

    name = "always_long"

    def decide(self, idx, row, portfolio):
        from hermes_bot.schemas import Action, ExitReason, RLRawDecision
        return RLRawDecision(
            entity="asset", action=Action.BUY, intent_to_alloc=1.0,
            zekerheid=1.0, timestamp=datetime.now(),
            rationale="test: always long", exit_reason=ExitReason.NONE,
        )


def _prices_with_open(closes, opens):
    dates = pd.date_range("2023-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {"open": opens,
         "high": [max(c, o) * 1.01 for c, o in zip(closes, opens, strict=True)],
         "low": [min(c, o) * 0.99 for c, o in zip(closes, opens, strict=True)],
         "close": closes,
         "volume": [1000.0] * len(closes)},
        index=dates,
    )


def test_next_bar_open_fill() -> None:
    """Decision at close t is filled at open t+1 when open data is available."""
    closes = [100.0, 110.0, 120.0]
    opens = [100.0, 105.0, 115.0]  # different from close
    prices = _prices_with_open(closes, opens)
    risk_cfg = {"risk": {"max_asset_weight": 1.0}}
    result = Backtester().run(prices, AlwaysLongStrategy(), risk_config=risk_cfg)
    # Day 0: decide to buy at close 100. Day 1: fill at open 105.
    # Final equity = 100k / 105 * 120 = 114285.71 gross.
    # No costs, so gross == net.
    expected_final = 100_000.0 / opens[1] * closes[-1]
    assert abs(result.final_capital - expected_final) < 1e-2
    assert result.total_costs == 0.0
    assert len(result.gross_returns) == len(result.net_returns)
    for g, n in zip(result.gross_returns, result.net_returns, strict=True):
        assert abs(g - n) < 1e-12


def test_known_three_bar_calculation() -> None:
    """Hand-check the full accounting over three bars with open data."""
    closes = [100.0, 110.0, 121.0]
    opens = [100.0, 108.0, 119.0]
    prices = _prices_with_open(closes, opens)
    risk_cfg = {"risk": {"max_asset_weight": 1.0}}
    result = Backtester().run(prices, AlwaysLongStrategy(), risk_config=risk_cfg)
    # Day 0: decide 100% long at close 100.
    # Day 1: fill at open 108, mark-to-market at close 110.
    #   qty = 100_000 / 108; equity = qty * 110 = 101_851.85
    # Day 2: hold, mark-to-market at close 121.
    #   equity = qty * 121 = 112_037.04
    expected_final = 100_000.0 / opens[1] * closes[-1]
    assert abs(result.final_capital - expected_final) < 1e-2
    assert result.total_costs == 0.0


def test_costs_reduce_return() -> None:
    """Commission + slippage reduce the net return relative to gross."""
    closes = [100.0, 110.0, 120.0]
    opens = [100.0, 105.0, 115.0]
    prices = _prices_with_open(closes, opens)
    risk_cfg = {"risk": {"max_asset_weight": 1.0}}
    cfg = {"commission_bps": 10.0, "slippage_bps": 5.0}
    result = Backtester(cfg).run(prices, AlwaysLongStrategy(), risk_config=risk_cfg)
    # Costs are positive, so net return must be strictly less than gross return.
    assert result.total_costs > 0.0
    assert sum(result.net_returns) < sum(result.gross_returns)
    # Expected costs: buy on day 1 plus the tiny rebalance on day 2.
    # Rebalances use the mark-to-market at the next-bar open, not at close.
    qty1 = 100_000.0 / opens[1]
    cost1 = 100_000.0 * 0.0015
    cash1 = -cost1
    total_value_at_open2 = cash1 + qty1 * opens[2]
    target_qty2 = total_value_at_open2 / opens[2]
    sell_notional = (qty1 - target_qty2) * opens[2]
    cost2 = sell_notional * 0.0015
    expected_cost = cost1 + cost2
    assert abs(result.total_costs - expected_cost) < 1e-2


def test_zero_cost_compatibility() -> None:
    """With zero commission and slippage, gross and net returns are equal."""
    prices = _prices([100.0, 102.0, 101.0, 103.0, 104.0])
    result = Backtester({"commission_bps": 0.0, "slippage_bps": 0.0}).run(
        prices, BuyAndHoldStrategy()
    )
    assert result.total_costs == 0.0
    assert len(result.gross_returns) == len(result.net_returns)
    for g, n in zip(result.gross_returns, result.net_returns, strict=True):
        assert abs(g - n) < 1e-12


def test_empty_prices() -> None:
    """An empty price series returns a valid but trivial result."""
    prices = _prices([])
    result = Backtester().run(prices, BuyAndHoldStrategy())
    assert result.final_capital == 100_000.0
    assert result.equity_curve == []
    assert result.total_costs == 0.0


def test_one_row_prices() -> None:
    """A single price row records zero return and no trades."""
    prices = _prices([100.0])
    result = Backtester().run(prices, BuyAndHoldStrategy())
    assert len(result.equity_curve) == 1
    assert result.final_capital == 100_000.0
    assert result.total_costs == 0.0
    assert result.n_trades == 0


def test_invalid_price_skipped() -> None:
    """Rows with non-positive close prices are skipped."""
    prices = _prices([100.0, 0.0, 102.0, -5.0, 104.0])
    result = Backtester().run(prices, BuyAndHoldStrategy())
    # Only the positive close rows contribute to the equity curve.
    assert len(result.equity_curve) == 3
    assert all(e > 0 for e in result.equity_curve)


def test_synthetic_crash_drawdown() -> None:
    """A synthetic crash produces a non-zero drawdown."""
    from hermes_bot.backtest import generate_synthetic_prices

    prices = generate_synthetic_prices(n=400, seed=7)
    result = Backtester().run(prices, BuyAndHoldStrategy())
    assert result.max_drawdown < -0.05  # crash segment produces material drawdown


class SingleBuyStrategy(Strategy):
    """Test strategy that buys once and then holds the same allocation."""

    name = "single_buy"

    def decide(self, idx, row, portfolio):
        from hermes_bot.schemas import Action, ExitReason, RLRawDecision
        if idx == 0:
            alloc = 1.0
        else:
            total = portfolio.cash
            pos = portfolio.positions.get("asset")
            price = float(row["close"])
            if pos:
                total += pos.qty * price
            alloc = (pos.qty * price / total) if pos and total > 0 else 0.0
            alloc = min(1.0, max(0.0, alloc))
        return RLRawDecision(
            entity="asset", action=Action.BUY, intent_to_alloc=alloc,
            zekerheid=1.0, timestamp=datetime.now(),
            rationale="test: single buy and hold", exit_reason=ExitReason.NONE,
        )


def test_net_returns_reconcile_to_equity_path() -> None:
    """With costs > 0, net_returns is the exact mark-to-market equity path."""
    closes = [100.0] * 8
    opens = [100.0] * 8
    prices = _prices_with_open(closes, opens)
    risk_cfg = {"risk": {"max_asset_weight": 1.0}}
    cfg = {"commission_bps": 10.0, "slippage_bps": 5.0}
    result = Backtester(cfg).run(prices, SingleBuyStrategy(), risk_config=risk_cfg)
    equity = result.equity_curve
    assert len(equity) == len(result.net_returns)
    assert result.net_returns[0] == 0.0
    for i in range(1, len(equity)):
        expected = equity[i] / equity[i - 1] - 1.0
        assert abs(result.net_returns[i] - expected) < 1e-12


def test_cost_impact_reconciles() -> None:
    """Flat prices + a single buy: gross-net spread matches total costs."""
    closes = [100.0] * 8
    opens = [100.0] * 8
    prices = _prices_with_open(closes, opens)
    risk_cfg = {"risk": {"max_asset_weight": 1.0}}
    cfg = {"commission_bps": 10.0, "slippage_bps": 5.0}
    result = Backtester(cfg).run(prices, SingleBuyStrategy(), risk_config=risk_cfg)
    assert result.total_costs > 0.0
    cost_impact = sum(result.gross_returns) - sum(result.net_returns)
    expected = result.total_costs / result.initial_capital
    assert abs(cost_impact - expected) < 1e-6
    # Final equity vs the zero-cost run is reduced by roughly total_costs.
    result_no_cost = Backtester({"commission_bps": 0.0, "slippage_bps": 0.0}).run(
        prices, SingleBuyStrategy(), risk_config=risk_cfg
    )
    assert abs(
        result.final_capital - (result_no_cost.final_capital - result.total_costs)
    ) < 1e-2

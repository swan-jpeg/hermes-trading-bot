"""LAYER 6: backtest engine — the mandatory gate before money. [[13]]

A real event-driven backtest that uses the RISK ENGINE (RiskEngine) for
position sizing, drawdown guard and crash resistance, and compares against a
buy-and-hold benchmark. Goal: good returns AND resilience against market crashes.

Strategie-interface:
    def decide(self, ts_idx, row, portfolio) -> RLRawDecision
Elke strategie retourneert een ruw besluit; de engine laat dat altijd
door een RiskEngine lopen (RL voorstelt, Risk keurt goed) — net als live.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from hermes_bot.portfolio import PortfolioState
from hermes_bot.rl import RulePolicy
from hermes_bot.schemas import ExitReason, MonteCarloResult, Position, RLRawDecision
from hermes_bot.simulation import MonteCarloEngine

from .metrics import BacktestResult


@dataclass
class _Trade:
    entry_price: float
    exit_price: float
    qty: float


def _sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _sortino(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    downside = np.minimum(r, 0.0)
    ds = downside.std()
    if r.size == 0 or ds == 0:
        return 0.0
    return float(r.mean() / ds * np.sqrt(252))


class Strategy:
    """Base interface for a backtest strategy."""

    name = "base"

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}

    def decide(self, idx, row: pd.Series, portfolio: PortfolioState) -> RLRawDecision:
        raise NotImplementedError


class VolTargetStrategy(Strategy):
    """Crash-veilige strategie: volatiliteit-targeting + drawdown-guard.

    Gaat volledig in een asset bij lage vol, schaalt terug bij hoge vol
    (crashweerstand), neemt winst/verlies via trailing rules. Gebruikt de
    RulePolicy-exit-logica voor prijs-gedreven winst nemen.
    """

    name = "vol_target"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.ann_vol_target = self.cfg.get("vol_target", 0.125)
        self.lookback = self.cfg.get("vol_lookback", 20)
        self.entry_exit = self.cfg.get("entry_exit", 0.9)  # ratio om in/uit te gaan
        self.policy = RulePolicy()
        self.hist_prices: list[float] = []

    def _current_vol(self, price: float) -> float:
        self.hist_prices.append(price)
        prices = np.asarray(self.hist_prices[-self.lookback :], dtype=float)
        if prices.size < 2:
            return 0.0
        # Returns from prices (not the price differences themselves).
        rets = np.diff(prices) / prices[:-1]
        if rets.size == 0 or rets.std() == 0:
            return 0.0
        return float(rets.std() * np.sqrt(252))

    def decide(self, idx, row: pd.Series, portfolio: PortfolioState) -> RLRawDecision:
        entity = self.cfg.get("entity", "SPY")
        price = float(row["close"])
        vol = self._current_vol(price)
        self._last_vol = vol

        # Sentiment replaced by a price-driven signal: momentum.
        hist = np.asarray(self.hist_prices[-6:], dtype=float)
        rets = np.diff(hist) / hist[:-1] if hist.size > 1 else np.array([])
        momentum = float(rets.sum()) if rets.size else 0.0
        # Normalized momentum: z-score-like over the lookback.
        if len(self.hist_prices) >= 20:
            window = np.asarray(self.hist_prices[-20:])
            wrets = np.diff(window) / window[:-1]
            if wrets.std() > 0:
                momentum = float((wrets[-5:].mean()) / wrets.std())
        sentiment = float(np.clip(momentum, -1, 1))

        # Position sizing based on vol: the higher the vol, the smaller the position.
        if vol > 0:
            vol_scale = min(1.0, self.ann_vol_target / vol)
        else:
            vol_scale = 1.0
        alloc = 0.0
        if vol_scale > self.entry_exit:
            # Less conservative: 90% of the vol scale, capped at 0.95.
            alloc = min(0.95, 0.9 * vol_scale)  # vol-targeting position size
        action = "hold"
        exit_reason = ExitReason.NONE

        # Let RulePolicy apply price exits (take-profit/stop/trailing).
        if portfolio.positions.get(entity) is not None:
            pos = portfolio.positions[entity]
            peak = portfolio.peak_for(entity, price)
            r = pos.exit_reason_at(price, peak)
            if r != ExitReason.NONE:
                return RLRawDecision(
                    entity=entity, action="sell", intent_to_alloc=-1.0,
                    zekerheid=0.9, timestamp=datetime.now(),
                    rationale=f"exit: {r.value}", exit_reason=r,
                )
            # Positie open + vol te hoog -> risk-off verkleinen.
            if vol > 0 and self.ann_vol_target / vol < self.entry_exit:
                return RLRawDecision(
                    entity=entity, action="sell", intent_to_alloc=-1.0,
                    zekerheid=0.9, timestamp=datetime.now(),
                    rationale="risk-off: vol te hoog", exit_reason=ExitReason.RISK_OFF,
                )
            # Position open and vol ok -> hold (no re-buy).
            action = "hold"
            alloc = 0.0
        elif alloc > 0:
            action = "buy"

        return RLRawDecision(
            entity=entity, action=action, intent_to_alloc=alloc,
            zekerheid=0.9, timestamp=datetime.now(),
            rationale=f"vol_target vol={vol:.3f} mom={sentiment:.2f}",
            exit_reason=exit_reason,
        )


class BuyAndHoldStrategy(Strategy):
    """Benchmark: buy on day 1, hold."""

    name = "buy_and_hold"

    def decide(self, idx, row, portfolio: PortfolioState) -> RLRawDecision:
        entity = self.cfg.get("entity", "SPY")
        action = "buy" if idx == 0 else "hold"
        return RLRawDecision(
            entity=entity, action=action, intent_to_alloc=0.99,
            zekerheid=1.0, timestamp=datetime.now(),
            rationale="benchmark: koop en houd", exit_reason=ExitReason.NONE,
        )


class Backtester:
    """Event-driven backtester: strategy + RiskEngine over a price series."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        self.risk = None
        self.mc = None

    def _mc_from_returns(self, returns: np.ndarray) -> MonteCarloResult | None:
        try:
            return MonteCarloEngine(seed=42, n_paths=2000).simulate(returns, horizon=20)
        except Exception:
            return None

    def run(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        initial_capital: float = 100_000.0,
        risk_config: dict | None = None,
    ) -> BacktestResult:
        """Run the backtest. `prices`: DataFrame with 'close' (and optionally OHLCV)."""
        entity = prices.columns[0] if isinstance(prices, pd.Series) else "asset"
        if isinstance(prices, pd.Series):
            close = prices
            prices = close.to_frame(name=entity)
        if "close" not in prices.columns and len(prices.columns) == 1:
            prices = prices.rename(columns={prices.columns[0]: "close"})

        # ---------- RISK ENGINE (the hard layer) ----------
        from hermes_bot.risk import RiskEngine
        self.risk = RiskEngine(risk_config or self.cfg)

        cash = initial_capital
        portfolio = PortfolioState(cash=cash, regime="unknown")
        equity = []
        timestamps = []
        closed_trades: list[_Trade] = []
        daily_returns: list[float] = []
        qty = 0.0
        entry_price = 0.0
        peak = 0.0

        closes = prices["close"].to_numpy(dtype=float)
        idx_list = prices.index.tolist()

        for i, t in enumerate(idx_list):
            price = float(closes[i])
            row = prices.iloc[i]
            price = float(price)
            if price <= 0:
                continue
            peak = max(peak, price)
            portfolio.peaks = {"asset": peak}
            # Sync the real position back to the portfolio so the strategy
            # and RiskEngine see the open position (price awareness).
            if qty > 0:
                portfolio.positions = {
                    "asset": Position(
                        entity="asset", qty=qty, entry_price=entry_price,
                        entry_time=t, take_profit_pct=0.15, stop_loss_pct=0.08,
                        trailing_stop_pct=0.05,
                    )
                }
                portfolio.cash = cash
            else:
                portfolio.positions = {}
                portfolio.cash = cash

            # Strategy proposes; RiskEngine approves.
            decision = strategy.decide(i, row, portfolio)

            # Monte Carlo for crash resilience (periodic, not every day).
            mc = None
            if len(daily_returns) >= 30 and i % 20 == 0:
                mc = self._mc_from_returns(np.asarray(daily_returns[-60:]))

            approval = self.risk.approve(decision, portfolio, mc)

            # ---- Execute goedgekeurde allocatie (target-volgend, fail-closed) ----
            total_value = cash + qty * price
            target_weight = (
                approval.target_alloc.get(decision.entity, 0.0)
                if approval.approved else 0.0
            )
            target_qty = (
                target_weight * total_value / price
                if price > 0 and total_value > 0 else 0.0
            )
            target_qty = max(0.0, min(target_qty, total_value / price))
            if target_qty > qty + 1e-9:  # koop meer
                cash -= (target_qty - qty) * price
                if entry_price <= 0:
                    entry_price = price
                qty = target_qty
                peak = max(peak, price)
            elif target_qty < qty - 1e-9 and qty > 0:  # verkoop (deels)
                proceeds = (qty - target_qty) * price
                cash += proceeds
                if target_qty <= 1e-9:
                    closed_trades.append(_Trade(entry_price, price, qty))
                    qty = 0.0
                    entry_price = 0.0
                else:
                    qty = target_qty

            total_value = cash + qty * price
            if equity:
                daily_returns.append(total_value / equity[-1] - 1.0)
            else:
                daily_returns.append(0.0)
            equity.append(total_value)
            timestamps.append(t)

        # ---- Benchmark: buy-and-hold ----
        bench_close = closes[0]
        bench_qty = initial_capital / bench_close if bench_close > 0 else 0.0
        bench_final = closes[-1] * bench_qty if closes.size else initial_capital
        benchmark_return = (bench_final - initial_capital) / initial_capital

        # ---- Metrics ----
        rets = np.asarray(daily_returns[1:], dtype=float)
        sharpe = _sharpe(rets)
        sortino = _sortino(rets)
        eq = np.asarray(equity, dtype=float)
        running_peak = np.maximum.accumulate(eq)
        dd = eq / running_peak - 1
        max_dd = float(dd.min()) if dd.size else 0.0
        total_return = (equity[-1] - initial_capital) / initial_capital if equity else 0.0
        wins = sum(1 for tr in closed_trades if tr.exit_price > tr.entry_price)
        win_rate = wins / len(closed_trades) if closed_trades else 0.0

        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=float(equity[-1]) if equity else initial_capital,
            equity_curve=[float(x) for x in equity],
            timestamps=timestamps,
            returns=[float(x) for x in daily_returns],
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            benchmark_return=benchmark_return,
            strategy_returns=[float(x) for x in daily_returns],
            # metadata
            n_trades=len(closed_trades),
            total_return=total_return,
            annualized_return=(
                (1 + total_return) ** (252.0 / max(1, len(equity))) - 1 if equity else 0.0
            ),
        )


def load_prices(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch prices via yfinance (free) and return a 'close' DataFrame."""
    import yfinance as yf

    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"Geen data voor {symbol}")
    # yfinance can return a DataFrame with MultiIndex columns; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Close": "close", "Open": "open", "High": "high",
                            "Low": "low", "Volume": "volume"})
    out = df[["open", "high", "low", "close", "volume"]]
    # Old/new yfinance returns 'close' as a Series or 1-column DF; force 1D.
    if isinstance(out["close"], pd.DataFrame):
        out["close"] = out["close"].iloc[:, 0]
    return out


def generate_synthetic_prices(
    n: int = 500, seed: int = 7, drift: float = 0.0004, vol: float = 0.015,
) -> pd.DataFrame:
    """Create realistic simulated prices (including a crash) — for demo/tests."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    # Add a crash (~day 350) to test crash resilience.
    if n > 360:
        rets[350:365] -= 0.05
    prices = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": prices, "high": prices * 1.01,
                         "low": prices * 0.99, "close": prices,
                         "volume": 1e6}, index=idx)

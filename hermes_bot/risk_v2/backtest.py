"""Backtest harness for Risk Engine v2 — full per-day logging.

Draait de v2-engine over een prijsreeks met een neutrale strategie (constant
long) zodat alle exposure-variatie van de v2-engine komt. Logt elke dag de
volledige multiplier-breakdown + metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2 import RiskEngineV2
from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2
from hermes_bot.schemas import ExitReason, RLRawDecision

OUT = Path(__file__).resolve().parent.parent.parent / "var" / "forensic_v2"


class ConstantLongStrategy:
    """Proposes 100% long every day (certainty 1.0). The v2 engine decides."""

    name = "constant_long"

    def decide(self, idx, row, portfolio: PortfolioState) -> RLRawDecision:
        return RLRawDecision(
            entity="asset", action="buy", intent_to_alloc=1.0, zekerheid=1.0,
            timestamp=datetime.now(), rationale="constant long",
            exit_reason=ExitReason.NONE,
        )


@dataclass
class DailyLog:
    date: str
    price: float
    return_pct: float
    portfolio_value: float
    drawdown: float
    volatility: float
    correlation: float
    var_95: float
    crash_prob: float
    regime: str
    certainty: float
    base_exposure: float
    vol_multiplier: float
    drawdown_multiplier: float
    correlation_multiplier: float
    tail_multiplier: float
    regime_multiplier: float
    certainty_multiplier: float
    final_exposure: float
    reason: str


@dataclass
class V2Result:
    name: str
    metrics: dict
    equity: list[float]
    timestamps: list[object]
    exposure: list[float]
    log: list[DailyLog] = field(default_factory=list)


def _metrics(equity, returns, n_trades, turnover, costs, exposure_series, cash_time):
    eq = np.asarray(equity, dtype=float)
    rets = np.asarray(returns, dtype=float)
    total = (eq[-1] - eq[0]) / eq[0] if eq.size else 0.0
    years = max(1.0, len(eq) / 252.0)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    max_dd = float(dd.min()) if dd.size else 0.0
    vol = float(rets.std() * np.sqrt(252)) if rets.size and rets.std() else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.size and rets.std() else 0.0
    downside = np.minimum(rets, 0.0)
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if downside.std() else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    avg_exp = float(np.mean(exposure_series)) if exposure_series else 0.0
    med_exp = float(np.median(exposure_series)) if exposure_series else 0.0
    min_exp = float(np.min(exposure_series)) if exposure_series else 0.0
    max_exp = float(np.max(exposure_series)) if exposure_series else 0.0
    return {
        "total_return": round(total, 4), "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4), "volatility": round(vol, 4),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "calmar": round(calmar, 3), "n_trades": n_trades,
        "turnover": round(turnover, 4), "transaction_costs": round(costs, 2),
        "avg_exposure": round(avg_exp, 4), "median_exposure": round(med_exp, 4),
        "min_exposure": round(min_exp, 4), "max_exposure": round(max_exp, 4),
        "cash_time_pct": round(cash_time, 4),
    }


def run_v2(
    prices: pd.DataFrame,
    risk_config: dict,
    name: str = "scenario",
    mc_every: int = 20,
    seed: int = 42,
) -> V2Result:
    """Run the v2 engine over a price series with full logging."""
    engine = RiskEngineV2(risk_config)
    strategy = ConstantLongStrategy()
    mc_engine = MonteCarloEngineV2(seed=seed, n_paths=1000)

    cash = 100_000.0
    portfolio = PortfolioState(cash=cash, regime="unknown")
    qty = 0.0
    equity: list[float] = []
    timestamps: list[object] = []
    exposure_series: list[float] = []
    daily_returns: list[float] = []
    log: list[DailyLog] = []
    n_trades = 0
    turnover = 0.0
    costs = 0.0
    cash_days = 0

    closes = prices["close"].to_numpy(dtype=float)
    idx_list = prices.index.tolist()

    for i, t in enumerate(idx_list):
        price = float(closes[i])
        if price <= 0:
            continue
        portfolio.peaks = {"asset": price}

        decision = strategy.decide(i, prices.iloc[i], portfolio)

        # Monte Carlo periodiek.
        mc = None
        if len(daily_returns) >= 30 and i % mc_every == 0:
            try:
                mc = mc_engine.simulate(np.asarray(daily_returns[-60:]), horizon=20)
            except Exception:
                mc = None

        # Current equity before the trade (for drawdown).
        current_equity = cash + qty * price
        final_exp, breakdown = engine.approve(decision, portfolio, current_equity, mc)

        # Execute (target-volgend).
        total_value = cash + qty * price
        target_qty = final_exp * total_value / price if price > 0 else 0.0
        target_qty = max(0.0, min(target_qty, total_value / price))
        if target_qty > qty + 1e-9:
            buy_qty = target_qty - qty
            cash -= buy_qty * price
            costs += buy_qty * price * 0.001
            qty = target_qty
            turnover += buy_qty * price / total_value
        elif target_qty < qty - 1e-9 and qty > 0:
            sell_qty = qty - target_qty
            cash += sell_qty * price
            costs += sell_qty * price * 0.001
            if target_qty <= 1e-9:
                n_trades += 1
            qty = target_qty
            turnover += sell_qty * price / total_value

        total_value = cash + qty * price
        if equity:
            ret = total_value / equity[-1] - 1.0
            daily_returns.append(ret)
            engine.record_return(ret)
        else:
            daily_returns.append(0.0)
        equity.append(total_value)
        timestamps.append(t)
        exposure = qty * price / total_value if total_value > 0 else 0.0
        exposure_series.append(exposure)
        if qty <= 1e-9:
            cash_days += 1

        log.append(DailyLog(
            date=str(t.date()), price=round(price, 2),
            return_pct=round(daily_returns[-1], 5),
            portfolio_value=round(total_value, 2),
            drawdown=breakdown.drawdown, volatility=breakdown.vol,
            correlation=breakdown.correlation, var_95=breakdown.var_95,
            crash_prob=breakdown.crash_prob, regime=breakdown.regime,
            certainty=decision.zekerheid, base_exposure=breakdown.base_exposure,
            vol_multiplier=breakdown.vol_multiplier,
            drawdown_multiplier=breakdown.drawdown_multiplier,
            correlation_multiplier=breakdown.correlation_multiplier,
            tail_multiplier=breakdown.tail_multiplier,
            regime_multiplier=breakdown.regime_multiplier,
            certainty_multiplier=breakdown.certainty_multiplier,
            final_exposure=breakdown.final_exposure, reason=breakdown.reason,
        ))

    metrics = _metrics(equity, daily_returns, n_trades, turnover, costs,
                       exposure_series, cash_days / max(1, len(equity)))
    return V2Result(name=name, metrics=metrics, equity=equity,
                    timestamps=timestamps, exposure=exposure_series, log=log)


def save_v2(result: V2Result, subdir: str = "") -> None:
    """Save the result as JSON (machine-readable)."""
    d = OUT / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.name}_metrics.json").write_text(json.dumps(result.metrics, indent=2))
    (d / f"{result.name}_equity.json").write_text(json.dumps({
        "ts": [str(t.date()) for t in result.timestamps],
        "equity": [round(x, 2) for x in result.equity],
        "exposure": result.exposure,
    }))
    (d / f"{result.name}_log.json").write_text(json.dumps(
        [e.__dict__ for e in result.log], indent=1
    ))

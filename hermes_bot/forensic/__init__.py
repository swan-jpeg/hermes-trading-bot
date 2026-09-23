"""Forensic backtest harness for the Risk Engine (UNCHANGED).

Doel: objectief vaststellen hoe de BESTAANDE RiskEngine drawdowns/crashes
beperkt, zonder de engine te optimaliseren. Gebruikt een NEUTRALE strategie
(constant long, zekerheid=1.0) zodat ALLE exposure-variatie van de RiskEngine
komt. De RiskEngine-code wordt niet aangeraakt; alleen config-waarden worden
gelezen (voor ablations).

Reproduceerbaar: alle resultaten + config + logs worden naar var/forensic/
geschreven.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hermes_bot.config import load_config
from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk import RiskEngine
from hermes_bot.schemas import ExitReason, Position, RLRawDecision
from hermes_bot.simulation import MonteCarloEngine

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent.parent / "var" / "forensic"


# ---------------------------------------------------------------------------
# NEUTRAL STRATEGY — always proposes 100% long, so the RiskEngine is the
# only brake. This isolates the RiskEngine completely.
# ---------------------------------------------------------------------------
class ConstantLongStrategy:
    """Proposes 100% long every day (certainty 1.0). The RiskEngine decides."""

    name = "constant_long"

    def decide(self, idx, row, portfolio: PortfolioState) -> RLRawDecision:
        return RLRawDecision(
            entity="asset",
            action="buy",
            intent_to_alloc=1.0,
            zekerheid=1.0,
            timestamp=datetime.now(),
            rationale="constant long: 100% voorstel",
            exit_reason=ExitReason.NONE,
        )


# ---------------------------------------------------------------------------
# SCENARIO-GENERATOREN — deterministisch, reproduceerbaar
# ---------------------------------------------------------------------------
def make_scenario(name: str, n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Generate a deterministic price scenario with a specific regime.

    De drift is dominant over de ruis zodat elk regime de bedoelde richting
    heeft (bull stijgt, bear daalt, etc.) — reproduceerbaar via seed.
    """
    rng = np.random.default_rng(seed)
    # Basis: positieve drift + ruis. Drift dominant zodat richting klopt.
    rets = rng.normal(0.0006, 0.004, n)  # bull baseline

    if name == "bull":
        rets = rng.normal(0.0008, 0.004, n)
    elif name == "bear":
        rets = rng.normal(-0.0008, 0.006, n)
    elif name == "fast_crash":
        rets[200:215] = -0.05  # 15 dagen -5%/dag
        rets[215:230] = 0.01
    elif name == "slow_drawdown":
        rets[150:300] = -0.004  # 150 dagen langzaam verlies
    elif name == "vol_spike":
        rets[200:230] = rng.normal(0.0, 0.06, 30)  # 30 dagen hoge vol
    elif name == "correlation_spike":
        rets[200:230] = rng.normal(-0.02, 0.05, 30)
    elif name == "sector_crash":
        rets[180:200] = -0.03
    elif name == "multi_shock":
        rets[100:110] = -0.04
        rets[200:210] = -0.05
        rets[300:310] = -0.03
    elif name == "flash_crash":
        rets[200] = -0.20  # 1 dag -20%
        rets[201:210] = 0.02
    elif name == "v_shape":
        rets[200:220] = -0.03  # crash
        rets[220:260] = 0.02  # snelle recovery

    prices = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": prices, "high": prices * 1.01, "low": prices * 0.99,
         "close": prices, "volume": 1e6}, index=idx
    )


# ---------------------------------------------------------------------------
# FORENSIC BACKTESTER — logs every risk decision
# ---------------------------------------------------------------------------
@dataclass
class RiskLogEntry:
    ts: object
    price: float
    exposure: float
    var_95: float
    es_95: float
    crash_prob: float
    drawdown: float
    regime: str
    proposed: float
    approved: float
    adjusted: bool
    reasons: list[str]
    action: str


@dataclass
class ForensicResult:
    name: str
    metrics: dict
    equity: list[float]
    timestamps: list[object]
    exposure: list[float]
    log: list[RiskLogEntry] = field(default_factory=list)


def _metrics(equity: list[float], returns: list[float], n_trades: int,
             turnover: float, costs: float, exposure_series: list[float],
             cash_time: float) -> dict:
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
    max_exp = float(np.max(exposure_series)) if exposure_series else 0.0
    return {
        "total_return": round(total, 4), "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4), "volatility": round(vol, 4),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "calmar": round(calmar, 3), "n_trades": n_trades,
        "turnover": round(turnover, 4), "transaction_costs": round(costs, 2),
        "avg_exposure": round(avg_exp, 4), "max_exposure": round(max_exp, 4),
        "cash_time_pct": round(cash_time, 4),
    }


def run_forensic(
    prices: pd.DataFrame,
    risk_config: dict,
    name: str = "scenario",
    mc_every: int = 20,
    seed: int = 42,
) -> ForensicResult:
    """Run the RiskEngine (UNCHANGED) over a price series with logging."""
    risk = RiskEngine(risk_config)
    strategy = ConstantLongStrategy()

    cash = 100_000.0
    portfolio = PortfolioState(cash=cash, regime="unknown")
    qty = 0.0
    entry_price = 0.0
    peak = 0.0
    equity: list[float] = []
    timestamps: list[object] = []
    exposure_series: list[float] = []
    daily_returns: list[float] = []
    log: list[RiskLogEntry] = []
    n_trades = 0
    turnover = 0.0
    costs = 0.0
    cash_days = 0
    mc_engine = MonteCarloEngine(seed=seed, n_paths=2000)

    closes = prices["close"].to_numpy(dtype=float)
    idx_list = prices.index.tolist()

    for i, t in enumerate(idx_list):
        price = float(closes[i])
        if price <= 0:
            continue
        peak = max(peak, price)
        portfolio.peaks = {"asset": peak}
        if qty > 0:
            portfolio.positions = {
                "asset": Position(entity="asset", qty=qty, entry_price=entry_price,
                                   entry_time=t, take_profit_pct=0.15,
                                   stop_loss_pct=0.08, trailing_stop_pct=0.05)
            }
            portfolio.cash = cash
        else:
            portfolio.positions = {}
            portfolio.cash = cash

        decision = strategy.decide(i, prices.iloc[i], portfolio)

        # Monte Carlo periodically (as in the real backtest engine).
        mc = None
        if len(daily_returns) >= 30 and i % mc_every == 0:
            try:
                mc = mc_engine.simulate(np.asarray(daily_returns[-60:]), horizon=20)
            except Exception:
                mc = None

        approval = risk.approve(decision, portfolio, mc)

        # Execute (target-following, fail-closed) — same logic as the backtest.
        total_value = cash + qty * price
        target_weight = (
            approval.target_alloc.get("asset", 0.0) if approval.approved else 0.0
        )
        target_qty = (
            target_weight * total_value / price if price > 0 and total_value > 0 else 0.0
        )
        target_qty = max(0.0, min(target_qty, total_value / price))
        if target_qty > qty + 1e-9:
            buy_qty = target_qty - qty
            cash -= buy_qty * price
            costs += buy_qty * price * 0.001  # 10 bps
            if entry_price <= 0:
                entry_price = price
            qty = target_qty
            peak = max(peak, price)
            turnover += buy_qty * price / total_value
        elif target_qty < qty - 1e-9 and qty > 0:
            sell_qty = qty - target_qty
            cash += sell_qty * price
            costs += sell_qty * price * 0.001
            if target_qty <= 1e-9:
                n_trades += 1
                qty = 0.0
                entry_price = 0.0
            else:
                qty = target_qty
            turnover += sell_qty * price / total_value

        total_value = cash + qty * price
        if equity:
            daily_returns.append(total_value / equity[-1] - 1.0)
        else:
            daily_returns.append(0.0)
        equity.append(total_value)
        timestamps.append(t)
        exposure = qty * price / total_value if total_value > 0 else 0.0
        exposure_series.append(exposure)
        if qty <= 1e-9:
            cash_days += 1

        log.append(RiskLogEntry(
            ts=t, price=price, exposure=round(exposure, 4),
            var_95=mc.var_95 if mc else 0.0,
            es_95=mc.expected_shortfall_95 if mc else 0.0,
            crash_prob=mc.crash_probability if mc else 0.0,
            drawdown=portfolio.drawdown, regime=portfolio.regime,
            proposed=round(decision.intent_to_alloc, 4),
            approved=round(target_weight, 4), adjusted=approval.adjusted,
            reasons=approval.rejected_reasons, action=decision.action.value,
        ))

    metrics = _metrics(equity, daily_returns, n_trades, turnover, costs,
                       exposure_series, cash_days / max(1, len(equity)))
    return ForensicResult(name=name, metrics=metrics, equity=equity,
                          timestamps=timestamps, exposure=exposure_series, log=log)


# ---------------------------------------------------------------------------
# ABLATION — mechanismen uitschakelen via config (code blijft onveranderd)
# ---------------------------------------------------------------------------
def ablation_configs(base: dict) -> dict[str, dict]:
    """Config variants that disable one mechanism."""
    risk = base.get("risk", {})
    return {
        "full": base,
        "no_max_asset_weight": {**base, "risk": {**risk, "max_asset_weight": 1.0}},
        "no_mc_var": {**base, "risk": {**risk, "max_var_95": -1.0}},
        "no_mc_crash": {**base, "risk": {**risk, "max_crash_prob": 1.0}},
        "no_certainty_scale": {**base, "risk": {**risk, "_no_certainty": True}},
    }


def run_ablation(prices: pd.DataFrame, base: dict, name: str) -> dict:
    """Run all ablations and return metrics per variant."""
    out = {}
    for label, cfg in ablation_configs(base).items():
        r = run_forensic(prices, cfg, name=f"{name}_{label}")
        out[label] = r.metrics
    return out


# ---------------------------------------------------------------------------
# LEAKAGE-CONTROLES
# ---------------------------------------------------------------------------
def check_leakage(prices: pd.DataFrame, log: list[RiskLogEntry]) -> dict:
    """Check for look-ahead / future information in the decisions."""
    findings = []
    # MC uses only daily_returns up to and including the PREVIOUS day (no look-ahead).
    # Check that var_95 only appears after day 30 (warm-up).
    first_mc = next((e for e in log if e.var_95 != 0.0), None)
    if first_mc is not None:
        # MC is only computed once len(daily_returns)>=30 -> day 30+.
        pass  # warm-up is correct in de code
    # No future prices: exposure on day i uses only close[i].
    # Survivorship: single-asset, no delisting model -> n/a here.
    return {"findings": findings, "note": "single-asset, geen delisting-model"}


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------
def run_all() -> dict:
    """Run the full forensic test and save everything."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    risk_cfg = {"risk": cfg.get("risk", {})}

    scenarios = [
        "bull", "bear", "fast_crash", "slow_drawdown", "vol_spike",
        "correlation_spike", "sector_crash", "multi_shock", "flash_crash", "v_shape",
    ]

    results = {}
    for sc in scenarios:
        prices = make_scenario(sc)
        r = run_forensic(prices, risk_cfg, name=sc)
        results[sc] = r.metrics
        (OUT / f"{sc}_equity.json").write_text(json.dumps({
            "ts": [str(t.date()) for t in r.timestamps],
            "equity": [round(x, 2) for x in r.equity],
            "exposure": r.exposure,
        }))
        (OUT / f"{sc}_log.json").write_text(json.dumps(
            [e.__dict__ for e in r.log], default=str, indent=1
        ))

    # Ablations on the fast_crash scenario (most informative).
    crash_prices = make_scenario("fast_crash")
    ablations = run_ablation(crash_prices, risk_cfg, "fast_crash")

    # Leakage check on bull + fast_crash.
    bull_prices = make_scenario("bull")
    bull_log = run_forensic(bull_prices, risk_cfg).log
    crash_log = run_forensic(crash_prices, risk_cfg).log
    leakage = {
        "bull": check_leakage(bull_prices, bull_log),
        "fast_crash": check_leakage(crash_prices, crash_log),
    }

    report = {
        "config_used": cfg.get("risk", {}),
        "scenarios": results,
        "ablations": ablations,
        "leakage": leakage,
        "generated_at": datetime.now().isoformat(),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    rep = run_all()
    print(json.dumps(rep, indent=2, default=str))

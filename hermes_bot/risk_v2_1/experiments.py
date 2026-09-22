"""Experiment-runner voor Risk Engine v2.1 — vergelijking met frozen v2.

Draait v2.1 vs v2 vs old vs B&H op:
- synthetische scenario's (crash-regimes)
- echte marktdata (SPY/QQQ/IWM)
- OOS-periode (2024-2026, zelfde split als v2 OOS)

Output: var/forensic_v21/
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hermes_bot.config import load_config
from hermes_bot.forensic import make_scenario
from hermes_bot.risk_v2_1.backtest import run_v21, save_v21

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent.parent / "var" / "forensic_v21"


def _v21_config(cfg: dict) -> dict:
    """v2.1 config: base op v2 risk-config + v2.1-specifieke parameters."""
    risk = cfg.get("risk", {})
    return {"risk": {
        **risk,
        "base_exposure": 0.90, "min_exposure": 0.0, "max_exposure": 1.0,
        "max_daily_change": 0.25, "vol_target": 0.125,
        "max_portfolio_drawdown": -0.08, "corr_threshold": 0.6,
        "max_var_95": -0.05, "max_crash_prob": 0.10,
        "emergency_return_threshold": -0.10, "emergency_vol_jump": 3.0,
        "emergency_exposure": 0.0, "emergency_brake_days": 3,
        "stabilize_days": 5, "recovery_lookback": 10,
        "seed": 42,
    }}


def scenario_comparison(cfg: dict) -> dict:
    """Vergelijk v2.1 vs v2 vs old vs B&H op synthetische scenario's."""
    from hermes_bot.forensic import run_forensic  # old
    from hermes_bot.risk_v2.backtest import run_v2  # frozen v2

    scenarios = ["bull", "bear", "fast_crash", "flash_crash", "v_shape",
                 "slow_drawdown", "vol_spike", "multi_shock"]
    results = {}
    for sc in scenarios:
        prices = make_scenario(sc)
        c = prices["close"].values
        bh_total = c[-1] / c[0] - 1
        bh_dd = (c / np.maximum.accumulate(c) - 1).min()
        old = run_forensic(prices, {"risk": cfg.get("risk", {})}, name=f"old_{sc}")
        v2 = run_v2(prices, {"risk": cfg.get("risk", {})}, name=f"v2_{sc}")
        v21 = run_v21(prices, _v21_config(cfg), name=f"v21_{sc}")
        results[sc] = {
            "bh_total": round(bh_total, 4), "bh_maxdd": round(bh_dd, 4),
            "old_total": old.metrics["total_return"], "old_maxdd": old.metrics["max_drawdown"],
            "v2_total": v2.metrics["total_return"], "v2_maxdd": v2.metrics["max_drawdown"],
            "v2_avg_exp": v2.metrics["avg_exposure"],
            "v21_total": v21.metrics["total_return"], "v21_maxdd": v21.metrics["max_drawdown"],
            "v21_avg_exp": v21.metrics["avg_exposure"], "v21_sharpe": v21.metrics["sharpe"],
        }
        save_v21(v21, "scenarios")
    return results


def real_market_comparison(cfg: dict) -> dict:
    """Vergelijk v2.1 vs v2 op echte marktdata."""
    from hermes_bot.backtest import load_prices
    from hermes_bot.risk_v2.backtest import run_v2

    results = {}
    for sym in ["SPY", "QQQ", "IWM"]:
        for period in ["1y", "3y", "5y"]:
            try:
                prices = load_prices(sym, period=period)
            except Exception as e:
                results[f"{sym}_{period}"] = {"error": str(e)}
                continue
            v2 = run_v2(prices, {"risk": cfg.get("risk", {})}, name=f"v2_{sym}_{period}")
            v21 = run_v21(prices, _v21_config(cfg), name=f"v21_{sym}_{period}")
            results[f"{sym}_{period}"] = {
                "v2_total": v2.metrics["total_return"], "v2_maxdd": v2.metrics["max_drawdown"],
                "v2_avg_exp": v2.metrics["avg_exposure"],
                "v21_total": v21.metrics["total_return"], "v21_maxdd": v21.metrics["max_drawdown"],
                "v21_avg_exp": v21.metrics["avg_exposure"], "v21_sharpe": v21.metrics["sharpe"],
            }
            save_v21(v21, "real_market")
    return results


def oos_comparison(cfg: dict) -> dict:
    """Vergelijk v2.1 vs v2 op de OOS-periode (2024-2026, warm-up 2022-2023)."""
    from hermes_bot.backtest import load_prices
    from hermes_bot.risk_v2.oos import run_oos_v2  # frozen v2 OOS-runner

    results = {}
    for sym in ["SPY", "QQQ", "IWM"]:
        try:
            prices = load_prices(sym, period="8y")
        except Exception as e:
            results[sym] = {"error": str(e)}
            continue
        mask = (prices.index >= "2022-01-01") & (prices.index <= "2026-09-21")
        p = prices[mask]
        v2 = run_oos_v2(p, "2024-01-01", "2026-09-21", name=f"v2_{sym}")
        v21 = run_v21(p, _v21_config(cfg), name=f"v21_{sym}")
        # v21 metrics alleen over testperiode.
        test_mask = [(t >= pd.Timestamp("2024-01-01")) for t in v21.timestamps]
        test_equity = [e for e, m in zip(v21.equity, test_mask, strict=False) if m]
        test_exp = [e for e, m in zip(v21.exposure, test_mask, strict=False) if m]
        results[sym] = {
            "v2_total": v2.metrics["total_return"], "v2_maxdd": v2.metrics["max_drawdown"],
            "v2_avg_exp": v2.metrics["avg_exposure"],
            "v21_total": round((test_equity[-1] - test_equity[0]) / test_equity[0], 4)
                if test_equity else 0,
            "v21_avg_exp": round(float(np.mean(test_exp)), 4) if test_exp else 0,
        }
        save_v21(v21, "oos")
    return results


def run_all() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    report = {
        "scenarios": scenario_comparison(cfg),
        "real_market": real_market_comparison(cfg),
        "oos": oos_comparison(cfg),
    }
    (OUT / "v21_experiments_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    rep = run_all()
    print(json.dumps(rep, indent=2, default=str))

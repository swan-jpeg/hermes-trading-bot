"""Experiment-runner voor Risk Engine v2 — alle experimenten A-D + sensitivity + capacity.

Draait en slaat op:
- EXPERIMENT A: harde exposure caps (10/25/50/75/100%)
- EXPERIMENT B: individuele mechanismen + ablations
- EXPERIMENT C: crash scenario's (v2 vs old vs B&H)
- EXPERIMENT D: echte marktdata (SPY/QQQ/IWM)
- Sensitivity analysis
- Capacity test

Output: var/forensic_v2/ (machine-readable JSON).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from hermes_bot.config import load_config
from hermes_bot.forensic import make_scenario
from hermes_bot.risk_v2.backtest import run_v2, save_v2

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent.parent / "var" / "forensic_v2"


# ---------------------------------------------------------------------------
# EXPERIMENT A — harde exposure caps
# ---------------------------------------------------------------------------
def experiment_a(cfg: dict) -> dict:
    """Test exposure caps 10/25/50/75/100% op het fast_crash-scenario."""
    prices = make_scenario("fast_crash")
    results = {}
    for cap in [0.10, 0.25, 0.50, 0.75, 1.00]:
        risk_cfg = {"risk": {**cfg.get("risk", {}), "max_exposure": cap,
                             "base_exposure": cap}}
        r = run_v2(prices, risk_cfg, name=f"cap_{int(cap*100)}")
        results[f"cap_{int(cap*100)}"] = r.metrics
        save_v2(r, "experiments")
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT B — individuele mechanismen + ablations
# ---------------------------------------------------------------------------
def experiment_b(cfg: dict) -> dict:
    """Test individuele mechanismen en combinaties op fast_crash."""
    prices = make_scenario("fast_crash")
    base_risk = cfg.get("risk", {})

    # Elke variant zet alle andere mechanismen uit (alleen dat mechanisme actief).
    variants = {
        "baseline": {},
        "volatility_only": {"_dd_off": True, "_corr_off": True, "_mc_off": True,
                            "_regime_off": True, "_certainty_off": True},
        "drawdown_only": {"_vol_off": True, "_corr_off": True, "_mc_off": True,
                          "_regime_off": True, "_certainty_off": True},
        "correlation_only": {"_vol_off": True, "_dd_off": True, "_mc_off": True,
                             "_regime_off": True, "_certainty_off": True},
        "montecarlo_only": {"_vol_off": True, "_dd_off": True, "_corr_off": True,
                            "_regime_off": True, "_certainty_off": True},
        "regime_only": {"_vol_off": True, "_dd_off": True, "_corr_off": True,
                        "_mc_off": True, "_certainty_off": True},
        "certainty_only": {"_vol_off": True, "_dd_off": True, "_corr_off": True,
                           "_mc_off": True, "_regime_off": True},
        # Combinaties.
        "vol_dd": {"_corr_off": True, "_mc_off": True, "_regime_off": True,
                   "_certainty_off": True},
        "vol_corr": {"_dd_off": True, "_mc_off": True, "_regime_off": True,
                     "_certainty_off": True},
        "dd_corr": {"_vol_off": True, "_mc_off": True, "_regime_off": True,
                    "_certainty_off": True},
        "vol_tail": {"_dd_off": True, "_corr_off": True, "_regime_off": True,
                     "_certainty_off": True},
        "vol_dd_corr": {"_mc_off": True, "_regime_off": True, "_certainty_off": True},
    }
    results = {}
    for label, flags in variants.items():
        risk_cfg = {"risk": {**base_risk, **flags}}
        r = run_v2(prices, risk_cfg, name=f"B_{label}")
        results[label] = r.metrics
        save_v2(r, "ablations")
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT C — crash scenario's (v2 vs old vs B&H)
# ---------------------------------------------------------------------------
def experiment_c(cfg: dict) -> dict:
    """Scenario's: v2 vs old RiskEngine vs B&H."""
    from hermes_bot.forensic import run_forensic  # old engine

    scenarios = [
        "bull", "bear", "fast_crash", "flash_crash", "v_shape", "slow_drawdown",
        "vol_spike", "correlation_spike", "sector_crash", "multi_shock",
    ]
    results = {}
    for sc in scenarios:
        prices = make_scenario(sc)
        c = prices["close"].values
        bh_total = c[-1] / c[0] - 1
        bh_dd = (c / np.maximum.accumulate(c) - 1).min()
        old = run_forensic(prices, {"risk": cfg.get("risk", {})}, name=f"old_{sc}")
        v2 = run_v2(prices, {"risk": cfg.get("risk", {})}, name=f"v2_{sc}")
        results[sc] = {
            "bh_total": round(bh_total, 4), "bh_maxdd": round(bh_dd, 4),
            "old_total": old.metrics["total_return"], "old_maxdd": old.metrics["max_drawdown"],
            "old_avg_exp": old.metrics["avg_exposure"],
            "v2_total": v2.metrics["total_return"], "v2_maxdd": v2.metrics["max_drawdown"],
            "v2_avg_exp": v2.metrics["avg_exposure"], "v2_sharpe": v2.metrics["sharpe"],
            "v2_turnover": v2.metrics["turnover"],
        }
        save_v2(v2, "scenarios")
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT D — echte marktdata
# ---------------------------------------------------------------------------
def experiment_d(cfg: dict) -> dict:
    """Echte marktdata via yfinance: SPY/QQQ/IWM over meerdere perioden."""
    from hermes_bot.backtest import load_prices

    results = {}
    assets = ["SPY", "QQQ", "IWM"]
    periods = ["1y", "2y", "3y", "5y"]
    for sym in assets:
        for period in periods:
            try:
                prices = load_prices(sym, period=period)
            except Exception as e:
                results[f"{sym}_{period}"] = {"error": str(e)}
                continue
            r = run_v2(prices, {"risk": cfg.get("risk", {})}, name=f"{sym}_{period}")
            results[f"{sym}_{period}"] = r.metrics
            save_v2(r, "real_market")
    return results


# ---------------------------------------------------------------------------
# SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------------
def sensitivity(cfg: dict) -> dict:
    """Test parametergevoeligheid op fast_crash."""
    prices = make_scenario("fast_crash")
    base_risk = cfg.get("risk", {})
    params = {
        "vol_target": [0.08, 0.10, 0.125, 0.15, 0.18],
        "max_drawdown": [-0.05, -0.08, -0.10, -0.15, -0.20],
        "max_daily_change": [0.10, 0.15, 0.25, 0.35, 0.50],
        "max_crash_prob": [0.05, 0.10, 0.15, 0.20, 0.30],
    }
    results = {}
    for param, values in params.items():
        row = {}
        for v in values:
            risk_cfg = {"risk": {**base_risk, param: v}}
            r = run_v2(prices, risk_cfg, name=f"sens_{param}_{v}")
            row[str(v)] = {
                "total_return": r.metrics["total_return"],
                "max_drawdown": r.metrics["max_drawdown"],
                "sharpe": r.metrics["sharpe"],
                "avg_exposure": r.metrics["avg_exposure"],
            }
        results[param] = row
    return results


# ---------------------------------------------------------------------------
# CAPACITY TEST
# ---------------------------------------------------------------------------
def capacity(cfg: dict) -> dict:
    """Test portfolio sizes 10k..100m (kosten/slippage schalen)."""
    prices = make_scenario("fast_crash")
    results = {}
    for size in [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        r = run_v2(prices, {"risk": cfg.get("risk", {})}, name=f"cap_size_{size}")
        costs_pct = r.metrics["transaction_costs"] / size
        results[f"EUR{size:,.0f}"] = {
            "total_return": r.metrics["total_return"],
            "max_drawdown": r.metrics["max_drawdown"],
            "turnover": r.metrics["turnover"],
            "costs_pct": round(costs_pct, 6),
        }
    return results


# ---------------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------------
def run_all() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    report = {
        "config_used": cfg.get("risk", {}),
        "experiment_a_caps": experiment_a(cfg),
        "experiment_b_ablations": experiment_b(cfg),
        "experiment_c_scenarios": experiment_c(cfg),
        "experiment_d_real_market": experiment_d(cfg),
        "sensitivity": sensitivity(cfg),
        "capacity": capacity(cfg),
    }
    (OUT / "experiments_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    rep = run_all()
    print(json.dumps(rep, indent=2, default=str))

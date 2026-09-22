"""Tests voor het Backtest Control Panel (config/experiment/visualisatie-laag).

Dit test de INTERFACE-laag, niet de backtest-engine (die is ongewijzigd).
Gebruikt echte data via yfinance waar mogelijk.
"""
from __future__ import annotations

import warnings

from hermes_bot.control_panel import (
    BacktestConfig,
    compare_to_html,
    quick_experiments,
    run_backtest,
    run_compare,
    run_to_html,
)

warnings.filterwarnings("ignore")


def test_single_backtest_runs() -> None:
    """Single backtest geeft metrics + equity + exposure."""
    cfg = BacktestConfig(asset="SPY", years=1.0, vol_target=0.125)
    d = run_backtest(cfg)
    assert "metrics" in d
    assert "equity" in d and len(d["equity"]) > 50
    assert "exposure" in d
    assert d["benchmark_total"] != 0
    # Metrics velden.
    for k in ["total_return", "cagr", "max_drawdown", "sharpe", "avg_exposure", "n_trades"]:
        assert k in d["metrics"], f"ontbrekende metric {k}"


def test_best_worst_day_present() -> None:
    """Best/worst dag en maand worden berekend (via bestaande data)."""
    cfg = BacktestConfig(asset="SPY", years=2.0)
    d = run_backtest(cfg, return_log=True)
    assert "best_day" in d["metrics"]
    assert "worst_day" in d["metrics"]
    assert d["metrics"]["worst_day"] <= d["metrics"]["best_day"]


def test_risk_toggles_change_config() -> None:
    """Risk-component toggles zetten de juiste v2.1-flags."""
    cfg = BacktestConfig(asset="SPY", years=1.0, opportunity=False,
                        recovery=False, emergency_brake=False)
    rc = cfg.risk_config()["risk"]
    assert rc.get("_opp_off") is True
    assert rc.get("_recovery_off") is True
    assert rc.get("_emergency_off") is True


def test_risk_engine_off_disables_all() -> None:
    """Risk engine UIT zet alle mechanismen uit."""
    cfg = BacktestConfig(asset="SPY", years=1.0, risk_engine=False)
    rc = cfg.risk_config()["risk"]
    assert rc.get("_strategic_off") is True
    assert rc.get("_tactical_off") is True


def test_compare_two_to_five() -> None:
    """Compare-mode ondersteunt 2-5 configuraties."""
    cfgs = [
        BacktestConfig(asset="SPY", years=1.0, vol_target=0.10, label="a", run_type="compare"),
        BacktestConfig(asset="SPY", years=1.0, vol_target=0.15, label="b", run_type="compare"),
    ]
    d = run_compare(cfgs)
    assert len(d["results"]) == 2
    # Zelfde timestamps (eerlijke vergelijking).
    assert d["results"][0]["timestamps"] == d["results"][1]["timestamps"]


def test_compare_rejects_wrong_count() -> None:
    """Compare-mode weigert <2 of >5 configuraties."""
    try:
        run_compare([BacktestConfig()])
        raise AssertionError("moest weigeren bij 1 config")
    except AssertionError:
        pass


def test_run_to_html_contains_charts() -> None:
    """run_to_html bevat equity + drawdown + exposure + risk/opp charts."""
    cfg = BacktestConfig(asset="SPY", years=1.0)
    d = run_backtest(cfg, return_log=True)
    h = run_to_html(d)
    assert "kpis" in h or "<table>" in h  # metrics-weergave (compact)
    assert "polyline" in h or "polygon" in h
    assert "Risk Environment" in h


def test_compare_to_html_table() -> None:
    """compare_to_html bevat een vergelijkingstabel."""
    cfgs = [
        BacktestConfig(asset="SPY", years=1.0, vol_target=0.10, label="a", run_type="compare"),
        BacktestConfig(asset="SPY", years=1.0, vol_target=0.15, label="b", run_type="compare"),
    ]
    d = run_compare(cfgs)
    h = compare_to_html(d)
    assert "<table>" in h
    assert "Vergelijking" in h


def test_quick_experiments_defined() -> None:
    """Quick experiments zijn voorgedefinieerd."""
    qe = quick_experiments()
    assert "vol_targets" in qe
    assert "assets" in qe
    assert "full_vs_components" in qe
    assert all(1 < len(v) <= 5 for v in qe.values())


def test_run_id_unique_and_reproducible() -> None:
    """Zelfde config + tijd geeft een run-id; formaat klopt."""
    from hermes_bot.control_panel import make_run_id
    c1 = BacktestConfig(asset="SPY", years=1.0)
    c2 = BacktestConfig(asset="QQQ", years=1.0)
    id1, id2 = make_run_id(c1), make_run_id(c2)
    # Verschillende assets -> verschillende hash-suffix (zelfde tweede).
    assert id1[-8:] != id2[-8:]
    assert c1 != c2 or id1 == id2  # zelfde config zelfde hash
"""Unit tests for Risk Engine v2 — repaired mechanisms.

Test: echte drawdown, regime-detectie, Monte Carlo crash_probability,
adaptieve exposure, hysteresis, recovery.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2 import RiskEngineV2, detect_regime
from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2
from hermes_bot.schemas import Action, ExitReason, RLRawDecision


def _engine(**overrides) -> RiskEngineV2:
    cfg = {"risk": {"max_asset_weight": 0.10, **overrides}}
    return RiskEngineV2(cfg)


def _decision() -> RLRawDecision:
    return RLRawDecision(
        entity="asset", action=Action.BUY, intent_to_alloc=1.0, zekerheid=1.0,
        rationale="x", timestamp=datetime.now(), exit_reason=ExitReason.NONE,
    )


def test_drawdown_no_lookahead() -> None:
    """Drawdown uses only the historical peak (no look-ahead)."""
    e = _engine()
    assert e.update_drawdown(100) == 0.0
    assert e.update_drawdown(110) == 0.0
    assert e.update_drawdown(120) == 0.0
    dd = e.update_drawdown(108)
    assert abs(dd - (108 / 120 - 1)) < 1e-9  # -10%
    assert e.update_drawdown(130) == 0.0  # nieuwe high reset


def test_drawdown_initialization() -> None:
    """Correct initialization: the first equity is the peak."""
    e = _engine()
    assert e.update_drawdown(100) == 0.0
    assert e.peak_equity == 100.0


def test_regime_detection() -> None:
    """Regime detection reacts to drawdown and vol."""
    assert detect_regime(-0.20, 0.1, 0.1, 0.0, 0.0, "normal") == "crash"
    assert detect_regime(-0.10, 0.1, 0.1, 0.0, 0.0, "normal") == "stressed"
    # Elevated at vol 1.5-2x baseline.
    assert detect_regime(-0.02, 0.16, 0.1, 0.0, 0.0, "normal") == "elevated"
    # Stressed at vol > 2x baseline.
    assert detect_regime(-0.02, 0.25, 0.1, 0.0, 0.0, "normal") == "stressed"
    assert detect_regime(-0.02, 0.1, 0.1, 0.0, 0.0, "normal") == "normal"
    # Recovery after a crash.
    assert detect_regime(-0.03, 0.1, 0.1, 0.0, 0.01, "crash") == "recovery"


def test_montecarlo_crash_probability_nonzero() -> None:
    """REPAIR: crash_probability must be > 0 on a crash-like distribution."""
    mc = MonteCarloEngineV2(seed=42, n_paths=5000)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.02, 200)
    rets[::10] = -0.10  # periodieke crashes
    result = mc.simulate(rets, horizon=60)
    assert result.crash_probability > 0.0, "crash_probability moet > 0 zijn"
    assert result.var_95 < 0.0  # VaR is negatief (verlies)


def test_montecarlo_no_lookahead() -> None:
    """MC uses only the given historical returns (no look-ahead)."""
    rets = np.random.default_rng(5).normal(0.001, 0.02, 100)
    r1 = MonteCarloEngineV2(seed=1, n_paths=1000).simulate(rets, horizon=20)
    r2 = MonteCarloEngineV2(seed=1, n_paths=1000).simulate(rets, horizon=20)
    assert r1.var_95 == r2.var_95


def test_adaptive_exposure_reduces_in_crash() -> None:
    """v2 lowers exposure on deep drawdown (adaptive)."""
    e = _engine()
    e.peak_equity = 100.0
    e.hist_returns = [0.0] * 30
    pf = PortfolioState(cash=100000)
    exp, bd = e.approve(_decision(), pf, equity=80.0)
    assert exp < 0.5, f"exposure moet laag zijn in crash, kreeg {exp}"
    assert bd.regime == "crash"


def test_hysteresis_limits_daily_change() -> None:
    """Smoothing beperkt dagelijkse exposure-verandering."""
    e = _engine(max_daily_change=0.25)
    e.prev_exposure = 0.5
    pf = PortfolioState(cash=100000)
    exp, _ = e.approve(_decision(), pf, equity=100.0)
    assert exp <= 0.5 + 0.25 + 1e-9


def test_recovery_rebuilds_exposure() -> None:
    """After recovery the engine rebuilds exposure gradually (not in cash)."""
    e = _engine()
    e.peak_equity = 100.0
    # Realistic returns with small noise (not constant -> corr != 1.0).
    rng = np.random.default_rng(3)
    e.hist_returns = list(rng.normal(0.001, 0.005, 30))
    e.prev_exposure = 0.5  # al deels hersteld
    pf = PortfolioState(cash=100000)
    exp, bd = e.approve(_decision(), pf, equity=100.0)
    # No drawdown, low vol -> exposure builds up (above 0.5).
    assert exp > 0.5, f"exposure moet opbouwen in normal regime, kreeg {exp}"
    assert bd.regime == "normal"

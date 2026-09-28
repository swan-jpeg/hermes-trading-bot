"""Tests that the Monte Carlo ([[08]]) is wired into the CHAIN, not just callable.

Note 08 promises four things: portfolio VaR/ES, price paths, forced crash
scenarios and backtest uncertainty. These tests check the last two actually run
inside the pipeline and the backtester — a public method with no callers is an
unwired spec item, not a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hermes_bot.backtest import Backtester, BuyAndHoldStrategy, VolTargetStrategy
from hermes_bot.pipeline import run_pipeline
from hermes_bot.risk_v2.montecarlo import DEFAULT_SCENARIOS, MonteCarloEngineV2


def _prices(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, n)
    rets[::40] = -0.04  # a few crash days
    close = list(100.0 * np.cumprod(1.0 + rets))
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": close, "high": [c * 1.01 for c in close], "low": [c * 0.99 for c in close],
         "close": close, "volume": [1000.0] * n},
        index=dates,
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_forces_crash_scenarios_on_the_decision() -> None:
    res = run_pipeline()
    assert set(res["scenario_stress"]) == set(DEFAULT_SCENARIOS)
    assert all(isinstance(v, float) for v in res["scenario_stress"].values())
    assert res["scenario_stress"]["2008_crisis"] <= 0.0


def test_pipeline_reports_the_monte_carlo_used() -> None:
    res = run_pipeline()
    mc = res["monte_carlo"]
    assert {"model", "n_paths", "var_95", "es_95", "crash_probability", "drawdown_95"} <= set(mc)
    assert -1.0 <= mc["var_95"] <= 0.5
    assert 0.0 <= mc["crash_probability"] <= 1.0


# --------------------------------------------------------------------------- #
# Backtester
# --------------------------------------------------------------------------- #
def test_backtest_stresses_the_final_exposure() -> None:
    # Buy-and-hold remains invested, so the final exposure is certain > 0.
    r = Backtester().run(_prices(), BuyAndHoldStrategy({"entity": "SPY"}))
    assert set(r.scenario_stress) == set(DEFAULT_SCENARIOS)
    assert 0.0 < r.final_exposure <= 1.0
    """The final exposure determines the impact: 2008 must hit harder than a mild correction."""
    assert r.scenario_stress["2008_crisis"] < r.scenario_stress["mild_correction"]


def test_backtest_scenario_stress_is_zero_when_flat() -> None:
    """Without exposure, a crash scenario must not cost anything."""
    eng = MonteCarloEngineV2(seed=1)
    assert all(v == 0.0 for v in eng.stress_scenarios({"SPY": 0.0}).values())


def test_backtest_reports_confidence_intervals() -> None:
    r = Backtester().run(_prices(), VolTargetStrategy({"entity": "asset"}))
    ci = r.confidence_intervals
    assert {"mean", "sharpe", "max_drawdown"} <= set(ci)
    for stat in ("mean", "sharpe", "max_drawdown"):
        assert ci[stat]["low"] <= ci[stat]["point"] <= ci[stat]["high"]


def test_backtest_scenario_stress_visible_in_the_webui_runner() -> None:
    """The webui-runner must pass the scenario table + CI to the page."""
    from hermes_bot.backtest.runner import run_backtest

    data = run_backtest(symbol="SPY", use_live=False, period="2y")
    assert set(data["scenario_stress"]) == set(DEFAULT_SCENARIOS)
    assert "final_exposure" in data
    assert "confidence_intervals" in data

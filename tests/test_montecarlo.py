"""Tests for the Monte Carlo engine ([[08]]) — the single MC implementation.

Covers every part that used to be unwired or missing:
- one engine only (`hermes_bot/simulation` is a shim);
- four path models (bootstrap / gbm / jump_diffusion / regime);
- copulas with real tail dependence (t > gaussian);
- multi-asset + correlation, and the impact-agent shock;
- drawdown/VaR/crash probability from the SAME paths;
- forced crash scenarios (scenario_table) + backtest confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pytest

from hermes_bot.risk_v2.montecarlo import (
    DEFAULT_SCENARIOS,
    MonteCarloEngineV2,
)


def _crash_like(n: int = 250, seed: int = 0) -> np.ndarray:
    """Volatile series with periodic crashes (drawdowns are guaranteed)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, n)
    rets[::25] = -0.10
    return rets


def _multi(n: int = 250, seed: int = 0) -> np.ndarray:
    """Three correlated assets (2D n_assets x n_days)."""
    rng = np.random.default_rng(seed)
    base = _crash_like(n, seed)
    return np.vstack([
        base,
        base * 0.8 + rng.normal(0, 0.005, n),
        rng.normal(0.0002, 0.01, n),
    ])


# --------------------------------------------------------------------------- #
# One engine (no second implementation)
# --------------------------------------------------------------------------- #
def test_simulation_module_is_a_shim_to_the_one_engine() -> None:
    """`hermes_bot.simulation` must point at the single engine class."""
    from hermes_bot import simulation

    assert simulation.MonteCarloEngine is MonteCarloEngineV2
    assert simulation.MonteCarloEngineV2 is MonteCarloEngineV2
    assert simulation.DEFAULT_SCENARIOS is DEFAULT_SCENARIOS


# --------------------------------------------------------------------------- #
# Path models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["bootstrap", "gbm", "jump_diffusion", "regime"])
def test_every_model_gives_valid_tail_risk(model: str) -> None:
    rets = _crash_like()
    r = MonteCarloEngineV2(seed=42, n_paths=3000, model=model).simulate(rets, horizon=60)
    assert r.model == model
    assert -1.0 < r.var_95 < 0.0
    assert r.expected_shortfall_95 <= r.var_95  # ES is deeper in the tail
    assert 0.0 <= r.crash_probability <= 1.0
    assert r.max_drawdown_distribution[95] < 0.0  # drawdown is écht negatief
    assert set(r.percentiles) == {5, 10, 25, 50, 75, 90, 95}


def test_jump_diffusion_has_fatter_tail_than_gbm() -> None:
    """Poisson jumps must make the crash tail heavier than pure GBM."""
    rets = np.random.default_rng(3).normal(0.0, 0.01, 300)
    jumps = {"intensity": 0.05, "mean": -0.06, "std": 0.02}
    gbm = MonteCarloEngineV2(seed=7, n_paths=4000, model="gbm").simulate(rets, horizon=60)
    jd = MonteCarloEngineV2(seed=7, n_paths=4000, model="jump_diffusion").simulate(
        rets, horizon=60, jumps=jumps)
    assert jd.var_95 < gbm.var_95
    assert jd.crash_probability > gbm.crash_probability


def test_regime_model_reacts_to_volatility_clustering() -> None:
    """A series with a real full-regime switch must yield crash probability."""
    rng = np.random.default_rng(11)
    calm = rng.normal(0.0005, 0.006, 150)
    wild = rng.normal(-0.002, 0.035, 150)
    rets = np.concatenate([calm, wild])
    r = MonteCarloEngineV2(seed=5, n_paths=3000, model="regime").simulate(rets, horizon=60)
    assert 0.0 < r.crash_probability <= 1.0
    assert r.var_95 < 0.0


def test_unknown_model_and_copula_raise() -> None:
    with pytest.raises(ValueError):
        MonteCarloEngineV2(model="nope")
    with pytest.raises(ValueError):
        MonteCarloEngineV2(copula="nope")
    eng = MonteCarloEngineV2(seed=1, n_paths=500)
    with pytest.raises(ValueError):
        eng.simulate(_crash_like(), horizon=10, model="nope")


# --------------------------------------------------------------------------- #
# Consistency: drawdown from THE SAME paths as the quantiles
# --------------------------------------------------------------------------- #
def test_drawdown_comes_from_the_same_paths_as_the_quantiles() -> None:
    """At horizon=1, the drawdown follows the path-return exactly.

    Per path dd = min(r, 0). With a series where every draw is negative
    (dd = r for every path) ALL drawdown quantiles must equal the return
    quantiles exactly. If the drawdown (as in the old v1/v2) were drawn
    separately, this would not hold.
    """
    always_down = np.random.default_rng(4).normal(-0.02, 0.01, 200)
    r = MonteCarloEngineV2(seed=4, n_paths=5000).simulate(always_down, horizon=1)
    for q in sorted(r.max_drawdown_distribution):
        assert np.isclose(r.max_drawdown_distribution[q], r.percentiles[q], atol=1e-12)


def test_single_day_horizon_cannot_exceed_the_daily_loss() -> None:
    """Drawdown probability at horizon=1 cannot be larger than the loss of 1 day."""
    r = MonteCarloEngineV2(seed=4, n_paths=5000).simulate(_crash_like(), horizon=1)
    assert r.max_drawdown_distribution[95] >= r.percentiles[5]  # dd never worse than VaR
    assert r.max_drawdown_distribution[50] >= min(r.percentiles[50], 0.0) - 1e-12


def test_drawdown_distribution_is_not_zero_on_crash_data() -> None:
    """Regression on the old bug: dd95 was 0.0 -> crash limit dead."""
    r = MonteCarloEngineV2(seed=42, n_paths=4000).simulate(_crash_like(), horizon=60)
    assert r.max_drawdown_distribution[95] < -0.02
    assert r.crash_probability > 0.0


# --------------------------------------------------------------------------- #
# Copulas: staart-afhankelijkheid
# --------------------------------------------------------------------------- #
def test_t_copula_has_tail_dependence_gaussian_does_not() -> None:
    """The joint crash probability must be much higher with the t-copula."""
    hist = np.vstack([np.random.default_rng(0).normal(0, 0.02, 300),
                      np.random.default_rng(1).normal(0, 0.02, 300)])

    def joint_crash(copula: str, df: float = 4.0) -> float:
        eng = MonteCarloEngineV2(seed=3, n_paths=5000, copula=copula, copula_df=df)
        paths = eng._sample_copula(hist, 50, None, copula, df)
        a, b = paths[:, :, 0].ravel(), paths[:, :, 1].ravel()
        return float(np.mean((a <= np.percentile(a, 5)) & (b <= np.percentile(b, 5))))

    gauss = joint_crash("gaussian")
    t = joint_crash("t")
    assert t > 2.0 * gauss  # duidelijke staart-afhankelijkheid
    assert 0.0 < gauss < 0.02  # gaussian copula: nagenoeg onafhankelijke staarten


def test_copula_marginals_match_historical_returns() -> None:
    """The margins must remain those of the history (Iman-Conover)."""
    hist = _multi()
    eng = MonteCarloEngineV2(seed=9, n_paths=2000, copula="t")
    paths = eng._sample_copula(hist, 40, None, "t", 5.0)
    for a in range(hist.shape[0]):
        drawn = paths[:, :, a].ravel()
        assert drawn.min() >= hist[a].min() - 1e-12
        assert drawn.max() <= hist[a].max() + 1e-12


# --------------------------------------------------------------------------- #
# Multi-asset + impact-shock
# --------------------------------------------------------------------------- #
def test_multi_asset_simulate_with_correlation() -> None:
    multi = _multi()
    corr = np.corrcoef(multi)
    r = MonteCarloEngineV2(seed=42, n_paths=2000).simulate(multi, horizon=40, corr=corr)
    assert r.n_paths == 2000
    assert r.var_95 < 0.0
    assert r.crash_probability >= 0.0


def test_impact_shock_bends_the_path_downwards_multi_asset() -> None:
    multi = _multi()
    corr = np.corrcoef(multi)
    base = MonteCarloEngineV2(seed=42, n_paths=3000).simulate(multi, horizon=40, corr=corr)
    shocked = MonteCarloEngineV2(seed=42, n_paths=3000).simulate_with_shock(
        multi, horizon=40, direction=0.0, magnitude=1.0, corr=corr)
    assert shocked.percentiles[50] < base.percentiles[50]
    assert shocked.crash_probability >= base.crash_probability


def test_impact_shock_supports_per_asset_direction() -> None:
    multi = _multi()
    r = MonteCarloEngineV2(seed=42, n_paths=1500).simulate_with_shock(
        multi, horizon=30, direction=np.array([0.0, 1.0, 0.5]),
        magnitude=np.array([1.0, 1.0, 0.0]))
    assert r.var_95 < 0.0  # asset 0 takes the negative shock


def test_invalid_input_shape_raises() -> None:
    with pytest.raises(ValueError):
        MonteCarloEngineV2(seed=1, n_paths=100).simulate(np.zeros((2, 3, 4)), horizon=5)


# --------------------------------------------------------------------------- #
# Crash-scenario's forceren ([[08]] punt 3)
# --------------------------------------------------------------------------- #
def test_stress_scenarios_has_all_default_crashes() -> None:
    table = MonteCarloEngineV2(seed=1).stress_scenarios({"AAPL": 0.10})
    assert set(table) == set(DEFAULT_SCENARIOS)
    assert min(table.values()) < 0.0
    assert MonteCarloEngineV2(seed=1).worst_scenario(table)[0] == "2008_crisis"


def test_stress_scenarios_normalises_weights_and_uses_asset_classes() -> None:
    """Percentages are normalized; BOND/CASH get their own shock."""
    table = MonteCarloEngineV2(seed=1).stress_scenarios({"AAPL": 50.0, "BOND": 50.0})
    # 2008: 0.5 * -0.40 + 0.5 * +0.05 = -0.175
    assert table["2008_crisis"] == pytest.approx(-0.175, abs=1e-6)


def test_stress_scenarios_respects_the_cash_buffer() -> None:
    """60% invested = 60% of the crash, not 100% (cash was not affected)."""
    table = MonteCarloEngineV2(seed=1).stress_scenarios({"SPY": 0.60})
    assert table["2008_crisis"] == pytest.approx(-0.24, abs=1e-6)
    assert table["mild_correction"] == pytest.approx(-0.048, abs=1e-6)


def test_stress_scenarios_without_weights_is_zero() -> None:
    table = MonteCarloEngineV2(seed=1).stress_scenarios({})
    assert all(v == 0.0 for v in table.values())


def test_scenario_table_is_filled_when_a_portfolio_is_given() -> None:
    r = MonteCarloEngineV2(seed=42, n_paths=1500).simulate(
        _crash_like(), horizon=30, portfolio={"AAPL": 0.10})
    assert set(r.scenario_table) == set(DEFAULT_SCENARIOS)
    assert r.scenario_table["2008_crisis"] < 0.0


def test_scenario_table_empty_without_portfolio() -> None:
    r = MonteCarloEngineV2(seed=42, n_paths=1000).simulate(_crash_like(), horizon=20)
    assert r.scenario_table == {}


# --------------------------------------------------------------------------- #
# Backtest-onzekerheid ([[08]] punt 4)
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(8)
    rets = rng.normal(-0.0005, 0.02, 300)
    ci = MonteCarloEngineV2(seed=2).bootstrap_ci(rets, n_resamples=800)
    for stat in ("mean", "sharpe", "max_drawdown"):
        lo, point, hi = ci[stat]["low"], ci[stat]["point"], ci[stat]["high"]
        assert lo <= point <= hi
    assert ci["max_drawdown"]["high"] < 0.0


def test_bootstrap_ci_handles_tiny_input() -> None:
    ci = MonteCarloEngineV2(seed=2).bootstrap_ci(np.array([0.01]), n_resamples=50)
    assert ci["mean"]["point"] == 0.0


# --------------------------------------------------------------------------- #
# Reproduceerbaarheid
# --------------------------------------------------------------------------- #
def test_same_seed_and_data_gives_identical_results() -> None:
    rets = _crash_like()
    r1 = MonteCarloEngineV2(seed=1, n_paths=1000).simulate(rets, horizon=20)
    r2 = MonteCarloEngineV2(seed=1, n_paths=1000).simulate(rets, horizon=20)
    assert r1.var_95 == r2.var_95
    assert r1.crash_probability == r2.crash_probability
    assert r1.max_drawdown_distribution == r2.max_drawdown_distribution

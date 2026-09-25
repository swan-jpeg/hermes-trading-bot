"""Impact-agent -> indicators + Monte Carlo link.

The impact agent decides WHICH instruments are affected by an event. This
module turns that decision into (a) technical indicators for exactly those
instruments (Category 8, computed from price data — NOT webscraping) and
(b) an impact-shocked Monte Carlo tail-risk estimate for the affected set.

Prices come from yfinance when online; offline (backtest/replay) callers
provide historical prices directly, so the chain never requires a network
call to produce an AssetContext.
"""
from __future__ import annotations

import numpy as np

from hermes_bot.indicators import compute_indicators, neutral_indicators
from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2
from hermes_bot.schemas import MonteCarloResult


def fetch_closes(entity: str, lookback: int = 120) -> np.ndarray | None:
    """Return the last `lookback` closing prices for an entity (oldest first).

    Returns None if the entity is not a tradeable ticker / not fetchable,
    so the caller can fall back to neutral indicators.
    """
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        df = yf.download(entity, period=f"{lookback}d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty or "Close" not in df:
            return None
        closes = df["Close"].dropna().astype(float).to_numpy().reshape(-1)
        if closes.size == 0:
            return None
        return closes[-lookback:]
    except Exception:
        return None


def entity_indicators(entity: str, closes: np.ndarray | None) -> dict[str, float]:
    """Compute Category-8 indicators for one entity (neutral if no data)."""
    if closes is None or closes.size < 5:
        return neutral_indicators()
    try:
        return compute_indicators(np.asarray(closes, dtype=float).reshape(-1))
    except Exception:
        return neutral_indicators()


def impact_shocked_mc(
    closes: np.ndarray,
    direction: float,
    magnitude: float,
    horizon: int = 20,
    seed: int = 42,
    n_paths: int = 1000,
) -> MonteCarloResult:
    """Run an impact-shocked Monte Carlo over an entity's price history.

    Returns a MonteCarloResult with the drift shock applied; if there is not
    enough data to bootstrap a return distribution, an empty-safe result is
    returned (zero VaR, no crash probability).
    """
    if closes is None or closes.size < 31:
        return MonteCarloResult(
            n_paths=0, horizon=horizon, percentiles={}, var_95=0.0,
            expected_shortfall_95=0.0, max_drawdown_distribution={},
            crash_probability=0.0)
    closes = np.asarray(closes, dtype=float).reshape(-1)
    returns = np.diff(closes[-90:]) / np.maximum(closes[-90:-1], 1e-9)
    engine = MonteCarloEngineV2(seed=seed, n_paths=n_paths)
    return engine.simulate_with_shock(returns, horizon, direction, magnitude)
"""Tests for the technical-indicator inputs (Category 8)."""
from __future__ import annotations

import numpy as np

from hermes_bot.indicators import (
    INDICATOR_FIELDS,
    compute_indicators,
    neutral_indicators,
)


def test_compute_indicators_covers_all_fields() -> None:
    """compute_indicators returns every declared field."""
    closes = np.linspace(100, 130, 50)
    out = compute_indicators(closes)
    assert set(out.keys()) == set(INDICATOR_FIELDS)
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_rising_market_bullish_signals() -> None:
    """A clean, steady uptrend shows momentum/trend up and low vol."""
    closes = np.linspace(100, 150, 60)
    out = compute_indicators(closes)
    assert out["momentum_10"] > 0.5      # rising
    assert out["trend_strength"] > 0.9   # near-perfect linear trend
    assert out["ema_cross"] > 0.5        # short EMA above long
    assert out["vol_20"] < 0.5           # steady rise = low realized vol


def test_falling_market_bearish_signals() -> None:
    """A clean, steady downtrend shows bearish momentum and low vol."""
    closes = np.linspace(150, 100, 60)
    out = compute_indicators(closes)
    assert out["momentum_10"] < 0.5      # falling
    assert out["ema_cross"] < 0.5        # short EMA below long


def test_trend_strength_bounded() -> None:
    """trend_strength is r^2, always in [0, 1]."""
    rng = np.random.default_rng(0)
    closes = np.cumprod(1 + rng.normal(0, 0.005, 100)) * 100
    out = compute_indicators(closes)
    assert 0.0 <= out["trend_strength"] <= 1.0


def test_flat_market_neutral_momentum() -> None:
    """A flat line has near-zero momentum and low vol."""
    closes = np.full(50, 100.0)
    out = compute_indicators(closes)
    assert abs(out["momentum_10"] - 0.5) < 1e-6
    assert out["ema_cross"] == 0.5
    assert out["vol_20"] < 0.1


def test_neutral_indicators_all_neutral() -> None:
    """No-data default keeps the vector complete and neutral."""
    n = neutral_indicators()
    assert set(n.keys()) == set(INDICATOR_FIELDS)
    assert n["rsi_14"] == 0.5
    assert n["ema_cross"] == 0.5
    assert n["momentum_10"] == 0.5
    assert n["vol_20"] == 0.5
    assert n["trend_strength"] == 0.0


def test_short_history_returns_neutral() -> None:
    """Too little price history degrades gracefully to neutral/default."""
    out = compute_indicators(np.array([100.0, 101.0, 102.0]))
    assert set(out.keys()) == set(INDICATOR_FIELDS)
    assert out["rsi_14"] == 0.5
    assert out["vol_20"] == 0.5

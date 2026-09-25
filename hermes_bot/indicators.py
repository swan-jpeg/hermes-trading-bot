"""Technical indicators for the RL model (Category 8).

A small, fixed set of standard quant trading indicators computed from price
history. These are NOT webscraping inputs — they are market-derived features
that the RL model uses alongside fusion + impact-agent to decide.

The impact agent decides which entities are affected; this module computes
the indicators for exactly those entities (prices fetched lazily via
yfinance, or provided directly for offline/backtest/replay).

Indicators (all normalized to 0..1 so PPO stays stable):
  rsi_14           — Relative Strength Index (0..1 directly; 0.5 = neutral)
  ema_cross        — EMA(short) vs EMA(long) gap, normalized (0.5 = flat)
  momentum_10      — 10-day return momentum, squashed to 0..1 (0.5 = flat)
  vol_20           — 20-day annualized volatility, normalized (0 = low, 1 = high)
  trend_strength   — 20-day linear trend r^2 (0..1, how clean the trend is)

Keeping it to five, low-complexity features: the bot's goal is not advanced
alpha, it is resilience + direction on top of fusion/impact signals.
"""
from __future__ import annotations

import numpy as np

# The indicator fields (subset of AssetContext category 8).
INDICATOR_FIELDS: tuple[str, ...] = (
    "rsi_14",
    "ema_cross",
    "momentum_10",
    "vol_20",
    "trend_strength",
)


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index over `period`, squashed to 0..1 (0.5 neutral)."""
    if len(closes) < period + 1:
        return 0.5
    diff = np.diff(closes)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 1.0 if avg_gain > 0 else 0.5
    rs = avg_gain / avg_loss
    rsi_raw = 100 - (100 / (1 + rs))
    return float(np.clip(rsi_raw / 100.0, 0, 1))


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def _ema_cross(closes: np.ndarray, short: int = 12, long: int = 26) -> float:
    """EMA(short) vs EMA(long) normalized gap: 0.5 = flat, >0.5 = bullish."""
    n = len(closes)
    if n < long + 1:
        return 0.5
    ema_s = _ema(closes, short)[-1]
    ema_l = _ema(closes, long)[-1]
    denom = max(abs(ema_l), 1e-9)
    gap = (ema_s - ema_l) / denom
    # Map gap (roughly -0.05..+0.05) to 0..1 around 0.5.
    return float(np.clip(0.5 + gap * 10.0, 0, 1))


def _momentum(closes: np.ndarray, lookback: int = 10) -> float:
    """10-day return momentum squashed to 0..1 (0.5 = flat)."""
    if len(closes) <= lookback:
        return 0.5
    ret = closes[-1] / max(closes[-1 - lookback], 1e-9) - 1.0
    # Map return (roughly -15%..+15%) to 0..1.
    return float(np.clip(0.5 + ret * 3.0, 0, 1))


def _vol(closes: np.ndarray, period: int = 20) -> float:
    """20-day annualized volatility normalized to 0..1 (0.5 ~ 20% ann.)."""
    if len(closes) <= period:
        return 0.5
    rets = np.diff(closes[-period - 1:]) / max(closes[-period - 1], 1e-9)
    ann = float(np.std(rets, ddof=1) * np.sqrt(252))
    # Map 0%..60% annualized vol to 0..1 (0.5 at ~30%).
    return float(np.clip(ann / 0.60, 0, 1))


def _trend(closes: np.ndarray, lookback: int = 20) -> float:
    """Linear trend r^2 over `lookback` days: 0 = choppy, 1 = clean trend."""
    n = len(closes)
    if n < 5:
        return 0.0
    x = np.arange(min(n, lookback), dtype=float)
    y = closes[-len(x):]
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr * corr, 0, 1))


def compute_indicators(closes: np.ndarray) -> dict[str, float]:
    """Compute the indicator set from a closing-price series (oldest first)."""
    closes = np.asarray(closes, dtype=float)
    return {
        "rsi_14": round(_rsi(closes), 4),
        "ema_cross": round(_ema_cross(closes), 4),
        "momentum_10": round(_momentum(closes), 4),
        "vol_20": round(_vol(closes), 4),
        "trend_strength": round(_trend(closes), 4),
    }


def neutral_indicators() -> dict[str, float]:
    """Neutral default when no price data is available (vector stays complete)."""
    return {
        "rsi_14": 0.5,
        "ema_cross": 0.5,
        "momentum_10": 0.5,
        "vol_20": 0.5,
        "trend_strength": 0.0,
    }

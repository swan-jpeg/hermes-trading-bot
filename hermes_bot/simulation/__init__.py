"""LAYER 6: Monte Carlo — DEPRECATED SHIM ([[08]]).

The Monte Carlo implementation moved to `hermes_bot/risk_v2/montecarlo.py`
(`MonteCarloEngineV2`), the only engine. This module exists purely so that
existing imports (`from hermes_bot.simulation import MonteCarloEngine`) keep
working and point at that same class.

Use this in new code:

    from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2

So there is no second, divergent Monte Carlo engine any more: the old version
here used a constant daily return per path (`paths[i].mean()/horizon`), which
made the drawdown distribution unrealistic (dd95 = 0.0) and `crash_probability`
always 0.0 — the crash limit in the risk engine was therefore functionally dead.
"""
from __future__ import annotations

from hermes_bot.risk_v2.montecarlo import (  # noqa: F401
    DD_PERCENTILES,
    DEFAULT_SCENARIOS,
    MODELS,
    PERCENTILES,
    MonteCarloEngineV2,
)

# Backwards-compatible name: the old class WAS the Monte Carlo engine.
MonteCarloEngine = MonteCarloEngineV2

__all__ = [
    "DEFAULT_SCENARIOS",
    "MonteCarloEngine",
    "MonteCarloEngineV2",
]

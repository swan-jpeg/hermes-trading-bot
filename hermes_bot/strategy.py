"""Deterministic baseline decision strategies (T09).

This module replaces the hard-coded pipeline BUY with typed, testable
strategies that propose a complete portfolio target map from point-in-time
market and portfolio inputs.  RL policies are optional and disabled by default.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Protocol

import numpy as np

from hermes_bot.impact.contracts import ImpactRecord
from hermes_bot.portfolio import PortfolioState
from hermes_bot.schemas import DecisionProposal, MarketState


class Strategy(Protocol):
    """Protocol for a deterministic target-weight proposal strategy."""

    @property
    def strategy_id(self) -> str:
        """Stable strategy identifier."""
        ...

    @property
    def strategy_version(self) -> str:
        """Strategy version string."""
        ...

    def propose_targets(
        self,
        as_of: datetime,
        market_state: MarketState,
        portfolio_state: PortfolioState,
        impact_records: list[ImpactRecord],
    ) -> DecisionProposal:
        """Return a complete target-weight map for the next cycle."""
        ...


def _last_price(instrument_id: str, market_state: MarketState) -> float | None:
    closes = market_state.closes_by_instrument.get(instrument_id)
    if closes:
        return float(closes[-1])
    return None


def _portfolio_weights(
    portfolio_state: PortfolioState,
    market_state: MarketState,
) -> tuple[dict[str, float], float]:
    """Current NAV weights from positions and cash.

    Returns ``(weights, cash_weight)``.  Uses the last available close for each
    position; falls back to ``entry_price`` when no market data are available.
    """
    weights: dict[str, float] = {}
    total_value = portfolio_state.cash
    prices: dict[str, float] = {}
    for entity, pos in portfolio_state.positions.items():
        price = _last_price(entity, market_state)
        if price is None:
            price = pos.entry_price
        prices[entity] = price
        total_value += pos.qty * price

    if total_value <= 0 or not math.isfinite(total_value):
        return weights, 1.0

    for entity, pos in portfolio_state.positions.items():
        weights[entity] = (pos.qty * prices[entity]) / total_value

    cash_weight = portfolio_state.cash / total_value
    return weights, cash_weight


def _instrument_universe(
    market_state: MarketState,
    portfolio_state: PortfolioState,
) -> set[str]:
    return set(market_state.closes_by_instrument) | set(portfolio_state.positions)


class NoTradeBaseline:
    """Preserve current holdings; impact direction does not alter targets."""

    strategy_id = "no_trade_baseline"
    strategy_version = "t09/v1"

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}

    def propose_targets(
        self,
        as_of: datetime,
        market_state: MarketState,
        portfolio_state: PortfolioState,
        impact_records: list[ImpactRecord],
    ) -> DecisionProposal:
        current_weights, _ = _portfolio_weights(portfolio_state, market_state)
        target: dict[str, float] = {
            iid: current_weights.get(iid, 0.0)
            for iid in _instrument_universe(market_state, portfolio_state)
        }
        return DecisionProposal(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            as_of=as_of,
            market_state_as_of=market_state.as_of,
            current_weights=dict(current_weights),
            target_weights=target,
            reasons=["preserve current holdings; no impact-driven allocation change"],
            impact_record_count=len(impact_records),
        )


class VolTargetBaseline:
    """Volatility-target baseline: size positions inversely to realized vol.

    For each instrument with at least 20 trailing close-to-close returns before
    ``as_of`` the target weight is
    ``min(0.10, target_annual_vol / max(realized_annual_vol, 0.05))``.
    Instruments with insufficient history keep their current weight.
    """

    strategy_id = "vol_target_baseline"
    strategy_version = "t09/v1"

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or {}
        decision_cfg = self.cfg.get("decision", {})
        vt_cfg = decision_cfg.get("vol_target", {})
        self.target_annual_vol = float(
            vt_cfg.get("target_annual_vol")
            or self.cfg.get("risk", {}).get("vol_target", 0.125)
        )

    def propose_targets(
        self,
        as_of: datetime,
        market_state: MarketState,
        portfolio_state: PortfolioState,
        impact_records: list[ImpactRecord],
    ) -> DecisionProposal:
        current_weights, _ = _portfolio_weights(portfolio_state, market_state)
        universe = _instrument_universe(market_state, portfolio_state)
        target: dict[str, float] = {}
        reasons: list[str] = [f"target_annual_vol={self.target_annual_vol}"]

        for iid in universe:
            closes = market_state.closes_by_instrument.get(iid, [])
            if len(closes) >= 21:
                recent = np.asarray(closes[-21:], dtype=float)
                returns = np.diff(recent) / recent[:-1]
                realized_vol = float(np.std(returns, ddof=1) * math.sqrt(252))
                weight = min(0.10, self.target_annual_vol / max(realized_vol, 0.05))
                weight = float(np.clip(weight, 0.0, 1.0))
                target[iid] = weight
                reasons.append(
                    f"{iid}: realized_annual_vol={realized_vol:.4f} -> target={weight:.4f}"
                )
            else:
                target[iid] = current_weights.get(iid, 0.0)
                reasons.append(
                    f"{iid}: insufficient history ({len(closes)} closes) -> hold"
                )

        return DecisionProposal(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            as_of=as_of,
            market_state_as_of=market_state.as_of,
            current_weights=dict(current_weights),
            target_weights=target,
            reasons=reasons,
            impact_record_count=len(impact_records),
        )


def build_strategy(config: dict | None = None, policy: object | None = None) -> Strategy:
    """Factory for the deterministic baseline strategy.

    ``policy`` is accepted for forward compatibility but is **not** used unless
    an explicit RL-enable flag is added later; the default is deterministic
    baseline only.
    """
    cfg = config or {}
    decision_cfg = cfg.get("decision", {})
    name = decision_cfg.get("strategy", "no_trade")
    if name == "vol_target":
        return VolTargetBaseline(cfg)
    return NoTradeBaseline(cfg)

"""Smoke-test: schemas + risico-engine + portfolio werken end-to-end."""
from __future__ import annotations

from datetime import UTC, datetime

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk import RiskEngine
from hermes_bot.schemas import Action, RLRawDecision


def _config() -> dict:
    return {"risk": {"max_asset_weight": 0.10, "max_leverage": 0.0, "cash_min_in_crash": 0.30}}


def test_risk_engine_approved_below_limit() -> None:
    eng = RiskEngine(_config())
    d = RLRawDecision(
        entity="AAPL",
        action=Action.BUY,
        intent_to_alloc=0.05,
        zekerheid=0.8,
        rationale="test",
        timestamp=datetime.now(UTC),
    )
    r = eng.approve(d, PortfolioState())
    assert r.approved
    assert abs(r.target_alloc["AAPL"] - 0.045) < 1e-9  # 0.05*(0.5+0.5*0.8)


def test_risk_engine_caps_at_asset_limit() -> None:
    eng = RiskEngine(_config())
    d = RLRawDecision(
        entity="AAPL",
        action=Action.BUY,
        intent_to_alloc=1.0,
        zekerheid=1.0,
        rationale="test",
        timestamp=datetime.now(UTC),
    )
    r = eng.approve(d, PortfolioState())
    assert r.target_alloc["AAPL"] <= 0.10  # geknipt op asset-limiet
    assert r.adjusted


def test_portfolio_rebalance_delta() -> None:
    from hermes_bot.portfolio import PortfolioAllocator

    al = PortfolioAllocator({})
    cur = PortfolioState(positions={"AAPL": 0.05}, cash=0.0)
    delta = al.rebalance(cur, {"AAPL": 0.08, "SPY": 0.02})
    assert {"asset": "SPY", "change": 0.02} in delta

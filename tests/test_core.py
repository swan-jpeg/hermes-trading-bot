"""Smoke-test: schemas + risico-engine + portfolio + winst-nemen-logica."""
from __future__ import annotations

from datetime import UTC, datetime

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk import RiskEngine
from hermes_bot.schemas import Action, ExitReason, Position, RLRawDecision
from hermes_bot.simulation import MonteCarloEngine


def _config() -> dict:
    return {"risk": {"max_asset_weight": 0.10, "max_leverage": 0.0, "cash_min_in_crash": 0.30}}


def _decision(entity: str, action: Action, alloc: float, zekerheid: float = 1.0,
              exit_reason: ExitReason = ExitReason.NONE) -> RLRawDecision:
    return RLRawDecision(
        entity=entity,
        action=action,
        intent_to_alloc=alloc,
        zekerheid=zekerheid,
        rationale="test",
        timestamp=datetime.now(UTC),
        exit_reason=exit_reason,
    )


def test_risk_engine_approved_below_limit() -> None:
    eng = RiskEngine(_config())
    r = eng.approve(_decision("AAPL", Action.BUY, 0.05, zekerheid=0.8), PortfolioState())
    assert r.approved
    assert abs(r.target_alloc["AAPL"] - 0.045) < 1e-9  # 0.05*(0.5+0.5*0.8)


def test_risk_engine_caps_at_asset_limit() -> None:
    eng = RiskEngine(_config())
    r = eng.approve(_decision("AAPL", Action.BUY, 1.0), PortfolioState())
    assert r.target_alloc["AAPL"] <= 0.10  # geknipt op asset-limiet
    assert r.adjusted


def test_risk_engine_always_approves_exit() -> None:
    """Taking profit / limiting loss must never be blocked."""
    eng = RiskEngine(_config())
    r = eng.approve(
        _decision("AAPL", Action.SELL, -1.0, exit_reason=ExitReason.TAKE_PROFIT),
        PortfolioState(),
    )
    assert r.approved
    assert r.target_alloc["AAPL"] == 0.0


def test_risk_engine_monte_carlo_reduces_alloc() -> None:
    """High VaR95 / crash probability reduces the allowed allocation."""
    import numpy as np

    eng = RiskEngine(_config())
    # Low vol -> allocation unchanged but MC fields filled.
    hist = np.random.default_rng(1).normal(0, 0.005, 250)
    mc = MonteCarloEngine(seed=1, n_paths=2000).simulate(hist, horizon=30)
    r = eng.approve(_decision("AAPL", Action.BUY, 0.10), PortfolioState(), mc)
    assert r.target_alloc["AAPL"] <= 0.10
    assert r.var_95 != 0.0
    assert r.es_95 != 0.0

    # High vol -> VaR95 well below the -5% threshold -> allocation must shrink.
    hist_high = np.random.default_rng(2).normal(0, 0.06, 250)
    mc_high = MonteCarloEngine(seed=2, n_paths=4000).simulate(hist_high, horizon=30)
    r_high = eng.approve(_decision("AAPL", Action.BUY, 0.10), PortfolioState(), mc_high)
    assert mc_high.var_95 < -0.05  # bevestig dat de drempel overschreden wordt
    assert r_high.target_alloc["AAPL"] < 0.10  # begrensd door MC


def test_position_take_profit() -> None:
    pos = Position(entity="AAPL", qty=100.0, entry_price=100.0, entry_time=datetime.now(UTC))
    assert pos.exit_reason_at(118.0, 118.0) == ExitReason.TAKE_PROFIT  # +18% > 15%


def test_position_stop_loss() -> None:
    pos = Position(entity="AAPL", qty=100.0, entry_price=100.0, entry_time=datetime.now(UTC))
    assert pos.exit_reason_at(90.0, 100.0) == ExitReason.STOP_LOSS  # -10% < -8%


def test_position_trailing_stop() -> None:
    pos = Position(entity="AAPL", qty=100.0, entry_price=100.0, entry_time=datetime.now(UTC))
    # Was at +12% (peak 112), now back to 106 -> -5.4% from peak = trailing stop.
    assert pos.exit_reason_at(106.0, 112.0) == ExitReason.TRAILING_STOP


def test_position_no_exit_inside_bands() -> None:
    pos = Position(entity="AAPL", qty=100.0, entry_price=100.0, entry_time=datetime.now(UTC))
    assert pos.exit_reason_at(103.0, 104.0) == ExitReason.NONE


def _position(entity: str, qty: float) -> Position:
    return Position(entity=entity, qty=qty, entry_price=100.0, entry_time=datetime.now(UTC))


def test_portfolio_rebalance_delta() -> None:
    from hermes_bot.portfolio import PortfolioAllocator

    al = PortfolioAllocator({})
    cur = PortfolioState(positions={"AAPL": _position("AAPL", 0.05)})
    delta = al.rebalance(cur, {"AAPL": 0.08, "SPY": 0.02})
    assert {"asset": "SPY", "change": 0.02} in delta


def test_portfolio_propose_target_sell_reduces() -> None:
    from hermes_bot.portfolio import PortfolioAllocator

    al = PortfolioAllocator({})
    cur = PortfolioState(positions={"AAPL": _position("AAPL", 0.10)})
    target = al.propose_target([_decision("AAPL", Action.SELL, 0.05)], cur)
    assert target["AAPL"] == 0.05  # 0.10 - 0.05


def test_rule_policy_takes_profit_on_existing_position() -> None:
    """The RL policy must take profit on an existing position that is above
    de take-profit staat, ook als sentiment positief blijft."""
    from hermes_bot.rl import RLFusionModel

    pos = Position(entity="AAPL", qty=100.0, entry_price=100.0, entry_time=datetime.now(UTC))
    portfolio = PortfolioState(positions={"AAPL": pos}, regime="bull")
    fusion = {"entity_id": "AAPL", "emotie": {"sentiment": 0.5}, "zekerheid": 0.9, "kwaliteit": 0.8}
    rl = RLFusionModel()
    decision = rl.decide(fusion, {"entity_id": "AAPL"}, portfolio, current_price=118.0)
    assert decision.action == Action.SELL
    assert decision.exit_reason == ExitReason.TAKE_PROFIT
    assert decision.intent_to_alloc < 0  # verkoop de winst


def test_rule_policy_no_position_buys_on_sentiment() -> None:
    from hermes_bot.rl import RLFusionModel

    portfolio = PortfolioState(regime="bull")
    fusion = {"entity_id": "MSFT", "emotie": {"sentiment": 0.6}, "zekerheid": 0.9, "kwaliteit": 0.8}
    rl = RLFusionModel()
    decision = rl.decide(fusion, {"entity_id": "MSFT"}, portfolio, current_price=200.0)
    assert decision.action == Action.BUY
    assert decision.exit_reason == ExitReason.NONE
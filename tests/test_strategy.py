"""Tests for deterministic baseline decision strategies (T09)."""
from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from hermes_bot.impact.contracts import ImpactRecord
from hermes_bot.portfolio import PortfolioState, Position
from hermes_bot.risk_v2_1 import RiskEngineV21
from hermes_bot.schemas import (
    DecisionProposal,
    MarketState,
    RiskRequest,
    RiskStatus,
    RiskTargetPosition,
)
from hermes_bot.strategy import (
    NoTradeBaseline,
    Strategy,
    VolTargetBaseline,
    build_strategy,
)

AS_OF = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _position(entity: str, qty: float, price: float) -> Position:
    return Position(
        entity=entity,
        qty=qty,
        entry_price=price,
        entry_time=AS_OF,
    )


def _market_state(closes: dict[str, list[float]]) -> MarketState:
    return MarketState(as_of=AS_OF, closes_by_instrument=closes)


def _impact(direction: str) -> ImpactRecord:
    return ImpactRecord(
        entity_id="ent_test",
        impact_type="revenue",
        direction=direction,  # type: ignore[arg-type]
        horizon_days=[1, 5, 20],
        estimated_return=None,
        estimated_return_semantics="none",
        subjective_confidence=0.5,
        confidence_label="medium",
        calibration_id=None,
        evidence_ids=["ev1"],
        reason_codes=["rule_test"],
        warnings=[],
    )


def test_no_trade_preserves_current_weights() -> None:
    portfolio = PortfolioState(
        cash=80_000.0,
        positions={"AAPL": _position("AAPL", 100.0, 200.0)},
    )
    market = _market_state({"AAPL": [200.0] * 10})
    strategy: Strategy = NoTradeBaseline()
    proposal = strategy.propose_targets(AS_OF, market, portfolio, [])

    assert proposal.strategy_id == "no_trade_baseline"
    assert proposal.target_weights == {"AAPL": 0.2}


def test_no_trade_includes_zero_weight_instruments() -> None:
    portfolio = PortfolioState(cash=100_000.0)
    market = _market_state({"AAPL": [100.0] * 25, "MSFT": [200.0] * 25})
    strategy: Strategy = NoTradeBaseline()
    proposal = strategy.propose_targets(AS_OF, market, portfolio, [])

    assert proposal.target_weights == {"AAPL": 0.0, "MSFT": 0.0}


def test_vol_target_formula_constant_returns() -> None:
    """Constant returns -> zero realized vol -> target capped at max 0.10."""
    portfolio = PortfolioState(cash=100_000.0)
    closes = list(np.cumprod([1.0] + [1.001] * 20) * 100.0)
    market = _market_state({"AAPL": closes})
    strategy: Strategy = VolTargetBaseline()
    proposal = strategy.propose_targets(AS_OF, market, portfolio, [])

    assert proposal.strategy_id == "vol_target_baseline"
    assert proposal.target_weights["AAPL"] == pytest.approx(0.10, abs=1e-9)


def test_vol_target_uses_only_past_bars() -> None:
    """Only the trailing 20 returns (21 closes) influence the vol estimate."""
    portfolio = PortfolioState(cash=100_000.0)
    # First 50 bars high vol, last 21 bars constant -> low realized vol.
    high_vol = list(100.0 * (1 + np.random.default_rng(1).normal(0.0, 0.03, 50)))
    constant = list(np.cumprod([1.0] + [1.0005] * 20) * high_vol[-1])
    closes = high_vol + constant
    market = _market_state({"AAPL": closes})
    strategy: Strategy = VolTargetBaseline()
    proposal = strategy.propose_targets(AS_OF, market, portfolio, [])

    # With low trailing vol the target hits the 0.10 cap.
    assert proposal.target_weights["AAPL"] == pytest.approx(0.10, abs=1e-9)


def test_vol_target_insufficient_history_holds_current_weight() -> None:
    portfolio = PortfolioState(
        cash=90_000.0,
        positions={"AAPL": _position("AAPL", 50.0, 200.0)},
    )
    market = _market_state({"AAPL": [200.0] * 5})
    strategy: Strategy = VolTargetBaseline()
    proposal = strategy.propose_targets(AS_OF, market, portfolio, [])

    assert proposal.target_weights["AAPL"] == pytest.approx(0.1, abs=1e-9)


def test_negative_or_unknown_impact_does_not_force_buy() -> None:
    portfolio = PortfolioState(cash=100_000.0)
    closes = list(np.cumprod([1.0] + [1.001] * 20) * 100.0)
    market = _market_state({"AAPL": closes})
    strategy: Strategy = VolTargetBaseline()

    for direction in ("negative", "unknown"):
        proposal = strategy.propose_targets(
            AS_OF, market, portfolio, [_impact(direction)]
        )
        assert proposal.target_weights["AAPL"] == pytest.approx(0.10, abs=1e-9)


def test_build_strategy_defaults_to_no_trade() -> None:
    assert build_strategy({}).strategy_id == "no_trade_baseline"


def test_build_strategy_selects_vol_target() -> None:
    cfg = {"decision": {"strategy": "vol_target"}}
    assert build_strategy(cfg).strategy_id == "vol_target_baseline"


def _proposal_to_request(proposal: DecisionProposal, max_position_count: int = 8) -> RiskRequest:
    """Convert a strategy proposal into a whole-portfolio risk request."""
    positions = [
        RiskTargetPosition(
            instrument_id=iid,
            target_weight=w,
            price_available_at=AS_OF,
            is_price_fresh=True,
            decision_time=AS_OF,
            strategy_id=proposal.strategy_id,
            strategy_version=proposal.strategy_version,
            reason_codes=proposal.reasons,
        )
        for iid, w in proposal.target_weights.items()
    ]
    return RiskRequest(
        request_id="req-test",
        as_of=AS_OF,
        nav=100_000.0,
        cash_weight=1.0 - math.fsum(proposal.current_weights.values()),
        current_weights=dict(proposal.current_weights),
        proposed_positions=positions,
        constraints_version="risk-v2.1",
    )


def test_multiasset_proposal_passed_once_to_risk(monkeypatch) -> None:
    """The pipeline calls approve_portfolio exactly once per cycle."""
    from hermes_bot.impact import ImpactResult
    from hermes_bot.pipeline import Pipeline

    calls: list[tuple[object, ...]] = []
    original = RiskEngineV21.approve_portfolio

    def spy(self, request):
        calls.append((self, request))
        return original(self, request)

    monkeypatch.setattr(RiskEngineV21, "approve_portfolio", spy)

    def stub(entity: str):
        rng = np.random.default_rng(abs(hash(entity)) % 2**31)
        return np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100

    pipe = Pipeline(
        {"decision": {"strategy": "vol_target"}, "closes_provider": stub}
    )

    # Dummy impact agent that points the pipeline at two test entities.
    class _FakeAgent:
        def analyze(self, text: str, source: str = "news", entity_id: str = "") -> ImpactResult:
            return ImpactResult(
                event=text or "test",
                impacted=[{"entity": "AAPL"}, {"entity": "MSFT"}],
            )

        def to_fusion_inputs(self, result, sentiment=0.0):
            return []

    pipe.impact_agent = _FakeAgent()
    pipe.run()

    assert len(calls) == 1
    _, request = calls[0]
    assert isinstance(request, RiskRequest)
    assert len(request.proposed_positions) == 2


def test_risk_rejection_leaves_no_approved_targets() -> None:
    """A proposal exceeding max_position_count is rejected with zero targets."""
    portfolio = PortfolioState(cash=100_000.0)
    rng = np.random.default_rng(7)
    closes = {
        f"ASSET{i}": list(np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100)
        for i in range(3)
    }
    market = _market_state(closes)
    strategy: Strategy = VolTargetBaseline()
    proposal = strategy.propose_targets(AS_OF, market, portfolio, [])

    engine = RiskEngineV21(
        {"risk": {
            "max_asset_weight": 0.10,
            "max_gross_exposure": 1.00,
            "max_leverage": 0.0,
            "min_cash_weight": 0.0,
            "max_position_count": 1,
        }},
        clock=lambda: AS_OF,
    )
    result = engine.approve_portfolio(_proposal_to_request(proposal))

    assert result.status == RiskStatus.REJECTED
    assert not result.executable
    assert all(w == 0.0 for w in result.approved_targets.values())


def test_deterministic_repeated_run() -> None:
    """Repeated identical inputs yield identical proposals and approvals."""
    portfolio = PortfolioState(cash=100_000.0)
    rng = np.random.default_rng(42)
    closes = {"AAPL": list(np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100)}
    market = _market_state(closes)
    strategy: Strategy = VolTargetBaseline()

    p1 = strategy.propose_targets(AS_OF, market, portfolio, [])
    p2 = strategy.propose_targets(AS_OF, market, portfolio, [])
    assert p1.target_weights == p2.target_weights

    engine = RiskEngineV21(
        {"risk": {
            "max_asset_weight": 0.10,
            "max_gross_exposure": 1.00,
            "max_leverage": 0.0,
            "min_cash_weight": 0.0,
            "max_position_count": 8,
        }},
        clock=lambda: AS_OF,
    )
    r1 = engine.approve_portfolio(_proposal_to_request(p1))
    r2 = engine.approve_portfolio(_proposal_to_request(p2))
    assert r1.model_dump() == r2.model_dump()

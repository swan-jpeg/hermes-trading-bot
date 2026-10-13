"""Unit tests for Risk Engine v2.1 — two-axis scores, layers, recovery.

Test: Risk Environment Score, Opportunity Score (onafhankelijk van risk),
risk budget matrix, emergency brake, recovery engine, effectieve exposure.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2_1 import EmergencyBrake, RecoveryEngine, RiskEngineV21
from hermes_bot.risk_v2_1.scores import (
    OpportunityScore,
    RiskEnvironmentScore,
    risk_budget_from_scores,
    state_label,
)
from hermes_bot.schemas import (
    Action,
    ExitReason,
    RiskRequest,
    RiskStatus,
    RiskTargetPosition,
    RLRawDecision,
)


def _engine(**overrides) -> RiskEngineV21:
    return RiskEngineV21({"risk": {"max_asset_weight": 0.10, **overrides}})


def _decision() -> RLRawDecision:
    return RLRawDecision(
        entity="asset", action=Action.BUY, intent_to_alloc=1.0, zekerheid=1.0,
        rationale="x", timestamp=datetime.now(), exit_reason=ExitReason.NONE,
    )


def test_risk_score_high_in_crash() -> None:
    """Risk Environment Score is high on deep drawdown + high vol."""
    s = RiskEnvironmentScore({"risk": {}})
    score = s.score(vol=0.30, vol_baseline=0.125, drawdown=-0.20, corr=0.7,
                    var_95=-0.08, crash_prob=0.4, extreme_return=-0.05)
    assert score > 60, f"risk score moet hoog zijn in crash, kreeg {score}"


def test_risk_score_low_in_bull() -> None:
    """Risk Environment Score is low in a healthy market."""
    s = RiskEnvironmentScore({"risk": {}})
    score = s.score(vol=0.10, vol_baseline=0.125, drawdown=0.0, corr=0.1,
                    var_95=-0.01, crash_prob=0.0, extreme_return=0.0)
    assert score < 40, f"risk score moet laag zijn in bull, kreeg {score}"


def test_opportunity_not_inverse_risk() -> None:
    """Opportunity is NOT (100 - risk): computed independently."""
    opp = OpportunityScore({"risk": {}})
    # Lage vol, positieve momentum -> hoge opportunity.
    returns = [0.003] * 20
    score_high, _ = opp.score(returns, vol=0.10, vol_baseline=0.125,
                              drawdown=0.0, corr=0.1)
    # Hoge vol, negatief momentum -> lage opportunity.
    returns_bad = [-0.003] * 20
    score_low, _ = opp.score(returns_bad, vol=0.30, vol_baseline=0.125,
                             drawdown=-0.10, corr=0.7)
    assert score_high > score_low, "opportunity moet hoger zijn bij goede condities"
    # Opportunity is not simply the inverse of risk: score_high is not
    # equal to (100 - risk_score) for the same conditions.
    risk = RiskEnvironmentScore({"risk": {}})
    risk_high = risk.score(0.10, 0.125, 0.0, 0.1, -0.01, 0.0, 0.0)
    assert abs(score_high - (100 - risk_high)) > 5, "opportunity mag niet inverse risk zijn"


def test_risk_budget_matrix() -> None:
    """Risk budget matrix: 4 kwadranten."""
    # Low risk + high opp -> AGGRESSIVE (hoog budget).
    assert risk_budget_from_scores(10, 90) > 0.7
    # High risk + low opp -> DEFENSIVE (laag budget).
    assert risk_budget_from_scores(90, 10) < 0.3
    # Low risk + low opp -> NEUTRAL (gematigd).
    b = risk_budget_from_scores(10, 10)
    assert 0.3 < b < 0.7
    # High risk + high opp -> SELECTIVE (risk domineert).
    assert risk_budget_from_scores(90, 90) < 0.3


def test_state_labels() -> None:
    assert state_label(90, 10) == "DEFENSIVE"
    assert state_label(90, 90) == "SELECTIVE"
    assert state_label(10, 90) == "AGGRESSIVE"
    assert state_label(10, 10) == "NEUTRAL"


def test_emergency_brake_triggers() -> None:
    """Emergency brake triggers on an extreme daily return."""
    b = EmergencyBrake({"risk": {"emergency_return_threshold": -0.10}})
    assert b.check(-0.15, 0.1, 0.1) is True
    assert b.active is True
    # Exposure becomes 0 during the brake.
    assert b.tick() == 0.0


def test_emergency_brake_no_false_trigger() -> None:
    """Emergency brake does NOT trigger on normal vol."""
    b = EmergencyBrake({"risk": {"emergency_return_threshold": -0.10}})
    assert b.check(-0.01, 0.1, 0.1) is False
    assert b.active is False
    assert b.tick() == 1.0


def test_recovery_state_machine() -> None:
    """Recovery engine doorloopt NORMAL -> ALERT -> DEFENSIVE -> STABILIZATION -> RECOVERY."""
    r = RecoveryEngine({"risk": {"stabilize_days": 2}})
    # NORMAL -> ALERT on stress.
    assert r.update(-0.10, 0.3, 0.125, [0.0] * 10) == "ALERT"
    # ALERT -> DEFENSIVE on deeper stress.
    assert r.update(-0.15, 0.4, 0.125, [-0.01] * 10) == "DEFENSIVE"
    # DEFENSIVE -> STABILIZATION on recovery.
    assert r.update(-0.03, 0.1, 0.125, [0.001] * 10) == "STABILIZATION"
    # STABILIZATION dag 1 (state_days < stabilize_days) -> blijft STABILIZATION.
    assert r.update(-0.02, 0.1, 0.125, [0.002] * 10) == "STABILIZATION"
    # STABILIZATION dag 2 (state_days >= stabilize_days) + positief -> RECOVERY.
    assert r.update(-0.02, 0.1, 0.125, [0.002] * 10) == "RECOVERY"
    # RECOVERY -> NORMAL on trend recovery.
    assert r.update(-0.01, 0.1, 0.125, [0.003] * 10) == "NORMAL"


def test_v21_reduces_exposure_in_crash() -> None:
    """v2.1 lowers exposure on deep drawdown (adaptive)."""
    e = _engine()
    e.peak_equity = 100.0
    # Realistische crash-returns: hoge vol + negatief.
    rng = np.random.default_rng(1)
    e.hist_returns = list(rng.normal(-0.01, 0.03, 30))
    pf = PortfolioState(cash=100000)
    exp, bd = e.approve(_decision(), pf, equity=80.0)
    assert exp < 0.5, f"exposure moet laag zijn in crash, kreeg {exp}"
    assert bd.risk_score > 50, f"risk score moet hoog zijn in crash, kreeg {bd.risk_score}"


def test_v21_high_exposure_in_bull() -> None:
    """v2.1 has high exposure in a healthy bull (low risk, high opp)."""
    e = _engine()
    e.peak_equity = 100.0
    rng = np.random.default_rng(3)
    e.hist_returns = list(rng.normal(0.002, 0.005, 30))
    e.prev_exposure = 0.5
    pf = PortfolioState(cash=100000)
    exp, bd = e.approve(_decision(), pf, equity=100.0)
    assert exp > 0.5, f"exposure moet hoog zijn in bull, kreeg {exp}"
    assert bd.opp_score > 50


def test_regime_alpha_lowers_exposure_in_crash() -> None:
    """regime_sentiment (risk input) reduces exposure during crash."""

    def exp_for(regime_sent: float) -> float:
        e = _engine()
        e.peak_equity = 100.0
        rng = np.random.default_rng(2)
        e.hist_returns = list(rng.normal(0.0005, 0.006, 40))
        e.prev_exposure = 0.5
        pf = PortfolioState(cash=100000)
        # Loop through so that _smooth can resolve its clamp.
        for i in range(5):
            exp, bd = e.approve(_decision(), pf, 100000 * (1 + 0.001 * i),
                                alpha_signals={"regime_sentiment": regime_sent,
                                               "regional_score": 0.5})
        return exp

    crash_exp = exp_for(-0.8)
    bull_exp = exp_for(0.5)
    assert crash_exp < bull_exp, f"crash exposure {crash_exp} moet < bull {bull_exp} zijn"


def test_regional_score_adds_opportunity() -> None:
    """regional_score as alpha signal increases the overall score monotonically."""
    opp = OpportunityScore({"risk": {}})
    returns = [0.002] * 20
    lo, bd_lo = opp.score(returns, vol=0.10, vol_baseline=0.125, drawdown=0.0,
                          corr=0.1, alpha_signals={"regional_score": 0.1})
    hi, bd_hi = opp.score(returns, vol=0.10, vol_baseline=0.125, drawdown=0.0,
                          corr=0.1, alpha_signals={"regional_score": 0.9})
    assert hi > lo, "hogere regionale score moet meer opportunity geven"
    assert "regional_score" in bd_hi["alpha_signals"]


# ===========================================================================
# T01 — whole-portfolio v2.1 approval (§4.6): deterministic hard limits
# ===========================================================================
AS_OF = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _pf_limits(**overrides) -> dict:
    base = {
        "max_asset_weight": 0.10,
        "max_gross_exposure": 1.00,
        "max_leverage": 0.0,
        "min_cash_weight": 0.0,
        "max_position_count": 8,
    }
    base.update(overrides)
    return {"risk": base}


def _pf_engine(**overrides) -> RiskEngineV21:
    return RiskEngineV21(_pf_limits(**overrides), clock=lambda: AS_OF)


def _pos(
    iid: str,
    weight: float,
    available: datetime | None = AS_OF,
    fresh: bool | None = None,
    sector: str | None = None,
) -> RiskTargetPosition:
    return RiskTargetPosition(
        instrument_id=iid,
        target_weight=weight,
        price_available_at=available,
        is_price_fresh=fresh,
        sector_id=sector,
    )


def _req(positions, **overrides) -> RiskRequest:
    kw = {
        "request_id": "req-1",
        "as_of": AS_OF,
        "nav": 1_000_000.0,
        "cash_weight": 0.0,
        "constraints_version": "limits-v1",
        "proposed_positions": positions,
    }
    kw.update(overrides)
    return RiskRequest(**kw)


def test_portfolio_valid_under_limits_approved() -> None:
    positions = [_pos(f"A{i}", 0.10) for i in range(4)]
    res = _pf_engine().approve_portfolio(_req(positions, cash_weight=0.6))
    assert res.status == RiskStatus.APPROVED
    assert res.executable
    assert res.approved_targets == {f"A{i}": 0.10 for i in range(4)}
    assert sum(res.approved_targets.values()) <= 1.0 + 1e-9


def test_portfolio_single_name_cap_reduced() -> None:
    res = _pf_engine().approve_portfolio(_req([_pos("AAPL", 0.50)]))
    assert res.status == RiskStatus.REDUCED
    assert res.executable
    assert abs(res.approved_targets["AAPL"] - 0.10) < 1e-12
    assert "max_asset_weight_exceeded" in res.reason_codes


def test_portfolio_aggregate_gross_cap_proportional() -> None:
    # 8 names x 0.20 = 1.60 gross -> proportional scale by cap/gross.
    positions = [_pos(f"B{i}", 0.20) for i in range(8)]
    eng = _pf_engine(max_asset_weight=0.15, max_position_count=8)
    res = eng.approve_portfolio(_req(positions))
    assert res.status == RiskStatus.REDUCED
    total = sum(res.approved_targets.values())
    assert abs(total - 1.0) < 1e-9, f"gross na schaal moet 1.0 zijn, kreeg {total}"
    # proportional: every weight scaled by the same factor 1/1.6.
    for w in res.approved_targets.values():
        assert abs(w - 0.125) < 1e-9
    assert "max_gross_exposure_exceeded" in res.reason_codes


def test_portfolio_min_cash_weight_constraint() -> None:
    positions = [_pos(f"C{i}", 0.20) for i in range(8)]
    eng = _pf_engine(min_cash_weight=0.30, max_asset_weight=0.10, max_position_count=8)
    res = eng.approve_portfolio(_req(positions))
    total = sum(res.approved_targets.values())
    assert total <= 1.0 - 0.30 + 1e-9
    assert abs(total - 0.70) < 1e-9
    assert res.executable


def test_portfolio_short_and_margin_rejected() -> None:
    neg = _pf_engine().approve_portfolio(_req([_pos("AAPL", -0.05)]))
    assert neg.status == RiskStatus.REJECTED
    assert "negative_weight_short" in neg.reason_codes

    short = _pf_engine().approve_portfolio(_req([_pos("AAPL", 0.05)], allow_short=True))
    assert short.status == RiskStatus.REJECTED
    assert "unsupported_short_request" in short.reason_codes

    margin = _pf_engine().approve_portfolio(_req([_pos("AAPL", 0.05)], allow_margin=True))
    assert margin.status == RiskStatus.REJECTED
    assert "unsupported_margin_request" in margin.reason_codes


def test_portfolio_zero_leverage_caps_gross_at_nav() -> None:
    # Gross of 2.0 with max_leverage=0 -> gross capped at 1.0 (no leverage).
    positions = [_pos(f"D{i}", 0.25) for i in range(8)]
    eng = _pf_engine(max_asset_weight=0.15, max_position_count=8)
    res = eng.approve_portfolio(_req(positions))
    assert sum(res.approved_targets.values()) <= 1.0 + 1e-9
    assert res.status == RiskStatus.REDUCED


def test_portfolio_max_position_count_rejected() -> None:
    positions = [_pos(f"E{i}", 0.05) for i in range(9)]
    res = _pf_engine(max_position_count=8).approve_portfolio(_req(positions))
    assert res.status == RiskStatus.REJECTED
    assert "max_position_count_exceeded" in res.reason_codes
    assert res.approved_targets == {f"E{i}": 0.0 for i in range(9)}


def test_portfolio_nan_inf_and_negative_nav_rejected() -> None:
    nan = _pf_engine().approve_portfolio(_req([_pos("AAPL", float("nan"))]))
    assert nan.status == RiskStatus.REJECTED
    assert "non_finite_weight" in nan.reason_codes

    inf = _pf_engine().approve_portfolio(_req([_pos("AAPL", float("inf"))]))
    assert inf.status == RiskStatus.REJECTED
    assert "non_finite_weight" in inf.reason_codes

    neg_nav = _pf_engine().approve_portfolio(_req([_pos("AAPL", 0.05)], nav=-1.0))
    assert neg_nav.status == RiskStatus.REJECTED
    assert "invalid_nav" in neg_nav.reason_codes

    weight_above_nav = _pf_engine().approve_portfolio(_req([_pos("AAPL", 1.5)]))
    assert weight_above_nav.status == RiskStatus.REJECTED
    assert "weight_above_nav" in weight_above_nav.reason_codes


def test_portfolio_missing_and_stale_price_paused() -> None:
    missing = _pf_engine().approve_portfolio(_req([_pos("AAPL", 0.05, available=None)]))
    assert missing.status == RiskStatus.PAUSED
    assert "missing_price" in missing.reason_codes

    stale = _pf_engine().approve_portfolio(
        _req([_pos("AAPL", 0.05, available=AS_OF + timedelta(days=1))])
    )
    assert stale.status == RiskStatus.PAUSED
    assert "stale_price" in stale.reason_codes

    explicit_stale = _pf_engine().approve_portfolio(
        _req([_pos("AAPL", 0.05, available=None, fresh=False)])
    )
    assert explicit_stale.status == RiskStatus.PAUSED
    assert "stale_price" in explicit_stale.reason_codes

    # A closed (zero-weight) position needs no fresh price.
    closed_ok = _pf_engine().approve_portfolio(_req([_pos("AAPL", 0.0, available=None)]))
    assert closed_ok.status == RiskStatus.APPROVED


def test_portfolio_missing_hard_config_paused() -> None:
    eng = RiskEngineV21({"risk": {"max_asset_weight": 0.10}}, clock=lambda: AS_OF)
    res = eng.approve_portfolio(_req([_pos("AAPL", 0.05)]))
    assert res.status == RiskStatus.PAUSED
    assert "missing_hard_limit:max_gross_exposure" in res.reason_codes
    assert "missing_hard_limit:max_position_count" in res.reason_codes
    # Never a permissive default: nothing executable.
    assert not res.executable


def test_portfolio_unknown_instrument_rejected() -> None:
    res = _pf_engine().approve_portfolio(
        _req([_pos("ZZZ", 0.05)], universe_ids=["AAPL", "MSFT"])
    )
    assert res.status == RiskStatus.REJECTED
    assert "unknown_instrument" in res.reason_codes


def test_portfolio_deterministic_repeated_result() -> None:
    positions = [_pos(f"F{i}", 0.12) for i in range(5)]
    eng = _pf_engine(max_asset_weight=0.15)
    a = eng.approve_portfolio(_req(positions))
    b = eng.approve_portfolio(_req(positions))
    assert a.model_dump() == b.model_dump()
    assert a.approval_id == b.approval_id
    assert a.expires_at == AS_OF + timedelta(seconds=eng.approval_ttl_seconds)


def test_portfolio_permuted_mapping_yields_equal_approval() -> None:
    positions = [_pos(f"G{i}", 0.20) for i in range(6)]
    permuted = [positions[i] for i in (3, 0, 5, 1, 4, 2)]
    eng = _pf_engine(max_asset_weight=0.15, max_position_count=6)
    a = eng.approve_portfolio(_req(positions))
    b = eng.approve_portfolio(_req(permuted))
    assert a.request_hash == b.request_hash
    assert a.approval_id == b.approval_id
    assert a.approved_targets == b.approved_targets
    assert a.status == b.status == RiskStatus.REDUCED


def test_portfolio_approved_weights_sum_within_all_constraints() -> None:
    # Mixed proposal: over gross, over single name, includes a zero close.
    positions = [
        _pos("AAPL", 0.60),
        _pos("MSFT", 0.30),
        _pos("NVDA", 0.25),
        _pos("JPM", 0.20),
        _pos("SPY", 0.0),
    ]
    eng = _pf_engine(max_asset_weight=0.10, max_position_count=5)
    res = eng.approve_portfolio(_req(positions))
    assert res.status == RiskStatus.REDUCED
    total = sum(res.approved_targets.values())
    peak = max(res.approved_targets.values())
    assert total <= 1.0 + 1e-9
    assert total <= 1.0 - 0.0 + 1e-9  # min_cash_weight = 0.0
    assert peak <= 0.10 + 1e-9
    assert res.approved_targets["SPY"] == 0.0
    assert set(res.approved_targets) == {"AAPL", "MSFT", "NVDA", "JPM", "SPY"}


def test_portfolio_sector_cap_rejects_when_exceeded() -> None:
    # Sector metadata present -> max_sector_weight becomes a required limit.
    missing_sector_cfg = _pf_engine()
    res_missing = missing_sector_cfg.approve_portfolio(
        _req([_pos("AAPL", 0.05, sector="tech"), _pos("MSFT", 0.05, sector="tech")])
    )
    assert res_missing.status == RiskStatus.PAUSED
    assert "missing_hard_limit:max_sector_weight" in res_missing.reason_codes

    eng = _pf_engine(max_sector_weight=0.08, max_asset_weight=0.05, max_position_count=8)
    res = eng.approve_portfolio(
        _req([_pos("AAPL", 0.05, sector="tech"), _pos("MSFT", 0.05, sector="tech")])
    )
    # tech total would be 0.10 > 0.08 sector cap, not fixable by uniform scale->
    assert res.status == RiskStatus.REJECTED
    assert "max_sector_weight_exceeded" in res.reason_codes


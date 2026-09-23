"""Unit tests for Risk Engine v2.1 — two-axis scores, layers, recovery.

Test: Risk Environment Score, Opportunity Score (onafhankelijk van risk),
risk budget matrix, emergency brake, recovery engine, effectieve exposure.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2_1 import EmergencyBrake, RecoveryEngine, RiskEngineV21
from hermes_bot.risk_v2_1.scores import (
    OpportunityScore,
    RiskEnvironmentScore,
    risk_budget_from_scores,
    state_label,
)
from hermes_bot.schemas import Action, ExitReason, RLRawDecision


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
    """regime_sentiment (risico-input) verlaagt de exposure bij crash."""

    def exp_for(regime_sent: float) -> float:
        e = _engine()
        e.peak_equity = 100.0
        rng = np.random.default_rng(2)
        e.hist_returns = list(rng.normal(0.0005, 0.006, 40))
        e.prev_exposure = 0.5
        pf = PortfolioState(cash=100000)
        # Loop door zodat _smooth zijn clamp kan oplossen.
        for i in range(5):
            exp, bd = e.approve(_decision(), pf, 100000 * (1 + 0.001 * i),
                                alpha_signals={"regime_sentiment": regime_sent,
                                               "regional_score": 0.5})
        return exp

    crash_exp = exp_for(-0.8)
    bull_exp = exp_for(0.5)
    assert crash_exp < bull_exp, f"crash exposure {crash_exp} moet < bull {bull_exp} zijn"


def test_regional_score_adds_opportunity() -> None:
    """regional_score als alpha-signaal verhoogt de opp-score monotoon."""
    opp = OpportunityScore({"risk": {}})
    returns = [0.002] * 20
    lo, bd_lo = opp.score(returns, vol=0.10, vol_baseline=0.125, drawdown=0.0,
                          corr=0.1, alpha_signals={"regional_score": 0.1})
    hi, bd_hi = opp.score(returns, vol=0.10, vol_baseline=0.125, drawdown=0.0,
                          corr=0.1, alpha_signals={"regional_score": 0.9})
    assert hi > lo, "hogere regionale score moet meer opportunity geven"
    assert "regional_score" in bd_hi["alpha_signals"]

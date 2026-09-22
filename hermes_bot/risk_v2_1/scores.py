"""Risk Engine v2.1 — twee onafhankelijke scores (Risk Environment + Opportunity).

De centrale innovatie van v2.1: twee onafhankelijke assen.

Axis 1 — Risk Environment Score (0-100):
    0 = extreem gezond / laag risico
    100 = extreem gevaarlijk
    Inputs: realized vol, vol-regime, vol-acceleratie, drawdown, correlatie,
            trend-deterioratie, extreme returns, VaR, CVaR, MC tail probability.

Axis 2 — Opportunity Score (0-100):
    0 = weinig reden om risico te nemen
    100 = sterke condities om risico te nemen
    Inputs: positieve trend, momentum, vol-normalisatie, marktherstel,
            stabiele correlaties, stabiel regime.
    Signaal-opportunity (alpha): prediction confidence, model agreement,
            fundamental quality, valuation, bottleneck score, news/speech.
            Deze bestaan nog niet -> neutrale waarde + expliciet 'unavailable'.

BELANGRIJK: opportunity is NIET simpelweg (100 - risk). Beide worden
onafhankelijk berekend. Dit test de hypothese dat de engine niet alleen moet
weten "hoe gevaarlijk is het", maar ook "hoeveel kans is er om risico te nemen".
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# AXIS 1 — RISK ENVIRONMENT SCORE (0-100, hoger = gevaarlijker)
# ---------------------------------------------------------------------------
class RiskEnvironmentScore:
    """Berekent de risico-score uit marktcondities (alleen historische info)."""

    def __init__(self, config: dict) -> None:
        c = config.get("risk", {})
        self.vol_target = c.get("vol_target", 0.125)
        self.max_drawdown = c.get("max_portfolio_drawdown", -0.08)
        self.corr_threshold = c.get("corr_threshold", 0.6)
        self.vol_lookback = c.get("vol_lookback", 20)
        self.corr_lookback = c.get("corr_lookback", 20)

    def score(
        self,
        vol: float,
        vol_baseline: float,
        drawdown: float,
        corr: float,
        var_95: float,
        crash_prob: float,
        extreme_return: float,
    ) -> float:
        """Combineer risico-inputs tot een 0-100 score.

        Elke component draagt 0-100 bij; het gewogen gemiddelde is de score.
        """
        # Vol-component: vol t.o.v. target. 0 bij lage vol, 100 bij 2x target.
        vol_comp = 0.0
        if vol > 0:
            vol_comp = float(np.clip((vol / max(0.05, self.vol_target) - 0.5) / 1.5, 0, 1) * 100)

        # Drawdown-component: 0 bij geen dd, 100 bij max_drawdown of dieper.
        dd_comp = 0.0
        if drawdown < 0:
            dd_comp = float(np.clip(abs(drawdown) / abs(self.max_drawdown), 0, 1) * 100)

        # Correlatie-component: 0 bij lage corr, 100 bij corr_threshold+.
        corr_comp = 0.0
        if corr > 0:
            corr_comp = float(np.clip(corr / self.corr_threshold, 0, 1) * 100)

        # Tail-risk component: VaR en crash-kans.
        tail_comp = 0.0
        if var_95 < 0:
            tail_comp += float(np.clip(abs(var_95) / 0.10, 0, 1) * 50)
        tail_comp += float(np.clip(crash_prob / 0.30, 0, 1) * 50)

        # Extreme-return component: 0 bij normaal, 100 bij -5% dag.
        extreme_comp = float(np.clip(abs(extreme_return) / 0.05, 0, 1) * 100)

        # Gewogen gemiddelde (vol en dd wegen zwaarst).
        score = (
            0.30 * vol_comp
            + 0.30 * dd_comp
            + 0.15 * corr_comp
            + 0.15 * tail_comp
            + 0.10 * extreme_comp
        )
        return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# AXIS 2 — OPPORTUNITY SCORE (0-100, hoger = meer kans om risico te nemen)
# ---------------------------------------------------------------------------
class OpportunityScore:
    """Berekent de opportunity-score uit marktcondities + alpha-signalen.

    Opportunity is NIET (100 - risk). Het meet of er reden is om risico te
    nemen: positieve trend, momentum, vol-normalisatie, herstel, stabiele
    correlaties. Alpha-signalen zijn interfaces voor toekomstige upstream
    informatie; als ze niet bestaan -> neutrale waarde + 'unavailable'.
    """

    def __init__(self, config: dict) -> None:
        c = config.get("risk", {})
        self.vol_target = c.get("vol_target", 0.125)
        self.momentum_lookback = c.get("momentum_lookback", 20)
        self.recovery_lookback = c.get("recovery_lookback", 10)

    def score(
        self,
        returns: list[float],
        vol: float,
        vol_baseline: float,
        drawdown: float,
        corr: float,
        alpha_signals: dict | None = None,
    ) -> tuple[float, dict]:
        """Combineer opportunity-inputs tot een 0-100 score.

        Returns: (score, breakdown) — breakdown toont elke component + welke
        alpha-signalen beschikbaar zijn.
        """
        alpha_signals = alpha_signals or {}

        # Trend/momentum-component: positieve recente returns = opportunity.
        momentum_comp = 0.0
        if len(returns) >= 5:
            recent = np.asarray(returns[-5:])
            mom = float(np.mean(recent))
            # 0 bij negatief/neutraal, 100 bij +0.5% dag-gemiddelde.
            momentum_comp = float(np.clip(mom / 0.005, 0, 1) * 100)

        # Vol-normalisatie-component: vol terug naar target = opportunity.
        vol_norm_comp = 0.0
        if vol > 0:
            # Hoe dichter vol bij target, hoe hoger de opportunity.
            ratio = vol / max(0.05, self.vol_target)
            vol_norm_comp = float(np.clip(1.0 - abs(ratio - 1.0) / 1.0, 0, 1) * 100)

        # Herstel-component: drawdown verbetert = opportunity.
        recovery_comp = 0.0
        if drawdown > -0.05:
            recovery_comp = float(np.clip((drawdown + 0.05) / 0.05, 0, 1) * 100)

        # Correlatie-stabiliteit: lage corr = meer opportunity.
        corr_comp = 0.0
        if corr <= 0.3:
            corr_comp = 100.0
        elif corr < 0.6:
            corr_comp = 50.0

        # Alpha-signalen (interfaces, neutraal als niet beschikbaar).
        alpha_comp = 0.0
        alpha_breakdown = {}
        for key in ["prediction_confidence", "model_agreement", "fundamental_quality",
                    "valuation", "bottleneck_score", "news_signal"]:
            if key in alpha_signals:
                v = float(alpha_signals[key])
                alpha_comp += float(np.clip(v, 0, 1)) * 100 / 6.0
                alpha_breakdown[key] = round(v, 3)
            else:
                alpha_breakdown[key] = "unavailable"

        # Gewogen gemiddelde (momentum en vol-normalisatie wegen zwaarst).
        score = (
            0.35 * momentum_comp
            + 0.25 * vol_norm_comp
            + 0.20 * recovery_comp
            + 0.10 * corr_comp
            + 0.10 * alpha_comp
        )
        breakdown = {
            "momentum": round(momentum_comp, 1),
            "vol_normalization": round(vol_norm_comp, 1),
            "recovery": round(recovery_comp, 1),
            "correlation_stability": round(corr_comp, 1),
            "alpha": round(alpha_comp, 1),
            "alpha_signals": alpha_breakdown,
        }
        return float(np.clip(score, 0, 100)), breakdown


# ---------------------------------------------------------------------------
# TWEEDIMENSIONALE RISK-BUDGET FUNCTIE
# ---------------------------------------------------------------------------
def risk_budget_from_scores(risk_score: float, opp_score: float) -> float:
    """Zet (risk, opportunity) om in een continu risk budget [0,1].

    Matrix:
        High risk + low opp  -> DEFENSIVE (laag budget)
        High risk + high opp -> SELECTIVE (gematigd)
        Low risk + low opp   -> NEUTRAL (gematigd)
        Low risk + high opp  -> AGGRESSIVE (hoog budget)

    Formule: budget = (1 - risk/100) * (0.5 + 0.5 * opp/100)
    - Bij risk=0, opp=100 -> 1.0 (aggressief)
    - Bij risk=100, opp=0 -> 0.0 (defensief)
    - Bij risk=0, opp=0   -> 0.5 (neutraal)
    - Bij risk=100, opp=100 -> 0.0 (selectief, maar risk domineert)
    """
    risk_norm = float(np.clip(risk_score / 100.0, 0, 1))
    opp_norm = float(np.clip(opp_score / 100.0, 0, 1))
    budget = (1.0 - risk_norm) * (0.5 + 0.5 * opp_norm)
    return float(np.clip(budget, 0, 1))


def state_label(risk_score: float, opp_score: float) -> str:
    """Beschrijvende staat (niet de sizing-mechanisme)."""
    risk_high = risk_score > 50
    opp_high = opp_score > 50
    if risk_high and not opp_high:
        return "DEFENSIVE"
    if risk_high and opp_high:
        return "SELECTIVE"
    if not risk_high and opp_high:
        return "AGGRESSIVE"
    return "NEUTRAL"

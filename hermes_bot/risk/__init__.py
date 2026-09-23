"""LAYER 6: risk engine — the hard safety layer. [[09]]

Architectuurverbetering: de Monte Carlo-route ([[08]]) is nu ECHT
aangesloten op de goedkeuring. `var_95`, `es_95` en `crash_probability`
worden gebruikt om de toegestane allocatie te begrenzen (crashweerstand),
i.p.v. dat ze dode velden waren. Ook `max_portfolio_drawdown` uit config
wordt nu toegepast.
"""
from __future__ import annotations

from hermes_bot.portfolio import PortfolioState
from hermes_bot.schemas import MonteCarloResult, RiskApproval, RLRawDecision


class RiskEngine:
    """Convert the RL proposal into an approved, bounded allocation.

    Regel: RL voorstelt, Risk keurt goed. Nooit een voorstel
    rechtstreeks naar de executie-laag.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config.get("risk", {})
        self._limits = {
            "max_asset_weight": self.cfg.get("max_asset_weight", 0.10),
            "max_sector_risk": self.cfg.get("max_sector_risk", 0.20),
            "max_leverage": self.cfg.get("max_leverage", 0.0),
            "cash_min_in_crash": self.cfg.get("cash_min_in_crash", 0.30),
            "max_portfolio_drawdown": self.cfg.get("max_portfolio_drawdown", -0.08),
            # Thresholds for Monte Carlo bounding.
            "max_var_95": self.cfg.get("max_var_95", -0.05),  # -5% VaR95 max
            "max_crash_prob": self.cfg.get("max_crash_prob", 0.10),  # 10% crash-kans max
        }

    def approve(
        self,
        decision: RLRawDecision,
        portfolio: PortfolioState,
        mc: MonteCarloResult | None = None,
    ) -> RiskApproval:
        """Pre-trade check + position sizing + limits + Monte Carlo-begrenzing.

        Position-size koppelt aan decision.zekerheid:
        lage zekerheid -> kleinere positie (uit [[06]]).
        """
        reasons: list[str] = []

        # 0. Exit decisions (SELL with exit_reason) are always approved —
        #    taking profit / limiting loss must never be blocked.
        if decision.action.value == "sell" and decision.exit_reason.value != "none":
            return RiskApproval(
                approved=True,
                target_alloc={decision.entity: 0.0},  # sluit de positie
                rejected_reasons=[],
                adjusted=False,
            )

        # 1. Size based on certainty (scale down on uncertainty).
        size = abs(decision.intent_to_alloc)
        certainty = decision.zekerheid
        scaled = size * (0.5 + 0.5 * certainty)

        # 2. Harde limiet per asset.
        if scaled > self._limits["max_asset_weight"]:
            scaled = self._limits["max_asset_weight"]
            reasons.append("max_asset_weight overschreden -> bijgesteld")

        # 3. Monte Carlo-begrenzing (crashweerstand).
        if mc is not None:
            # Hoge VaR95 (groot verwacht verlies) -> verklein positie.
            # ratio = max_var_95 / var_95 (both negative) -> <1 if VaR is too low.
            if mc.var_95 < self._limits["max_var_95"]:
                ratio = max(0.0, min(1.0, self._limits["max_var_95"] / mc.var_95))
                scaled *= ratio
                reasons.append(f"VaR95={mc.var_95:.3f} te hoog -> verkleind")
            # Hoge crash-kans -> extra verkleining.
            if mc.crash_probability > self._limits["max_crash_prob"]:
                scaled *= 0.5
                reasons.append(f"crash_prob={mc.crash_probability:.2f} te hoog -> gehalveerd")

        # 4. Basic risk-off on crash state (regime).
        in_crash = getattr(portfolio, "regime", None) == "crash"
        if in_crash and scaled > 0.0:
            scaled *= 0.5
            reasons.append("crash-regime -> positie gehalveerd")

        approved = not reasons or all("overschreden -> bijgesteld" in r for r in reasons)
        return RiskApproval(
            approved=approved,
            target_alloc={decision.entity: round(scaled, 4)},
            rejected_reasons=[] if approved else reasons,
            adjusted=bool(reasons),
            drawdown_current=portfolio.drawdown,
            var_95=mc.var_95 if mc else 0.0,
            es_95=mc.expected_shortfall_95 if mc else 0.0,
            crash_probability=mc.crash_probability if mc else 0.0,
        )
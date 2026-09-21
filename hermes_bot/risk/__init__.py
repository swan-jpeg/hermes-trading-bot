"""LAAG 6: risico-engine — de harde veiligheidslaag. [[09]]"""
from __future__ import annotations

from hermes_bot.portfolio import PortfolioState
from hermes_bot.schemas import RiskApproval, RLRawDecision


class RiskEngine:
    """Zet RL-voorstel om in goedgekeurde, begrensde allocatie.

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
        }

    def approve(self, decision: RLRawDecision, portfolio: PortfolioState) -> RiskApproval:
        """Pre-trade check + position sizing + limits.

        Position-size koppelt aan decision.zekerheid:
        lage zekerheid -> kleinere positie (uit [[06]]).
        """
        reasons: list[str] = []

        # 1. Grootte o.b.v. zekerheid (scale-down bij onzekerheid).
        size = abs(decision.intent_to_alloc)
        certainty = decision.zekerheid
        scaled = size * (0.5 + 0.5 * certainty)

        # 2. Harde limiet per asset.
        if scaled > self._limits["max_asset_weight"]:
            scaled = self._limits["max_asset_weight"]
            reasons.append("max_asset_weight overschreden -> bijgesteld")

        # Basis risk-off bij crash-toestand (regime zou dit vullen).
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
        )

"""Multi-asset portfolio-constructie. [[11]]"""
from __future__ import annotations

from dataclasses import dataclass, field

from hermes_bot.schemas import Position


@dataclass
class PortfolioState:
    """Huidige portefeuille: cash + open posities + regime.

    Posities dragen entry-prijs zodat de beslissingslaag winst kan nemen
    en verlies kan beperken ([[07]]/[[09]]).
    """

    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)  # entity -> positie
    regime: str = "unknown"
    # Highest price since entry per entity (for trailing stop).
    peaks: dict[str, float] = field(default_factory=dict)

    def total_exposure(self) -> float:
        return sum(p.qty for p in self.positions.values())

    def unrealized_pnl_pct(self, entity: str, current_price: float) -> float:
        pos = self.positions.get(entity)
        if pos is None:
            return 0.0
        return pos.unrealized_pnl_pct(current_price)

    def peak_for(self, entity: str, current_price: float) -> float:
        """Updated high since entry (for trailing stop)."""
        return max(self.peaks.get(entity, current_price), current_price)

    @property
    def drawdown(self) -> float:
        # TODO: from history/equity curve. Deliberately 0.0 while there is no history.
        return 0.0


class PortfolioAllocator:
    """Strategic + tactical allocation across asset classes."""

    def __init__(self, config: dict) -> None:
        self.cfg = config

    def propose_target(self, decisions: list, state: PortfolioState) -> dict[str, float]:
        """Combine approved allocations into portfolio targets.

        - BUY  -> verhoog gewicht met intent_to_alloc.
        - SELL -> verlaag gewicht (of sluit als intent <= -huidig gewicht).
        - HOLD -> behoud huidig gewicht.
        """
        target: dict[str, float] = {}
        for d in decisions:
            current = state.positions.get(d.entity)
            cur_w = current.qty if current else 0.0
            if d.action.value == "buy":
                target[d.entity] = round(cur_w + abs(d.intent_to_alloc), 4)
            elif d.action.value == "sell":
                target[d.entity] = round(max(0.0, cur_w - abs(d.intent_to_alloc)), 4)
            else:  # hold / hedge / other
                target[d.entity] = round(cur_w, 4)
        return target

    def rebalance(self, current: PortfolioState, target: dict[str, float]) -> list[dict]:
        """Return the list of desired net changes (input for execution)."""
        delta: list[dict] = []
        for asset, tw in target.items():
            cw = current.positions.get(asset).qty if current.positions.get(asset) else 0.0
            if abs(tw - cw) > 0.001:
                delta.append({"asset": asset, "change": round(tw - cw, 4)})
        return delta
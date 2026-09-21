"""Multi-asset portfolio-constructie. [[11]]"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioState:
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)  # asset -> gewicht
    regime: str = "unknown"

    def total_exposure(self) -> float:
        return sum(self.positions.values())

    @property
    def drawdown(self) -> float:
        # TODO: uit historie/equity-curve.
        return 0.0


class PortfolioAllocator:
    """Strategische + tactische toewijzing over assetklassen."""

    def __init__(self, config: dict) -> None:
        self.cfg = config

    def propose_target(self, decisions: list, state: PortfolioState) -> dict[str, float]:
        """Combineer goedgekeurde allocaties + sleeves tot portfolio-targets."""
        raise NotImplementedError

    def rebalance(self, current: PortfolioState, target: dict[str, float]) -> list[dict]:
        """Return lijst van gewenste nettowoorderingen (input voor executie)."""
        delta: list[dict] = []
        for asset, tw in target.items():
            cw = current.positions.get(asset, 0.0)
            if abs(tw - cw) > 0.001:
                delta.append({"asset": asset, "change": round(tw - cw, 4)})
        return delta

"""LAAG 5: fusiemodel — combineert heterogene signalen. [[06]]"""
from __future__ import annotations

from hermes_bot.schemas import FusionSignal


class BaseFusion:
    """Interface voor fusie-aanpakken."""

    def fuse(self, inputs: list[object]) -> FusionSignal:
        raise NotImplementedError


class WeightedFusion(BaseFusion):
    """Gewogen aggregatie met bronweging + onzekerheid als variance."""

    def __init__(self, source_weights: dict[str, float] | None = None) -> None:
        self.weights = source_weights or {
            "audiovisual": 0.2,
            "agent": 0.4,
            "regional": 0.15,
            "market": 0.25,
        }

    def fuse(self, inputs: list[object]) -> FusionSignal:
        """TODO: normaliseer per-bron scores, weeg samen, bereken kwaliteit/zekerheid.
        Zekerheid = 1 - genormaliseerde variantie (daalt in chaos).
        """
        raise NotImplementedError


def build_fusion(config: dict) -> BaseFusion:
    """Fabriek: kies fusie-aanpak uit config."""
    return WeightedFusion()

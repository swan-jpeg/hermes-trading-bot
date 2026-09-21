"""LAAG 5: fusiemodel — combineert heterogene signalen. [[06]]

Implementatie: genormaliseerde gewogen aggregatie met onzekerheid
(= variantie over bronnen). Zekerheid daalt in chaos.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from hermes_bot.schemas import FusionSignal


class BaseFusion:
    """Interface voor fusie-aanpakken."""

    def fuse(self, inputs: list[dict]) -> FusionSignal:
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

    def fuse(self, inputs: list[dict]) -> FusionSignal:
        """inputs: lijst van {source, sentiment, confidence, entity_id}.

        sentiment: -1..1 richting. confidence: 0..1.
        Kwaliteit = gewogen gemiddelde confidence.
        Zekerheid = 1 - genormaliseerde variantie (daalt bij meningsverschil).
        Emotie = gewogen sentiment per bron.
        """
        if not inputs:
            raise ValueError("geen inputs voor fusie")

        entity = inputs[0]["entity_id"]
        weights = self.weights
        total_w = sum(weights.get(i["source"], 0.1) * i.get("confidence", 0.5) for i in inputs)
        if total_w <= 0:
            total_w = 1.0

        # Gewogen sentiment (emotie) en kwaliteit.
        sentiment = 0.0
        kvaliteit = 0.0
        for i in inputs:
            w = weights.get(i["source"], 0.1) * i.get("confidence", 0.5)
            sentiment += w * i.get("sentiment", 0.0)
            kvaliteit += w * i.get("confidence", 0.5)
        sentiment /= total_w
        kvaliteit /= total_w

        # Zekerheid = 1 - variantie van bron-sentimenten (genormaliseerd).
        sentiments = [i.get("sentiment", 0.0) for i in inputs]
        var = float(np.var(sentiments)) if len(sentiments) > 1 else 0.0
        zekerheid = float(np.clip(1.0 - var, 0.0, 1.0))

        return FusionSignal(
            entity_id=entity,
            tijdstip=datetime.now(UTC),
            kwaliteit=round(kvaliteit, 4),
            zekerheid=round(zekerheid, 4),
            emotie={"sentiment": round(sentiment, 4), "angst": round(max(0.0, -sentiment), 4)},
            source_breakdown={i["source"]: round(i.get("sentiment", 0.0), 4) for i in inputs},
        )


def build_fusion(config: dict) -> BaseFusion:
    """Fabriek: kies fusie-aanpak uit config."""
    return WeightedFusion()

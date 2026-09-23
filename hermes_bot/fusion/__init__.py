"""LAAG 5: fusiemodel — combineert heterogene signalen. [[06]]

Implementatie: genormaliseerde gewogen aggregatie met onzekerheid
(= variantie over bronnen). Zekerheid daalt in chaos.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from hermes_bot.schemas import FusionSignal


class BaseFusion:
    """Interface for fusion approaches."""

    def fuse(self, inputs: list[dict]) -> FusionSignal:
        raise NotImplementedError


class WeightedFusion(BaseFusion):
    """Weighted aggregation with source weighting + uncertainty as variance."""

    def __init__(self, source_weights: dict[str, float] | None = None) -> None:
        # Weighting per source. Web/bottleneck are now explicit sources.
        self.weights = source_weights or {
            "audiovisual": 0.15,
            "speech": 0.15,      # speeches van CEO's/landsleiders (webscraping)
            "report": 0.15,      # quarterly reports, overheidsuitgaven
            "alert": 0.10,       # nieuws-alerts, point-loops
            "regional": 0.10,    # regionale scores
            "bottleneck": 0.10,  # B2B-vraag / supply-chain knelpunten
            "agent": 0.15,       # fundamentele AI-agent
            "market": 0.10,      # koersen
        }

    def fuse(self, inputs: list[dict]) -> FusionSignal:
        """inputs: list of {source, sentiment, confidence, entity_id}.

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

        # Weighted sentiment (emotion) and quality.
        sentiment = 0.0
        kvaliteit = 0.0
        for i in inputs:
            w = weights.get(i["source"], 0.1) * i.get("confidence", 0.5)
            sentiment += w * i.get("sentiment", 0.0)
            kvaliteit += w * i.get("confidence", 0.5)
        sentiment /= total_w
        kvaliteit /= total_w

        # Certainty = 1 - variance of source sentiments (normalized).
        sentiments = [i.get("sentiment", 0.0) for i in inputs]
        var = float(np.var(sentiments)) if len(sentiments) > 1 else 0.0
        zekerheid = float(np.clip(1.0 - var, 0.0, 1.0))

        # Per-bron gewogen gemiddelde sentiment (i.p.v. alleen laatste input).
        source_sent: dict[str, float] = {}
        source_w: dict[str, float] = {}
        for i in inputs:
            s = i["source"]
            w = weights.get(s, 0.1) * i.get("confidence", 0.5)
            source_sent[s] = source_sent.get(s, 0.0) + w * i.get("sentiment", 0.0)
            source_w[s] = source_w.get(s, 0.0) + w
        source_breakdown = {
            s: round(source_sent[s] / source_w[s], 4) if source_w[s] else 0.0
            for s in source_sent
        }

        return FusionSignal(
            entity_id=entity,
            tijdstip=datetime.now(UTC),
            kwaliteit=round(kvaliteit, 4),
            zekerheid=round(zekerheid, 4),
            emotie={"sentiment": round(sentiment, 4), "angst": round(max(0.0, -sentiment), 4)},
            source_breakdown=source_breakdown,
        )


def build_fusion(config: dict) -> BaseFusion:
    """Factory: choose the fusion approach from config."""
    return WeightedFusion()

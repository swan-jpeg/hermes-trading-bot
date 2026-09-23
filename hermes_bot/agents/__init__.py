"""LAAG 4: fundamentele AI-agent. [[05]]

Implementatie: regelsgebaseerde fundamentele scores + optionele LLM-synthese.
"""
from __future__ import annotations

from datetime import UTC, datetime

from hermes_bot.schemas import EntityAnalysis


class FundamentalAgent:
    """Combineert structuurdata + alternatieve data + nieuws per entiteit."""

    def __init__(self, llm: object | None = None) -> None:
        self.llm = llm

    def analyze(self, entity_id: str, features: dict, news: list[dict]) -> EntityAnalysis:
        """Bouw per-entiteit analyse.
        features: fundamentale waarden (pe, ev_ebitda, growth, margin, fcf).
        news: gescored berichten [{sentiment, credibility}].
        """
        fundamentals = self._score_fundamentals(features)
        sentiment = self._score_news(news)
        regions = features.get("regions_exposure", {})
        thesis = self._build_thesis(entity_id, fundamentals, sentiment, regions)

        return EntityAnalysis(
            entity_id=entity_id,
            time=datetime.now(UTC),
            fundamentals=fundamentals,
            sentiment_score=round(sentiment, 4),
            regions_exposure=regions,
            thesis=thesis,
            confidence=round(0.5 + 0.5 * min(1.0, len(news) / 10), 4),
            n_sources=len(news),
        )

    def _score_fundamentals(self, features: dict) -> dict[str, float]:
        """Waarde/kwaliteit/groei/marges → subscores 0..1 (hoger = beter)."""
        pe = features.get("pe", 20.0)
        growth = features.get("growth", 0.0)
        margin = features.get("margin", 0.0)
        fcf = features.get("fcf_yield", 0.0)

        value = float(max(0.0, min(1.0, 25.0 / max(pe, 1.0))))  # lage PE = beter
        quality = float(max(0.0, min(1.0, margin / 0.3)))  # hoge marge = beter
        growth_s = float(max(0.0, min(1.0, growth / 0.3)))
        fcf_s = float(max(0.0, min(1.0, fcf / 0.1)))
        return {
            "value": round(value, 4),
            "quality": round(quality, 4),
            "growth": round(growth_s, 4),
            "margin": round(margin, 4),
            "fcf_yield": round(fcf_s, 4),
        }

    def _score_news(self, news: list[dict]) -> float:
        """Weighted sentiment over messages (credibility-weighted)."""
        if not news:
            return 0.0
        total = sum(n.get("credibility", 0.5) for n in news)
        if total <= 0:
            return 0.0
        return sum(n.get("sentiment", 0.0) * n.get("credibility", 0.5) for n in news) / total

    def _build_thesis(
        self, entity_id: str, fundamentals: dict, sentiment: float, regions: dict
    ) -> str:
        score = (
            0.4 * fundamentals["value"]
            + 0.3 * fundamentals["growth"]
            + 0.3 * fundamentals["quality"]
        )
        direction = "positief" if score > 0.5 else "neutraal/negatief"
        return (
            f"{entity_id}: fundamentele score {score:.2f} ({direction}), "
            f"nieuws-sentiment {sentiment:.2f}, regio's {list(regions.keys()) or 'n.v.t.'}"
        )

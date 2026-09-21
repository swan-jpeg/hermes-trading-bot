"""LAAG 4: fundamentele AI-agent. [[05]]"""
from __future__ import annotations

from hermes_bot.schemas import EntityAnalysis


class FundamentalAgent:
    """Combineert structuurdata + alternatieve data + nieuws per entiteit."""

    def __init__(self, llm: object | None = None) -> None:
        # LLM optioneel; scores kunnen ook regelsgebaseerd.
        self.llm = llm

    def analyze(self, entity_id: str, features: dict, news: list[dict]) -> EntityAnalysis:
        """Bouw per-entiteit analyse.
        features: fundamentale waarden. news: gescored berichten.
        """
        raise NotImplementedError

    def _score_fundamentals(self, features: dict) -> dict[str, float]:
        """Waarde/kwaliteit/groei/marges → subscores."""
        raise NotImplementedError

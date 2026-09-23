"""Tests for the Impact Agent (links events to affected instruments)."""
from __future__ import annotations

from hermes_bot.impact import ImpactAgent


def test_speech_world_leader_matches_macro() -> None:
    """A speech from a world leader impacts macro instruments (SPY/QQQ/AGG)."""
    agent = ImpactAgent()
    r = agent.analyze("", source="speech", entity_id="trump")
    assert r.impacted
    entities = {i["entity"] for i in r.impacted}
    assert "SPY" in entities
    assert "AGG" in entities


def test_semiconductor_keyword_matches_chips() -> None:
    """A report mentioning 'semiconductor' impacts chip stocks."""
    agent = ImpactAgent()
    r = agent.analyze("Nvidia reports record semiconductor demand for AI chips",
                      source="report")
    entities = {i["entity"] for i in r.impacted}
    assert "NVDA" in entities
    assert "AMD" in entities


def test_government_spending_matches_bonds() -> None:
    """Government spending impacts bond instruments."""
    agent = ImpactAgent()
    r = agent.analyze("New government infrastructure spending announced",
                      source="report")
    entities = {i["entity"] for i in r.impacted}
    assert "AGG" in entities
    assert "TLT" in entities


def test_no_match_returns_empty() -> None:
    """An unrelated event returns no impacted instruments."""
    agent = ImpactAgent()
    r = agent.analyze("The weather is nice today", source="alert")
    assert r.impacted == []


def test_to_fusion_inputs_per_entity() -> None:
    """Impact results become per-entity fusion inputs."""
    agent = ImpactAgent()
    r = agent.analyze("AI chip demand rising", source="report")
    inputs = agent.to_fusion_inputs(r, sentiment=0.5)
    assert inputs
    assert all(i["source"] == "impact" for i in inputs)
    assert all(i["entity_id"] for i in inputs)
    assert all(i["sentiment"] == 0.5 for i in inputs)


def test_deduplicates_entities() -> None:
    """The same entity matched by multiple sectors appears once."""
    agent = ImpactAgent()
    r = agent.analyze("AI chips and electric vehicle batteries", source="report")
    entities = [i["entity"] for i in r.impacted]
    assert len(entities) == len(set(entities))  # no duplicates


def test_to_fusion_inputs_keeps_per_entity_llm_sentiment() -> None:
    """The LLM's per-instrument sentiment (e.g. NVDA +0.8) must survive."""
    from hermes_bot.impact import LLMImpactAgent
    agent = LLMImpactAgent()
    # Keyword-match only -> no LLM call (no key), so we simulate a surplus
    # result where one entity carries its own sentiment (as the LLM fills in).
    r = agent.analyze("AI chip demand rising", source="report")
    # Give one entity a specific sentiment, like the LLM does.
    for item in r.impacted:
        if item["entity"] == "NVDA":
            item["sentiment"] = 0.8
    inputs = agent.to_fusion_inputs(r, sentiment=0.5)  # global fallback 0.5
    nvda = [i for i in inputs if i["entity_id"] == "NVDA"]
    assert nvda and nvda[0]["sentiment"] == 0.8, "NVDA per-entity sentiment verloren"
    # Andere (zonder eigen sentiment) krijgen de global 0.5.
    other = [i for i in inputs if i["entity_id"] != "NVDA"]
    if other:
        assert all(i["sentiment"] == 0.5 for i in other)

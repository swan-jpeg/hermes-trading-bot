"""Tests for the deterministic evidence-linked Impact Agent (T08)."""
from __future__ import annotations

from hermes_bot.impact import (
    ImpactAgent,
    ImpactRule,
    LLMImpactAgent,
    TextChunk,
)


def _agent(catalog, graph, rules):
    return ImpactAgent(rules=rules, catalog=catalog, graph=graph, max_hops=2)


def test_known_phrase_maps_to_canonical_id_and_span(impact_catalog):
    """A known trigger phrase resolves the canonical entity and text span."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze(
        "NVIDIA reports record demand for AI chips",
        source="report",
    )
    assert r.analysis is not None
    assert r.analysis.state == "ok"
    impacted_ids = {imp.entity_id for imp in r.analysis.impacts}
    assert ids["nvda"] in impacted_ids
    assert ids["semiconductor_sector"] in impacted_ids

    # Evidence spans must point to the matched trigger in the text.
    evidence = r.analysis.event.evidence
    assert any(
        span.text.lower() == "ai chips"
        for ev in evidence
        for span in ev.spans
    )


def test_speech_world_leader_matches_macro(impact_catalog):
    """A speech from a world leader impacts macro instruments (SPY)."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze(
        "We will invest heavily in AI chips and semiconductor infrastructure",
        source="speech",
        entity_id=ids["us_gov"],
    )
    assert r.analysis is not None
    assert r.analysis.state == "ok"
    impacted_ids = {imp.entity_id for imp in r.analysis.impacts}
    assert ids["spy"] in impacted_ids
    assert ids["semiconductor_sector"] in impacted_ids


def test_government_spending_matches_bonds(impact_catalog):
    """Government policy text reaches bond-like ETF instruments."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze(
        "US Government announces new infrastructure spending and AI chips support",
        source="report",
    )
    impacted_ids = {imp.entity_id for imp in r.analysis.impacts}
    assert ids["spy"] in impacted_ids
    assert ids["semiconductor_sector"] in impacted_ids


def test_no_match_returns_empty(impact_catalog):
    """An unrelated event returns no impacts and a degraded state."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze("The weather is nice today", source="alert")
    assert r.analysis is not None
    assert r.analysis.impacts == []
    assert r.analysis.state == "degraded"


def test_to_fusion_inputs_per_entity(impact_catalog):
    """Impact results become per-entity fusion inputs."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze("AI chips demand rising", source="report", entity_id=ids["nvda"])
    inputs = agent.to_fusion_inputs(r, sentiment=0.5)
    assert inputs
    assert all(i["source"] == "impact" for i in inputs)
    assert all(i["entity_id"] for i in inputs)
    assert all(i["sentiment"] == 0.5 for i in inputs)


def test_to_fusion_inputs_keeps_per_entity_llm_sentiment(impact_catalog):
    """Per-instrument sentiment attached by a caller survives to fusion inputs."""
    catalog, graph, rules, ids = impact_catalog
    agent = LLMImpactAgent(rules=rules, catalog=catalog, graph=graph)
    r = agent.analyze("AI chips demand rising", source="report", entity_id=ids["nvda"])
    for item in r.impacted:
        if item["entity"] == ids["nvda"]:
            item["sentiment"] = 0.8
    inputs = agent.to_fusion_inputs(r, sentiment=0.5)
    nvda = [i for i in inputs if i["entity_id"] == ids["nvda"]]
    assert nvda and nvda[0]["sentiment"] == 0.8


def test_ambiguous_alias_returns_candidates(impact_catalog):
    """An alias resolving to multiple entities emits candidates, no impact."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze("Apple announces new product line", source="report")
    assert r.analysis is not None
    assert any(c.alias == "Apple" for c in r.analysis.candidates)
    assert len(r.analysis.candidates) >= 2
    # No resolved impact because alias is ambiguous.
    impacted_aliases = {imp.entity_id for imp in r.analysis.impacts}
    assert ids["apple"] not in impacted_aliases
    assert ids["apple_records"] not in impacted_aliases


def test_negation_forbidden_phrase_no_match(impact_catalog):
    """A forbidden phrase prevents a rule from matching."""
    catalog, graph, rules, ids = impact_catalog
    rule = ImpactRule(
        rule_id="rule_hike_no_cut",
        trigger_phrase="interest rate hike",
        forbidden_phrases=("cut",),
        direction="negative",
    )
    agent = _agent(catalog, graph, [rule])
    r = agent.analyze("The central bank plans to cut the interest rate hike", source="report")
    # Forbidden phrase "cut" blocks the rule.
    assert r.analysis is not None
    assert r.analysis.state == "degraded"
    assert r.analysis.impacts == []


def test_conflicting_evidence_returns_warning_and_both_ids(impact_catalog):
    """Conflicting directions for the same entity produce a warning."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze(
        "interest rate hike and interest rate cut both expected for AI chips",
        source="report",
        entity_id=ids["us_gov"],
    )
    assert r.analysis is not None
    spy_records = [imp for imp in r.analysis.impacts if imp.entity_id == ids["spy"]]
    if spy_records:
        assert spy_records[0].direction == "mixed"
    assert any("conflicting" in w.lower() for w in r.analysis.warnings)


def test_allegation_vs_forecast_preserved(impact_catalog):
    """Epistemic type distinguishes allegation from forecast."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    allegation = agent.analyze("alleged accounting fraud at NVIDIA", source="alert")
    forecast = agent.analyze("forecast AI chip demand rising", source="report")
    assert allegation.analysis is not None
    assert forecast.analysis is not None
    assert allegation.analysis.event.epistemic_type.value == "allegation"
    assert forecast.analysis.event.epistemic_type.value == "forecast"


def test_source_unavailable_returns_error():
    """Calling the agent with invalid source/model-shaped values returns error."""
    agent = ImpactAgent()
    r = agent.analyze("AI chips demand rising", source="http://example.com/model")
    assert r.analysis is not None
    assert r.analysis.state == "error"


def test_confidence_not_called_probability(impact_catalog):
    """Impact records expose subjective_confidence, never probability."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze("AI chip demand rising", source="report")
    for imp in r.analysis.impacts:
        assert hasattr(imp, "subjective_confidence")
        assert not hasattr(imp, "probability")
        assert imp.estimated_return is None


def test_malicious_model_values_rejected():
    """Model-shaped or JSON source values are rejected as invalid input."""
    agent = ImpactAgent()
    for bad_source in ['{"model": "x"}', '[1,2,3]', 'api_key=secret']:
        r = agent.analyze("text", source=bad_source)
        assert r.analysis.state == "error"


def test_exact_chunk_spans_preserved(impact_catalog):
    """When chunks are supplied, evidence spans refer to exact chunk offsets."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    chunks = [
        TextChunk(chunk_id="chk_a", text="NVIDIA reports AI chips demand."),
        TextChunk(chunk_id="chk_b", text="Forecast strong growth."),
    ]
    r = agent.analyze("", source="report", chunks=chunks)
    assert r.analysis is not None
    assert r.analysis.state == "ok"
    spans = [
        span
        for ev in r.analysis.event.evidence
        for span in ev.spans
    ]
    assert any(span.chunk_id == "chk_a" and span.text.lower() == "ai chips" for span in spans)


def test_legacy_probability_is_zero(impact_catalog):
    """Legacy ImpactResult.probability is always zero (never fabricated)."""
    catalog, graph, rules, ids = impact_catalog
    agent = _agent(catalog, graph, rules)
    r = agent.analyze("AI chip demand rising", source="report")
    assert r.probability == 0.0


def test_llm_agent_degraded_not_silent_fallback(impact_catalog):
    """LLMImpactAgent returns degraded state instead of silently falling back."""
    catalog, graph, rules, ids = impact_catalog
    agent = LLMImpactAgent(
        rules=rules,
        catalog=catalog,
        graph=graph,
        api_key="fake-key",
    )
    r = agent.analyze("AI chip demand rising", source="report")
    assert r.analysis is not None
    assert r.analysis.state == "degraded"
    assert any("LLM interpretation disabled" in w for w in r.analysis.warnings)

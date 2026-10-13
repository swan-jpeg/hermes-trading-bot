"""Tests for multi-entity impact-agent output running separately through RL+risk."""
from __future__ import annotations

import numpy as np

from hermes_bot.impact import ImpactAgent
from hermes_bot.pipeline import Pipeline


def _stub_closes(entity: str) -> np.ndarray:
    """Offline deterministic price series for any entity (no network)."""
    rng = np.random.default_rng(abs(hash(entity)) % 2**31)
    return np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100


def _make_pipe(impact_agent):
    return Pipeline({"closes_provider": _stub_closes, "impact_agent": impact_agent})


def test_impact_agent_multi_entity(impact_catalog):
    """One event can hit many instruments, each with its own evidence."""
    catalog, graph, rules, ids = impact_catalog
    agent = ImpactAgent(rules=rules, catalog=catalog, graph=graph, max_hops=2)
    r = agent.analyze(
        "The European Union announces a 50 billion euro investment plan in AI chips"
        " and semiconductors",
        source="government",
        entity_id=ids["eu"],
    )
    # Deterministic extraction hits the semiconductor sector and related entities.
    assert len(r.analysis.impacts) >= 2
    impacted_ids = {imp.entity_id for imp in r.analysis.impacts}
    assert ids["semiconductor_sector"] in impacted_ids
    assert ids["nvda"] in impacted_ids


def test_pipeline_builds_per_entity_contexts(impact_catalog):
    """The pipeline builds one AssetContext per affected entity."""
    catalog, graph, rules, ids = impact_catalog
    agent = ImpactAgent(rules=rules, catalog=catalog, graph=graph, max_hops=2)
    pipe = _make_pipe(agent)
    web_inputs = [{
        "source": "government",
        "entity_id": ids["eu"],
        "body": "The European Union announces a 50 billion euro investment plan"
        " in AI chips and semiconductors",
        "sentiment": 0.5,
    }]
    result = pipe.run(price=100.0, entity_id=ids["eu"], web_inputs=web_inputs)
    # One AssetContext per impacted entity.
    assert len(result.asset_contexts) == len(result.impacted)
    assert len(result.asset_contexts) >= 2
    # Category 8 indicators are populated (65 obs keys) + entity + asset_class.
    for c in result.asset_contexts:
        assert len(c) == 67  # 65 obs (cat 8 erbij) + entity + asset_class
        assert c["entity"]
        for k in ("rsi_14", "ema_cross", "momentum_10", "vol_20", "trend_strength"):
            assert k in c
            assert 0.0 <= float(c[k]) <= 1.0


def test_pipeline_per_entity_decisions(impact_catalog):
    """Each affected entity gets its own decision + risk approval."""
    catalog, graph, rules, ids = impact_catalog
    agent = ImpactAgent(rules=rules, catalog=catalog, graph=graph, max_hops=2)
    pipe = _make_pipe(agent)
    web_inputs = [{
        "source": "government",
        "entity_id": ids["eu"],
        "body": "The European Union announces a 50 billion euro investment plan"
        " in AI chips and semiconductors",
        "sentiment": 0.5,
    }]
    result = pipe.run(price=100.0, entity_id=ids["eu"], web_inputs=web_inputs)
    # One decision + exposure per entity.
    assert len(result.decisions) == len(result.impacted)
    assert len(result.exposures) == len(result.impacted)
    # Each decision names its own entity.
    entities = {d["entity"] for d in result.decisions}
    assert entities == {c["entity"] for c in result.asset_contexts}

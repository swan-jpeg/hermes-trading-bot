"""Tests for multi-entity impact-agent output running separately through RL+risk."""
from __future__ import annotations

import numpy as np

from hermes_bot.impact import ImpactAgent
from hermes_bot.pipeline import Pipeline


def _stub_closes(entity: str) -> np.ndarray:
    """Offline deterministic price series for any entity (no network)."""
    rng = np.random.default_rng(abs(hash(entity)) % 2**31)
    return np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100


def _make_pipe() -> Pipeline:
    return Pipeline({"closes_provider": _stub_closes})


def test_impact_agent_multi_entity() -> None:
    """One event can hit many instruments, each with its own sentiment."""
    agent = ImpactAgent()
    r = agent.analyze(
        "The European Union announces a 50 billion euro investment plan in AI chips"
        " and semiconductors",
        source="government", entity_id="eu")
    # Keyword matching hits multiple sectors (tech + semiconductor + macro).
    assert len(r.impacted) >= 5
    entities = {i["entity"] for i in r.impacted}
    assert "NVDA" in entities  # semiconductor
    assert "MSFT" in entities  # tech


def test_pipeline_builds_per_entity_contexts() -> None:
    """The pipeline builds one AssetContext per affected entity."""
    pipe = _make_pipe()
    web_inputs = [{
        "source": "government",
        "entity_id": "eu",
        "body": "The European Union announces a 50 billion euro investment plan"
        " in AI chips and semiconductors",
        "sentiment": 0.5,
    }]
    result = pipe.run(price=100.0, entity_id="eu", web_inputs=web_inputs)
    # One AssetContext per impacted entity.
    assert len(result.asset_contexts) == len(result.impacted)
    assert len(result.asset_contexts) >= 5
    # Category 8 indicators are populated (65 obs keys) + entity + asset_class.
    for c in result.asset_contexts:
        assert len(c) == 67  # 65 obs (cat 8 erbij) + entity + asset_class
        assert c["entity"]
        for k in ("rsi_14", "ema_cross", "momentum_10", "vol_20", "trend_strength"):
            assert k in c
            assert 0.0 <= float(c[k]) <= 1.0


def test_pipeline_per_entity_decisions() -> None:
    """Each affected entity gets its own decision + risk approval."""
    pipe = _make_pipe()
    web_inputs = [{
        "source": "government",
        "entity_id": "eu",
        "body": "The European Union announces a 50 billion euro investment plan"
        " in AI chips and semiconductors",
        "sentiment": 0.5,
    }]
    result = pipe.run(price=100.0, entity_id="eu", web_inputs=web_inputs)
    # One decision + exposure per entity.
    assert len(result.decisions) == len(result.impacted)
    assert len(result.exposures) == len(result.impacted)
    # Each decision names its own entity.
    entities = {d["entity"] for d in result.decisions}
    assert entities == {c["entity"] for c in result.asset_contexts}

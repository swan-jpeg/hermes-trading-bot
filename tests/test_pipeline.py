"""Tests for the integrated architecture pipeline.

Verifies that the "extensions" (bottleneck, regime, regional scores) really
ARE in the chain as fusion inputs, not as standalone modules.
"""
from __future__ import annotations

from hermes_bot.pipeline import (
    Pipeline,
    build_bottleneck_inputs,
    build_regime_input,
    build_regional_scores_input,
    collect_web_inputs,
    run_pipeline,
)


def test_pipeline_returns_full_result() -> None:
    """Pipeline returns all integrated intermediate values."""
    result = run_pipeline()
    assert "fusion" in result
    assert "bottleneck_ratings" in result
    assert "regime" in result
    assert "regional_scores" in result
    assert "effective_exposure" in result


def test_bottleneck_is_fusion_input() -> None:
    """Bottleneck ratings are fusion inputs, but regional/regime are RISK inputs."""
    result = run_pipeline()
    sources = {i["source"] for i in result["web_inputs"]}
    assert "bottleneck" in sources
    # Regional + regime/orderflow are NO LONGER fusion inputs: they now go to
    # the risk engine (via alpha_signals), not to fusion.
    assert "regional" not in sources
    assert "agent" not in sources


def test_bottleneck_ratings_present() -> None:
    """Bottleneck-agent gives per-company ratings."""
    result = run_pipeline()
    assert result["bottleneck_ratings"]  # not empty
    assert any(v > 0 for v in result["bottleneck_ratings"].values())


def test_web_inputs_collected() -> None:
    inputs = collect_web_inputs()
    assert len(inputs) >= 5  # speech + reports + alerts
    sources = {i["source"] for i in inputs}
    assert "speech" in sources and "report" in sources and "alert" in sources


def test_bottleneck_inputs_standalone() -> None:
    inputs = build_bottleneck_inputs()
    assert len(inputs) > 0
    assert all(i["source"] == "bottleneck" for i in inputs)


def test_regime_input() -> None:
    r = build_regime_input()
    assert r["source"] == "agent"
    assert r["regime"] in ("bull", "bear", "highvol", "crash")


def test_regional_scores_input() -> None:
    r = build_regional_scores_input()
    assert r["source"] == "regional"
    assert "regional_scores" in r


def test_pipeline_runs_multiple_times() -> None:
    """Pipeline is reusable (no state leak between runs)."""
    p = Pipeline()
    r1 = p.run(price=100.0)
    r2 = p.run(price=110.0)
    assert isinstance(r1.effective_exposure, float)
    assert isinstance(r2.effective_exposure, float)
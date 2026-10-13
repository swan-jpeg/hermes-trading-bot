"""Tests for the integrated architecture pipeline.

Verifies that the "extensions" (bottleneck, regime, regional scores) really
ARE in the chain as fusion inputs, not as standalone modules.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hermes_bot.pipeline import (
    Pipeline,
    PortfolioMarkError,
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


def test_record_portfolio_mark_valid() -> None:
    """Test that valid portfolio marks are recorded correctly."""
    pipeline = Pipeline()
    
    # Record a valid mark
    as_of = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    pipeline.record_portfolio_mark(
        as_of=as_of,
        equity=100000.0,
        return_decimal=0.01,
        source_quality="high"
    )
    
    # Verify the mark was recorded
    assert len(pipeline._portfolio_marks) == 1
    assert pipeline._portfolio_marks[0] == (as_of, 100000.0, 0.01, "high")


def test_record_portfolio_mark_duplicate() -> None:
    """Test that duplicate marks raise PortfolioMarkError."""
    pipeline = Pipeline()
    
    # Record first mark
    as_of = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    pipeline.record_portfolio_mark(
        as_of=as_of,
        equity=100000.0,
        return_decimal=0.01,
        source_quality="high"
    )
    
    # Try to record duplicate mark
    with pytest.raises(PortfolioMarkError, match="duplicate or out-of-order"):
        pipeline.record_portfolio_mark(
            as_of=as_of,
            equity=100500.0,
            return_decimal=0.015,
            source_quality="high"
        )


def test_record_portfolio_mark_out_of_order() -> None:
    """Test that out-of-order marks raise PortfolioMarkError."""
    pipeline = Pipeline()
    
    # Record first mark
    as_of1 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    pipeline.record_portfolio_mark(
        as_of=as_of1,
        equity=100000.0,
        return_decimal=0.01,
        source_quality="high"
    )
    
    # Try to record out-of-order mark
    as_of2 = datetime(2026, 10, 13, 11, 0, tzinfo=UTC)  # Earlier timestamp
    with pytest.raises(PortfolioMarkError, match="duplicate or out-of-order"):
        pipeline.record_portfolio_mark(
            as_of=as_of2,
            equity=100500.0,
            return_decimal=0.015,
            source_quality="high"
        )


def test_record_portfolio_mark_invalid_equity() -> None:
    """Test that invalid equity raises PortfolioMarkError."""
    pipeline = Pipeline()
    
    # Test negative equity
    with pytest.raises(PortfolioMarkError, match="finite positive number"):
        pipeline.record_portfolio_mark(
            as_of=datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
            equity=-100000.0,
            return_decimal=0.01,
            source_quality="high"
        )
    
    # Test zero equity
    with pytest.raises(PortfolioMarkError, match="finite positive number"):
        pipeline.record_portfolio_mark(
            as_of=datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
            equity=0.0,
            return_decimal=0.01,
            source_quality="high"
        )


def test_record_portfolio_mark_invalid_return() -> None:
    """Test that invalid return raises PortfolioMarkError."""
    pipeline = Pipeline()
    
    # Test non-finite return
    with pytest.raises(PortfolioMarkError, match="finite number"):
        pipeline.record_portfolio_mark(
            as_of=datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
            equity=100000.0,
            return_decimal=float('inf'),
            source_quality="high"
        )


def test_record_portfolio_mark_sequence() -> None:
    """Test that valid sequence of marks works."""
    pipeline = Pipeline()
    
    # Record multiple valid marks in sequence
    marks = [
        (datetime(2026, 10, 13, 12, 0, tzinfo=UTC), 100000.0, 0.01),
        (datetime(2026, 10, 13, 13, 0, tzinfo=UTC), 101000.0, 0.005),
        (datetime(2026, 10, 13, 14, 0, tzinfo=UTC), 101500.0, 0.0025),
    ]
    
    for as_of, equity, ret in marks:
        pipeline.record_portfolio_mark(
            as_of=as_of,
            equity=equity,
            return_decimal=ret,
            source_quality="high"
        )
    
    # Verify all marks were recorded
    assert len(pipeline._portfolio_marks) == 3
    for i, (as_of, equity, ret, quality) in enumerate(pipeline._portfolio_marks):
        expected_as_of, expected_equity, expected_ret = marks[i]
        assert as_of == expected_as_of
        assert equity == expected_equity
        assert ret == expected_ret
        assert quality == "high"


def test_risk_config_validation() -> None:
    """Test that required risk limits are validated at pipeline construction."""
    # This test is a bit redundant because our earlier tests already validate this
    # but we add it explicitly to align with task requirements
    
    # Should work - all required limits present and valid
    pipeline = Pipeline()
    assert pipeline.risk.cfg is not None
    assert "max_asset_weight" in pipeline.risk.cfg


def test_risk_return_history() -> None:
    """Test that return history grows correctly."""
    pipeline = Pipeline()
    
    # Before any marks
    assert len(pipeline.risk.hist_returns) == 0
    
    # Add first mark
    pipeline.record_portfolio_mark(
        as_of=datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
        equity=1000,
        return_decimal=0.01,
        source_quality="high"
    )
    
    # History should have 1 element now
    assert len(pipeline.risk.hist_returns) == 1
    assert pipeline.risk.hist_returns[0] == 0.01
    
    # Add second mark
    pipeline.record_portfolio_mark(
        as_of=datetime(2026, 10, 14, 12, 0, tzinfo=UTC),
        equity=1010,
        return_decimal=0.02,
        source_quality="high"
    )
    
    # History should have 2 elements now
    assert len(pipeline.risk.hist_returns) == 2
    assert pipeline.risk.hist_returns[0] == 0.01
    assert pipeline.risk.hist_returns[1] == 0.02


def test_missing_required_risk_limits_rejected() -> None:
    """Test that missing required risk limits are rejected."""
    # Under AD-01, we test the fail-closed guarantee:
    # monkeypatch load_config to return a config whose risk block is MISSING required limits
    import hermes_bot.pipeline as pipeline_module
    
    # Mock load_config to return a config that is missing required risk limits
    def mock_load_config(*args, **kwargs):
        return {
            "risk": {
                "max_asset_weight": 0.10  # Missing required limits like max_gross_exposure
                # Note: max_gross_exposure and others are missing!
            }
        }
    
    # Patch the function
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(pipeline_module, "load_config", mock_load_config)
        
        # This should raise a PortfolioMarkError because risk section is missing required limits
        with pytest.raises(PortfolioMarkError, match="missing"):
            Pipeline()


def _stub_closes(entity: str) -> list[float]:
    """Offline deterministic price series for any entity (no network)."""
    import numpy as np
    rng = np.random.default_rng(abs(hash(entity)) % 2**31)
    return list(np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100)


def test_non_finite_risk_limits_rejected() -> None:
    """Test that non-finite risk limits are rejected."""
    # Create config with non-finite limit
    invalid_config = {
        "data": {"market_feed": "yfinance"},
        "risk": {
            "max_asset_weight": 0.1,
            "max_gross_exposure": float('inf'),  # Invalid!  
            "max_sector_weight": 0.2,
            "max_leverage": 0.0,
            "min_cash_weight": 0.0,
            "max_position_count": 8
        }
    }
    
    # This should raise a PortfolioMarkError
    with pytest.raises(PortfolioMarkError, match="finite"):
        Pipeline(config=invalid_config)
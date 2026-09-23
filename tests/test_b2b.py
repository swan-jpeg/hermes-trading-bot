"""Tests for B2B bottleneck + regional scores (T8)."""
from __future__ import annotations

from hermes_bot.expansions import BottleneckAnalyzer, RegionalScorer


def test_bottleneck_analyzer_ingest() -> None:
    """Test dat BottleneckAnalyzer ingest werkt."""
    analyzer = BottleneckAnalyzer()
    
    # Test ingest
    scores = analyzer.ingest([])
    
    # Controleer resultaat
    assert isinstance(scores, dict)


def test_regional_scorer_score() -> None:
    """Test dat RegionalScorer score_from_reports werkt."""
    scorer = RegionalScorer()
    
    # Test score_from_reports
    scores = scorer.score_from_reports([])
    
    # Controleer resultaat
    assert isinstance(scores, dict)


def test_bottleneck_analyzer_basic() -> None:
    """Test that BottleneckAnalyzer works without errors."""
    analyzer = BottleneckAnalyzer()
    
    # Test basic functionality
    assert hasattr(analyzer, 'add_node')
    assert hasattr(analyzer, 'add_edge')
    assert hasattr(analyzer, 'bottleneck_score')
    assert hasattr(analyzer, 'top_bottlenecks')
    assert hasattr(analyzer, 'ingest')


def test_regional_scorer_basic() -> None:
    """Test that RegionalScorer works without errors."""
    scorer = RegionalScorer()
    
    # Test basic functionality
    assert hasattr(scorer, 'score_from_reports')
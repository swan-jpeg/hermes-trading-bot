"""Tests for NLP sentiment (T2).

Deze tests mocken de zware FinBERT-pipeline zodat de suite stabiel draait
zonder torch/transformers te importeren (die op sommige CPU's een harde
SIGILL-crash geven die niet met try/except te vangen is). De
lexicon-fallback wordt direct getest.
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_bot.signals.sentiment import SentimentAnalyzer


class _FailingModel:
    """Mock that raises an error -> triggers the lexicon fallback."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("model niet beschikbaar")


def test_sentiment_analyzer_basic() -> None:
    """Test that the analyzer returns a list of AlertSignals (lexicon)."""
    analyzer = SentimentAnalyzer()
    # Force fallback: set a model that fails, so transformers is never
    # imported (SIGILL-safe).
    analyzer._finbert_model = _FailingModel()

    texts = [
        "De markt is goed gegroeid vandaag.",
        "Er is sprake van een negatieve trend.",
    ]

    results = analyzer.analyze(texts)

    assert len(results) == len(texts)
    for result in results:
        assert hasattr(result, "sentiment")
        assert hasattr(result, "confidence")
        assert hasattr(result, "credibility")
        assert hasattr(result, "source")
        assert hasattr(result, "source_name")
        assert hasattr(result, "entity_id")


def test_sentiment_analyzer_fallback() -> None:
    """Test that the analyzer uses the lexicon fallback."""
    analyzer = SentimentAnalyzer()
    analyzer._finbert_model = _FailingModel()

    texts = [
        "De markt is goed en positief.",
        "Er is sprake van een slechte trend en verlies.",
    ]

    results = analyzer.analyze(texts)

    assert len(results) == len(texts)
    for result in results:
        assert -1.0 <= result.sentiment <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.credibility <= 1.0


def test_sentiment_uses_finbert_when_available() -> None:
    """Test that FinBERT is used when the pipeline is available (mocked)."""
    analyzer = SentimentAnalyzer()

    def fake_pipeline(text):
        return [{"label": "POSITIVE", "score": 0.9}]

    with patch.object(analyzer, "_finbert_model", fake_pipeline):
        results = analyzer.analyze(["De markt stijgt."])

    assert len(results) == 1
    assert results[0].sentiment > 0.0  # POSITIVE -> positief
    assert results[0].confidence == 0.9

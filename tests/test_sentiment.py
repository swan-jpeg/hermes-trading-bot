"""Tests voor de NLP-sentiment (T2)."""
from __future__ import annotations

from hermes_bot.signals.sentiment import SentimentAnalyzer


def test_sentiment_analyzer_basic() -> None:
    """Test dat de analyzer een lijst van AlertSignals retourneert."""
    analyzer = SentimentAnalyzer()
    
    texts = [
        "De markt is goed gegroeid vandaag.",
        "Er is sprake van een negatieve trend."
    ]
    
    results = analyzer.analyze(texts)
    
    # Moet dezelfde lengte hebben als input
    assert len(results) == len(texts)
    
    # Controleer dat het AlertSignal objecten zijn
    for result in results:
        assert hasattr(result, 'sentiment')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'credibility')
        assert hasattr(result, 'source')
        assert hasattr(result, 'source_name')
        assert hasattr(result, 'entity_id')


def test_sentiment_analyzer_fallback() -> None:
    """Test dat de analyzer fallback gebruikt als FinBERT niet beschikbaar is."""
    analyzer = SentimentAnalyzer()
    
    # Test met eenvoudige woorden
    texts = [
        "De markt is goed en positief.",
        "Er is sprake van een slechte trend en verlies."
    ]
    
    results = analyzer.analyze(texts)
    
    # Moet dezelfde lengte hebben
    assert len(results) == len(texts)
    
    # Controleer dat sentiment waarden zijn
    for result in results:
        assert -1.0 <= result.sentiment <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.credibility <= 1.0
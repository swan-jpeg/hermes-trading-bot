"""LAYER 3: sentiment analysis via FinBERT or lexicon fallback."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from hermes_bot.schemas import AlertSignal, SourceKind

if TYPE_CHECKING:
    pass


class SentimentAnalyzer:
    """Analyses news/reports into a sentiment score with confidence."""

    def __init__(self) -> None:
        self._finbert_model = None
        self._finbert_tokenizer = None
        self._lexicon = self._load_lexicon()

    def _load_lexicon(self) -> dict[str, float]:
        """Load a simple lexicon for offline fallback."""
        # Simple word sentiments (for offline fallback)
        return {
            "good": 0.5,
            "bad": -0.5,
            "excellent": 0.8,
            "terrible": -0.8,
            "positive": 0.6,
            "negative": -0.6,
            "strong": 0.4,
            "weak": -0.4,
            "up": 0.3,
            "down": -0.3,
            "rise": 0.4,
            "fall": -0.4,
            "profit": 0.5,
            "loss": -0.5,
            "increase": 0.4,
            "decrease": -0.4,
            "growth": 0.6,
            "decline": -0.6,
            "stable": 0.0,
            "volatile": -0.3,
            "uncertain": -0.2,
            "optimistic": 0.5,
            "pessimistic": -0.5,
        }

    def analyze(self, texts: list[str]) -> list[AlertSignal]:
        """Analyse a list of texts into sentiment scores.

        Gebruikt FinBERT indien beschikbaar, anders lexicon fallback.
        """
        results: list[AlertSignal] = []
        
        for text in texts:
            # Probeer FinBERT eerst
            try:
                if self._finbert_model is None:
                    from transformers import pipeline
                    self._finbert_model = pipeline(
                        "sentiment-analysis",
                        model="ProsusAI/finbert",
                        device=-1,  # CPU; GPU via device=0 als torch beschikbaar is
                    )
                
                # Analyse with FinBERT
                result = self._finbert_model(text)
                label = result[0]['label']
                sentiment = result[0]['score'] if label == 'POSITIVE' else -result[0]['score']
                confidence = result[0]['score']
            except Exception:
                # Fall back to lexicon if FinBERT is not available
                sentiment, confidence = self._analyze_with_lexicon(text)
            
            # Create an AlertSignal
            signal = AlertSignal(
                source=SourceKind.ALERT,
                source_name="sentiment",
                entity_id="general",
                timestamp=datetime.now(),
                confidence=confidence,
                kind="sentiment",
                headline=text[:100],  # Kortere titel
                body=text,
                sentiment=sentiment,
                credibility=0.7,  # Standaard bronweging
            )
            results.append(signal)
        
        return results

    def _analyze_with_lexicon(self, text: str) -> tuple[float, float]:
        """Analyse with lexicon fallback."""
        words = text.lower().split()
        scores = []
        
        for word in words:
            # Verwijder leestekens
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in self._lexicon:
                scores.append(self._lexicon[clean_word])
        
        if not scores:
            return 0.0, 0.0  # Neutraal als geen woorden gevonden
        
        avg_score = sum(scores) / len(scores)
        # Confidence based on word count and the number of positive/negative words
        confidence = min(len(scores) / 10.0, 1.0)  # Max 1.0 confidence
        
        return avg_score, confidence
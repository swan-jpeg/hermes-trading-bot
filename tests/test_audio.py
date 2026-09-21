"""Tests voor audio-analyse (T3)."""
from __future__ import annotations

from hermes_bot.signals.audio import AudioAnalyzer
from hermes_bot.signals.audiovisual import SpeechSignal


def test_audio_analyzer_basic() -> None:
    """Test dat AudioAnalyzer werkt zonder fouten."""
    analyzer = AudioAnalyzer()
    
    # Test met een niet-bestaand bestand
    signal = analyzer.analyze("/non/existent/audio.wav")
    
    # Moet een SpeechSignal teruggeven
    assert isinstance(signal, SpeechSignal)
    assert signal.confidence == 0.0  # Neutrale output


def test_audio_analyzer_structure() -> None:
    """Test dat de structuur van SpeechSignal correct is."""
    analyzer = AudioAnalyzer()
    
    # Test met een niet-bestaand bestand
    signal = analyzer.analyze("/non/existent/audio.wav")
    
    # Controleer velden
    assert hasattr(signal, 'transcript')
    assert hasattr(signal, 'emotion_scores')
    assert hasattr(signal, 'prosody')
    assert hasattr(signal, 'pauses')
    assert hasattr(signal, 'video_features')
    assert hasattr(signal, 'audience')
    assert signal.source.value == "speech"
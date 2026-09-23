"""Tests for audio analysis (T3)."""
from __future__ import annotations

from hermes_bot.signals.audio import AudioAnalyzer
from hermes_bot.signals.audiovisual import SpeechSignal


def test_audio_analyzer_basic() -> None:
    """Test that AudioAnalyzer works without errors."""
    analyzer = AudioAnalyzer()
    
    # Test with a non-existent file
    signal = analyzer.analyze("/non/existent/audio.wav")
    
    # Must return a SpeechSignal
    assert isinstance(signal, SpeechSignal)
    assert signal.confidence == 0.0  # Neutrale output


def test_audio_analyzer_structure() -> None:
    """Test that the SpeechSignal structure is correct."""
    analyzer = AudioAnalyzer()
    
    # Test with a non-existent file
    signal = analyzer.analyze("/non/existent/audio.wav")
    
    # Controleer velden
    assert hasattr(signal, 'transcript')
    assert hasattr(signal, 'emotion_scores')
    assert hasattr(signal, 'prosody')
    assert hasattr(signal, 'pauses')
    assert hasattr(signal, 'video_features')
    assert hasattr(signal, 'audience')
    assert signal.source.value == "speech"
"""Tests voor video-analyse (T4)."""
from __future__ import annotations

from hermes_bot.signals.video import VideoAnalyzer


def test_video_analyzer_basic() -> None:
    """Test dat VideoAnalyzer werkt zonder fouten."""
    analyzer = VideoAnalyzer()
    
    # Test met een niet-bestaand bestand
    result = analyzer.analyze("/non/existent/video.mp4")
    
    # Moet een dict retourneren, zelfs als het niet bestaat
    assert isinstance(result, dict)
    assert "facial_expression" in result
    assert "gesture" in result
    assert "body_language" in result
    assert "audience" in result
    assert "confidence" in result


def test_video_analyzer_offline_safe() -> None:
    """Test dat de analyzer offline veilig is."""
    analyzer = VideoAnalyzer()
    
    # Test dat zelfs zonder echte video het geen fout geeft
    result = analyzer.analyze("/non/existent/video.mp4")
    
    # Controleer dat de resultaten neutraal zijn
    assert result["confidence"] == 0.0
    assert result["facial_expression"] == {}
    assert result["gesture"] == {}
    assert result["body_language"] == {}
    assert result["audience"] == {}
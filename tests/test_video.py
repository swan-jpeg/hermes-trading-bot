"""Tests for video analysis (T4)."""
from __future__ import annotations

from hermes_bot.signals.video import VideoAnalyzer


def test_video_analyzer_basic() -> None:
    """Test that VideoAnalyzer works without errors."""
    analyzer = VideoAnalyzer()
    
    # Test with a non-existent file
    result = analyzer.analyze("/non/existent/video.mp4")
    
    # Must return a dict, even if it does not exist
    assert isinstance(result, dict)
    assert "facial_expression" in result
    assert "gesture" in result
    assert "body_language" in result
    assert "audience" in result
    assert "confidence" in result


def test_video_analyzer_offline_safe() -> None:
    """Test that the analyzer is offline-safe."""
    analyzer = VideoAnalyzer()
    
    # Test that even without a real video it does not error
    result = analyzer.analyze("/non/existent/video.mp4")
    
    # Check that the results are neutral
    assert result["confidence"] == 0.0
    assert result["facial_expression"] == {}
    assert result["gesture"] == {}
    assert result["body_language"] == {}
    assert result["audience"] == {}
"""LAAG 3: multimodale signaalverwerking (speech/audio/video)."""
from __future__ import annotations

from hermes_bot.schemas import BaseSignal, SourceKind


class SpeechSignal(BaseSignal):
    """Structural signal from audio/video analysis. [[03]]"""

    source: SourceKind = SourceKind.SPEECH
    transcript: str = ""
    # audio
    emotion_scores: dict[str, float] = {"confident": 0.0, "anxious": 0.0, "optimistic": 0.0}
    prosody: dict[str, float] = {"pace": 0.0, "pitch_var": 0.0, "volume": 0.0}
    pauses: float = 0.0
    # video
    video_features: dict[str, float] = {
        "facial_expression": 0.0,
        "gesture": 0.0,
        "body_language": 0.0,
    }
    audience: dict[str, float] = {"laughter": 0.0, "applause": 0.0}

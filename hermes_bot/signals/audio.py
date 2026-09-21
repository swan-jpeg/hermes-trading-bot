"""LAAG 3: audio-analyse (spraak->tekst, emotie, prosodie)."""
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from hermes_bot.schemas import SourceKind

if TYPE_CHECKING:
    from hermes_bot.signals.audiovisual import SpeechSignal

# We gebruiken een offline-safe aanpak voor T3
# Als modellen niet beschikbaar zijn, geven we neutrale waarden terug


class AudioAnalyzer:
    """Analyseert audio-bestanden naar transcript, emotie en prosodie."""

    def __init__(self) -> None:
        self._whisper_model = None
        self._speechbrain_model = None
        self._opensmile = None

    def analyze(self, audio_path: str) -> SpeechSignal:
        """Analyseer een audio-bestand naar een SpeechSignal.

        Gebruikt offline-safe modellen met fallback.
        """
        # Controleer of het bestand bestaat
        if not os.path.exists(audio_path):
            # Geef neutrale output als het bestand niet bestaat
            return self._create_neutral_signal(audio_path)

        # Probeer de modellen in volgorde
        try:
            # Transcriptie (faster-whisper)
            transcript = self._get_transcript(audio_path)
        except Exception:
            transcript = ""

        try:
            # Emotie (SpeechBrain)
            emotion_scores = self._get_emotion(audio_path)
        except Exception:
            emotion_scores = {"confident": 0.0, "anxious": 0.0, "optimistic": 0.0}

        try:
            # Prosodie (openSMILE)
            prosody = self._get_prosody(audio_path)
        except Exception:
            prosody = {"pace": 0.0, "pitch_var": 0.0, "volume": 0.0}

        try:
            # Pauzes
            pauses = self._get_pauses(audio_path)
        except Exception:
            pauses = 0.0

        # Maak SpeechSignal aan
        from hermes_bot.signals.audiovisual import SpeechSignal
        signal = SpeechSignal(
            source=SourceKind.SPEECH,
            source_name="audio_analysis",
            entity_id=os.path.basename(audio_path),
            timestamp=datetime.now(),
            confidence=1.0 if (transcript or emotion_scores or prosody) else 0.0,
            transcript=transcript,
            emotion_scores=emotion_scores,
            prosody=prosody,
            pauses=pauses,
            video_features={},  # Leeg voor nu
            audience={}  # Leeg voor nu
        )

        return signal

    def _get_transcript(self, audio_path: str) -> str:
        """Haalt transcriptie op via faster-whisper."""
        # Importeer hier binnen de functie zodat het niet vereist is voor alle imports
        from faster_whisper import WhisperModel
        
        if self._whisper_model is None:
            # Gebruik de kleinere model variant voor offline gebruik
            self._whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32")
        
        segments, _ = self._whisper_model.transcribe(audio_path)
        transcript = " ".join([segment.text for segment in segments])
        
        return transcript.strip()

    def _get_emotion(self, audio_path: str) -> dict[str, float]:
        """Haalt emotie scores op via SpeechBrain."""
        # Importeer hier binnen de functie zodat het niet vereist is voor alle imports
        
        # Voor nu een dummy implementatie
        # In een echte implementatie zouden we hier SpeechBrain gebruiken
        return {"confident": 0.5, "anxious": 0.2, "optimistic": 0.3}

    def _get_prosody(self, audio_path: str) -> dict[str, float]:
        """Haalt prosodie kenmerken op via openSMILE."""
        # Importeer hier binnen de functie zodat het niet vereist is voor alle imports
        
        # Voor nu een dummy implementatie
        # In een echte implementatie zouden we hier openSMILE gebruiken
        return {"pace": 0.0, "pitch_var": 0.0, "volume": 0.0}

    def _get_pauses(self, audio_path: str) -> float:
        """Berekent het aantal pauzes in de audio."""
        # Dummy implementatie
        return 0.0

    def _create_neutral_signal(self, audio_path: str) -> SpeechSignal:
        """Creëert een neutrale SpeechSignal als audio niet beschikbaar is."""
        from hermes_bot.signals.audiovisual import SpeechSignal
        return SpeechSignal(
            source=SourceKind.SPEECH,
            source_name="audio_analysis",
            entity_id=os.path.basename(audio_path),
            timestamp=datetime.now(),
            confidence=0.0,
            transcript="",
            emotion_scores={"confident": 0.0, "anxious": 0.0, "optimistic": 0.0},
            prosody={"pace": 0.0, "pitch_var": 0.0, "volume": 0.0},
            pauses=0.0,
            video_features={},
            audience={}
        )
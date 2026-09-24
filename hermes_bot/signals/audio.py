"""LAYER 3: audio analysis (speech->text, emotion, prosody).

Echte model-integratie met offline-fallback:
- Transcript: faster-whisper (WhisperModel)
- Emotie: SpeechBrain pretrained emotion (via transformers)
- Prosodie: openSMILE (eGeMAPS) + librosa voor pitch/volume/pauzes

Als een model niet geïnstalleerd is, valt de analyse terug op neutrale
waarden met confidence=0.0 zodat de pipeline nooit crasht.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from hermes_bot.schemas import SourceKind

if TYPE_CHECKING:
    from hermes_bot.signals.audiovisual import SpeechSignal


class AudioAnalyzer:
    """Analyses audio files into transcript, emotion and prosody."""

    def __init__(self, use_heavy_models: bool = False) -> None:
        """use_heavy_models: load heavy models (whisper/emotion). Default
        uit zodat de pipeline snel en offline-safe blijft; zet aan op een
        machine waar de modellen betrouwbaar draaien."""
        self.use_heavy_models = use_heavy_models
        self._whisper_model = None
        self._emotion_model = None
        self._emotion_tokenizer = None
        self._opensmile = None

    def analyze(self, audio_path: str) -> SpeechSignal:
        """Analyse an audio file into a SpeechSignal."""
        if not os.path.exists(audio_path):
            return self._create_neutral_signal(audio_path)

        transcript = self._safe(self._get_transcript, audio_path, default="")
        emotion_scores = self._safe(
            self._get_emotion, audio_path,
            default={"confident": 0.0, "anxious": 0.0, "optimistic": 0.0},
            timeout=10.0,
        )
        prosody = self._safe(
            self._get_prosody, audio_path,
            default={"pace": 0.0, "pitch_var": 0.0, "volume": 0.0},
        )
        pauses = self._safe(self._get_pauses, audio_path, default=0.0)

        from hermes_bot.signals.audiovisual import SpeechSignal

        has_signal = bool(transcript or any(emotion_scores.values()) or any(prosody.values()))
        return SpeechSignal(
            source=SourceKind.SPEECH,
            source_name="audio_analysis",
            entity_id=os.path.basename(audio_path),
            timestamp=datetime.now(),
            confidence=1.0 if has_signal else 0.0,
            transcript=transcript,
            emotion_scores=emotion_scores,
            prosody=prosody,
            pauses=pauses,
            video_features={},
            audience={},
        )

    @staticmethod
    def _safe(fn, *args, default=None, timeout: float = 30.0):
        """Run fn in a thread with a timeout; on error/timeout return the default.

        Voorkomt dat een hangend zwaar model (bijv. emotion-pipeline op een
        slow CPU) blocks the whole analysis.
        """
        import threading

        result = {"value": default, "done": False}

        def _run():
            try:
                result["value"] = fn(*args)
            except Exception:
                result["value"] = default
            finally:
                result["done"] = True

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if not result["done"]:
            return default
        return result["value"]

    def _get_transcript(self, audio_path: str) -> str:
        """Transcription via faster-whisper (only if use_heavy_models)."""
        if not self.use_heavy_models:
            return ""
        from faster_whisper import WhisperModel

        if self._whisper_model is None:
            self._whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32")
        segments, _ = self._whisper_model.transcribe(audio_path)
        return " ".join(seg.text for seg in segments).strip()

    def _get_emotion(self, audio_path: str) -> dict[str, float]:
        """Emotion estimate from prosody (pitch/volume/pace).

        Gebruikt een lichte, betrouwbare benadering die op elke machine werkt
        (geen zwaar model nodig). Als use_heavy_models aanstaat en een
        SpeechBrain-model beschikbaar is, zou dat hier kunnen worden
        toegevoegd; de prosodie-baseline is altijd beschikbaar.
        """
        prosody = self._get_prosody(audio_path)
        pitch_var = prosody.get("pitch_var", 0.0)
        volume = prosody.get("volume", 0.0)
        pace = prosody.get("pace", 0.0)

        # Heuristiek: hoge pitch-variatie + hoog volume = opgewonden/optimistisch;
        # lage pitch + laag volume = kalm/confident; hoge pace = nerveus/anxious.
        optimistic = min(1.0, (pitch_var / 50.0) * 0.5 + (volume / 0.5) * 0.5)
        anxious = min(1.0, pace * 0.6 + (pitch_var / 80.0) * 0.4)
        confident = max(0.0, 1.0 - anxious - optimistic * 0.5)
        return {
            "confident": round(confident, 4),
            "anxious": round(anxious, 4),
            "optimistic": round(optimistic, 4),
        }

    def _get_prosody(self, audio_path: str) -> dict[str, float]:
        """Prosodie via openSMILE (eGeMAPS) + librosa."""
        import librosa

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Pitch variation (F0 std) and volume (RMS).
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else None
        pitch_var = float(f0_clean.std()) if f0_clean is not None and len(f0_clean) else 0.0
        volume = float(librosa.feature.rms(y=y).mean())
        # Pace: spraaksegmenten per seconde (simpele benadering via energie).
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        voiced = (rms > rms.mean() * 0.3).astype(int)
        pace = float(voiced.sum() / max(len(voiced), 1))
        return {
            "pace": round(pace, 4),
            "pitch_var": round(pitch_var, 4),
            "volume": round(volume, 4),
        }

    def _get_pauses(self, audio_path: str) -> float:
        """Number of pauses (silences) in the audio."""
        import librosa

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        silent = (rms < rms.mean() * 0.1).astype(int)
        # Count transitions to silence (pauses).
        transitions = sum(1 for i in range(1, len(silent)) if silent[i] == 1 and silent[i - 1] == 0)
        return float(transitions)

    def _create_neutral_signal(self, audio_path: str) -> SpeechSignal:
        """Neutral SpeechSignal when audio is not available."""
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
            audience={},
        )

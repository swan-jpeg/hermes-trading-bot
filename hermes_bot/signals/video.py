"""LAAG 3: video-analyse (gezichtsuitdrukking, pose, publiek).

Echte model-integratie met offline-fallback:
- Gezichtsuitdrukking: DeepFace (emotion)
- Pose/lichaamstaal: MediaPipe Pose (33 landmarks)
- Publiek: MediaPipe Face Detection (aantal gezichten per frame)

De zware modellen worden alleen geïmporteerd als `use_heavy_models=True`.
Standaard uit (offline-safe) zodat de pipeline snel en stabiel blijft op
machines waar de modellen niet werken (bv. sommige CPU's geven een harde
SIGILL-crash bij het importeren van mediapipe/deepface). Zet aan op een
machine waar de modellen betrouwbaar draaien (bv. Windows met GPU).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class VideoAnalyzer:
    """Analyseert video-bestanden naar video-features."""

    def __init__(self, use_heavy_models: bool = False) -> None:
        """use_heavy_models: laad zware modellen (DeepFace/MediaPipe)."""
        self.use_heavy_models = use_heavy_models
        self._deepface = None
        self._pose = None
        self._face_detection = None

    def analyze(self, video_path: str) -> dict[str, object]:
        """Analyseer een video-bestand naar video-features."""
        if not os.path.exists(video_path):
            return self._create_neutral_output(video_path)

        facial_expression = self._safe(self._get_facial_expression, video_path, default={})
        gesture = self._safe(self._get_gesture, video_path, default={})
        body_language = self._safe(self._get_body_language, video_path, default={})
        audience = self._safe(self._get_audience, video_path, default={})

        has_signal = bool(facial_expression or gesture or body_language or audience)
        return {
            "facial_expression": facial_expression,
            "gesture": gesture,
            "body_language": body_language,
            "audience": audience,
            "confidence": 1.0 if has_signal else 0.0,
        }

    @staticmethod
    def _safe(fn, *args, default=None):
        """Voer fn uit; bij elke fout de default teruggeven (offline-safe)."""
        try:
            return fn(*args)
        except Exception:
            return default

    def _get_facial_expression(self, video_path: str) -> dict[str, float]:
        """Gezichtsuitdrukking via DeepFace (emotion) op het eerste frame."""
        if not self.use_heavy_models:
            return {}
        import cv2
        from deepface import DeepFace

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return {}
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        return {k: float(v) for k, v in result.get("emotion", {}).items()}

    def _get_gesture(self, video_path: str) -> dict[str, float]:
        """Gebaren via MediaPipe Pose (hand omhoog, armen gekruist)."""
        if not self.use_heavy_models:
            return {}
        import cv2
        import mediapipe as mp

        if self._pose is None:
            self._pose = mp.solutions.pose.Pose(static_image_mode=True)
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return {}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return {}
        lm = results.pose_landmarks.landmark
        left_wrist, left_shoulder = lm[15], lm[11]
        right_wrist, right_shoulder = lm[16], lm[12]
        hand_raised = float(
            (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y)
        )
        arm_crossed = float(abs(left_wrist.x - right_wrist.x) < 0.1)
        return {"hand_raised": hand_raised, "arm_crossed": arm_crossed, "pointing": 0.0}

    def _get_body_language(self, video_path: str) -> dict[str, float]:
        """Lichaamstaal via MediaPipe Pose (staand/zittend)."""
        if not self.use_heavy_models:
            return {}
        import cv2
        import mediapipe as mp

        if self._pose is None:
            self._pose = mp.solutions.pose.Pose(static_image_mode=True)
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return {}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return {}
        lm = results.pose_landmarks.landmark
        hip_y = (lm[23].y + lm[24].y) / 2
        knee_y = (lm[25].y + lm[26].y) / 2
        standing = float(hip_y < knee_y)
        return {"standing": standing, "sitting": 1.0 - standing, "walking": 0.0, "running": 0.0}

    def _get_audience(self, video_path: str) -> dict[str, float]:
        """Publiek: aantal gezichten via MediaPipe Face Detection."""
        if not self.use_heavy_models:
            return {}
        import cv2
        import mediapipe as mp

        if self._face_detection is None:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return {}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb)
        count = len(results.detections) if results.detections else 0
        return {"count": float(count), "engaged": 0.0, "neutral": 0.0}

    def _create_neutral_output(self, video_path: str) -> dict[str, object]:
        """Neutrale output als video niet beschikbaar is."""
        return {
            "facial_expression": {},
            "gesture": {},
            "body_language": {},
            "audience": {},
            "confidence": 0.0,
        }

"""Basic blink-based liveness detection.

This module is intentionally optional because landmark detection adds CPU cost.
Enable it for high-risk doors or exams where a quick blink challenge is useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import face_recognition
import numpy as np

from .utils import FaceLocation


Point = tuple[int, int]


@dataclass
class LivenessState:
    verified: bool
    blink_count: int
    ear: float
    message: str


def eye_aspect_ratio(eye_points: Sequence[Point]) -> float:
    """Compute eye aspect ratio from six eye landmark points."""

    if len(eye_points) < 6:
        return 1.0

    points = np.asarray(eye_points, dtype=np.float32)
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 1.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


class BlinkDetector:
    """Track a small blink challenge for one person/session."""

    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_closed_frames: int = 2,
        required_blinks: int = 1,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.consecutive_closed_frames = consecutive_closed_frames
        self.required_blinks = required_blinks
        self.closed_frames = 0
        self.blink_count = 0

    def reset(self) -> None:
        self.closed_frames = 0
        self.blink_count = 0

    def update(self, rgb_frame, face_location: FaceLocation | None = None) -> LivenessState:
        face_locations = [face_location] if face_location is not None else None
        landmarks = face_recognition.face_landmarks(rgb_frame, face_locations=face_locations)

        if not landmarks:
            return LivenessState(False, self.blink_count, 1.0, "No landmarks")

        face_landmarks = landmarks[0]
        left_eye = face_landmarks.get("left_eye", [])
        right_eye = face_landmarks.get("right_eye", [])
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.consecutive_closed_frames:
                self.blink_count += 1
            self.closed_frames = 0

        verified = self.blink_count >= self.required_blinks
        message = "Liveness verified" if verified else "Blink once"
        return LivenessState(verified, self.blink_count, round(ear, 3), message)


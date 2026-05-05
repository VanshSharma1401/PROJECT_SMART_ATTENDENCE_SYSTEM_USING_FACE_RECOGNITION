"""Real-time face recognition and attendance marking."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import face_recognition
import numpy as np

from .attendance import AttendanceEvent, AttendanceLogger
from .config import settings
from .encode import FaceEncodingStore
from .liveness import BlinkDetector
from .utils import (
    FaceLocation,
    bgr_to_rgb,
    classify_distance,
    distance_to_confidence,
    draw_face_box,
    logger,
    open_camera,
    overlay_timestamp,
    resize_for_detection,
    scale_face_locations,
)
from .vector_index import FaissFaceIndex, NumpyFaceIndex


@dataclass
class RecognitionResult:
    name: str
    distance: float | None
    confidence: float
    classification: str
    location: FaceLocation
    attendance_event: AttendanceEvent | None
    liveness_verified: bool
    label: str


class FaceRecognizer:
    """Recognize faces from frames and optionally mark attendance."""

    def __init__(
        self,
        encoding_store: FaceEncodingStore | None = None,
        attendance_logger: AttendanceLogger | None = None,
        recognition_threshold: float = settings.recognition_threshold,
        require_liveness: bool = False,
        mark_probable: bool = False,
        use_faiss: bool = False,
    ) -> None:
        self.encoding_store = encoding_store or FaceEncodingStore()
        self.attendance_logger = attendance_logger or AttendanceLogger()
        self.recognition_threshold = recognition_threshold
        self.require_liveness = require_liveness
        self.mark_probable = mark_probable
        self.use_faiss = use_faiss
        self.liveness_detectors: dict[str, BlinkDetector] = {}
        self.names: list[str] = []
        self.encodings = np.empty((0, 128), dtype=np.float32)
        self.index: NumpyFaceIndex | FaissFaceIndex = NumpyFaceIndex(self.encodings)
        self.reload_encodings()

    def reload_encodings(self) -> None:
        """Build cache if stale, then load the known face index."""

        self.encoding_store.build(force=False)
        self.names, self.encodings, _ = self.encoding_store.load()

        if self.use_faiss:
            self.index = FaissFaceIndex(self.encodings)
        else:
            self.index = NumpyFaceIndex(self.encodings)

        logger.info("Loaded %s known face encodings.", len(self.names))

    def _match(self, face_encoding: np.ndarray) -> tuple[str, float | None, str, float]:
        search_result = self.index.search(face_encoding)
        if search_result is None:
            return "Unknown", None, "Invalid", 0.0

        distance = search_result.distance
        classification = classify_distance(distance)

        if distance >= self.recognition_threshold or classification == "Invalid":
            return "Unknown", distance, "Invalid", distance_to_confidence(distance)

        return (
            self.names[search_result.index],
            distance,
            classification,
            distance_to_confidence(distance),
        )

    def _liveness_for(self, name: str) -> BlinkDetector:
        if name not in self.liveness_detectors:
            self.liveness_detectors[name] = BlinkDetector()
        return self.liveness_detectors[name]

    def recognize_frame(
        self,
        frame_bgr: np.ndarray,
        mark_attendance: bool = False,
        annotate: bool = True,
    ) -> tuple[np.ndarray, list[RecognitionResult]]:
        """Recognize all faces in a frame."""

        annotated = frame_bgr.copy()
        # Compress for detection
        scale = settings.frame_scale
        small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        # Flip BGR to RGB using numpy slice
        rgb_small = small[:, :, ::-1].copy()

        small_locations = face_recognition.face_locations(
            rgb_small,
            model=settings.detection_model,
        )
        small_encodings = face_recognition.face_encodings(
            rgb_small,
            known_face_locations=small_locations,
            model=settings.encoding_model,
        )
        locations = scale_face_locations(small_locations, scale)

        full_rgb = bgr_to_rgb(frame_bgr) if self.require_liveness else None
        results: list[RecognitionResult] = []

        for location, face_encoding in zip(locations, small_encodings):
            name, distance, classification, confidence = self._match(face_encoding)
            liveness_verified = not self.require_liveness
            liveness_message = ""

            if self.require_liveness and name != "Unknown" and full_rgb is not None:
                liveness_state = self._liveness_for(name).update(full_rgb, location)
                liveness_verified = liveness_state.verified
                liveness_message = f" | {liveness_state.message}"

            should_mark = (
                mark_attendance
                and name != "Unknown"
                and liveness_verified
                and (classification == "Valid" or (self.mark_probable and classification == "Probable"))
            )

            attendance_event = None
            if should_mark:
                attendance_event = self.attendance_logger.mark(
                    name=name,
                    confidence=confidence,
                    distance=distance,
                    status=classification,
                )

            display_name = name if name != "Unknown" else "Unknown"
            label = f"{display_name} {confidence:.0f}% {classification}{liveness_message}"
            if attendance_event is not None:
                label = f"{label} | {attendance_event.reason}"
            elif self.require_liveness and name != "Unknown" and not liveness_verified:
                label = f"{label} | not marked"

            color = self._color_for_result(name, classification, liveness_verified)
            if annotate:
                draw_face_box(annotated, location, label, color)

            results.append(
                RecognitionResult(
                    name=name,
                    distance=distance,
                    confidence=confidence,
                    classification=classification,
                    location=location,
                    attendance_event=attendance_event,
                    liveness_verified=liveness_verified,
                    label=label,
                )
            )

        if annotate:
            overlay_timestamp(annotated)

        return annotated, results

    @staticmethod
    def _color_for_result(
        name: str,
        classification: str,
        liveness_verified: bool,
    ) -> tuple[int, int, int]:
        if name == "Unknown" or classification == "Invalid":
            return (45, 45, 220)
        if not liveness_verified:
            return (220, 145, 35)
        if classification == "Probable":
            return (0, 180, 255)
        return (40, 180, 80)


def run_cli(
    camera_index: int = settings.camera_index,
    storage: str = "csv",
    require_liveness: bool = False,
    mark_probable: bool = False,
) -> None:
    """Run real-time attendance in an OpenCV window."""

    attendance_logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
    recognizer = FaceRecognizer(
        attendance_logger=attendance_logger,
        require_liveness=require_liveness,
        mark_probable=mark_probable,
    )

    camera = open_camera(camera_index)
    frame_count = 0
    last_annotated = None
    previous_time = time.time()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame could not be read.")

            frame_count += 1
            if frame_count % settings.process_every_n_frames == 0 or last_annotated is None:
                last_annotated, _ = recognizer.recognize_frame(
                    frame,
                    mark_attendance=True,
                    annotate=True,
                )
            else:
                last_annotated = frame.copy()
                overlay_timestamp(last_annotated)

            current_time = time.time()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            cv2.putText(
                last_annotated,
                f"FPS: {fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Smart Attendance - Recognition", last_annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time face attendance.")
    parser.add_argument("--camera", type=int, default=settings.camera_index)
    parser.add_argument("--storage", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--liveness", action="store_true", help="Require one blink before marking.")
    parser.add_argument(
        "--mark-probable",
        action="store_true",
        help="Also mark probable matches below the recognition threshold.",
    )
    args = parser.parse_args()

    run_cli(
        camera_index=args.camera,
        storage=args.storage,
        require_liveness=args.liveness,
        mark_probable=args.mark_probable,
    )


if __name__ == "__main__":
    main()


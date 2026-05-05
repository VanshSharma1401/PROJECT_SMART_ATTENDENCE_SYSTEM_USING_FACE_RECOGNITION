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
    detect_faces_robust,
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
    """Orchestrates face detection, liveness, recognition, and attendance logging."""

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
        self.require_liveness = require_liveness
        self.mark_probable = mark_probable
        
        # LBPH distance threshold. Lower is better. Typically 0 to ~80 is a match.
        self.lbph_threshold = 85.0
        
        self.liveness_detectors: dict[str, BlinkDetector] = {}
        self.label_to_name: dict[int, str] = {}
        
        import cv2
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.reload_encodings()

    def reload_encodings(self) -> None:
        """Build cache if stale, then load the known face model."""

        self.encoding_store.build(force=False)
        
        # Load the cache mapping
        try:
            import pickle
            with self.encoding_store.encodings_file.open("rb") as f:
                cache = pickle.load(f)
                self.label_to_name = cache.get("label_to_name", {})
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            
        lbph_model_file = self.encoding_store.encodings_file.with_name("lbph_model.yml")
        if lbph_model_file.exists():
            self.recognizer.read(str(lbph_model_file))
            logger.info("Loaded LBPH known face model.")
        else:
            logger.warning("No LBPH model found.")

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
        """Recognize all faces in a frame using LBPH."""

        annotated = frame_bgr.copy()
        import cv2
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Use robust detection (Mediapipe with OpenCV fallback)
        locations = detect_faces_robust(
            rgb, 
            upsample=settings.detection_upsample, 
            model=settings.detection_model
        )
        
        full_rgb = rgb if self.require_liveness else None
        results: list[RecognitionResult] = []

        for location in locations:
            top, right, bottom, left = location
            
            face_crop = gray[top:bottom, left:right]
            if face_crop.size == 0:
                continue
                
            face_crop = cv2.resize(face_crop, (200, 200))
            
            try:
                # Prediction returns label_id and confidence distance (lower is better)
                label_id, distance = self.recognizer.predict(face_crop)
                
                if distance < self.lbph_threshold:
                    name = self.label_to_name.get(label_id, "Unknown")
                    classification = "Valid"
                    confidence = max(0.0, min(100.0, 100.0 - (distance / 2)))
                else:
                    name = "Unknown"
                    classification = "Invalid"
                    confidence = 0.0
                    
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                name = "Unknown"
                classification = "Invalid"
                confidence = 0.0
                distance = 999.0

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


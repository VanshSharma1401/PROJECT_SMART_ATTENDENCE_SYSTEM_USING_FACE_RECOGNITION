"""Face registration workflow for collecting clean user samples."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import face_recognition

from .config import settings
from .utils import (
    FaceLocation,
    assess_image_quality,
    bgr_to_rgb,
    draw_face_box,
    list_image_files,
    logger,
    open_camera,
    resize_for_detection,
    sanitize_person_name,
    save_frame,
    scale_face_locations,
)


@dataclass
class RegistrationResult:
    saved: bool
    status: str
    message: str
    person_name: str
    image_path: Path | None
    face_count: int
    face_location: FaceLocation | None
    quality: dict[str, float | str]
    prompt: str


class FaceRegistrar:
    """Capture and validate dataset images for a person."""

    pose_prompts = (
        "Look straight",
        "Turn slightly left",
        "Turn slightly right",
        "Tilt chin up",
        "Tilt chin down",
        "Smile naturally",
        "Neutral expression",
        "Move closer",
        "Move farther",
    )

    def __init__(
        self,
        dataset_dir: Path = settings.dataset_dir,
        detection_model: str = settings.detection_model,
        frame_scale: float = settings.frame_scale,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.detection_model = detection_model
        self.frame_scale = frame_scale
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def person_dir(self, person_name: str) -> Path:
        safe_name = sanitize_person_name(person_name)
        target = self.dataset_dir / safe_name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def count_samples(self, person_name: str) -> int:
        return len(list_image_files(self.person_dir(person_name)))

    def next_prompt(self, saved_count: int) -> str:
        return self.pose_prompts[saved_count % len(self.pose_prompts)]

    def _next_image_path(self, person_name: str) -> Path:
        target_dir = self.person_dir(person_name)
        sample_number = len(list_image_files(target_dir)) + 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return target_dir / f"{sample_number:03d}_{timestamp}.jpg"

    def detect_faces(self, frame_bgr) -> list[FaceLocation]:
        small_frame = resize_for_detection(frame_bgr, self.frame_scale)
        rgb_small = bgr_to_rgb(small_frame)
        small_locations = face_recognition.face_locations(
            rgb_small,
            model=self.detection_model,
        )
        return scale_face_locations(small_locations, self.frame_scale)

    def validate_and_save_frame(self, frame_bgr, person_name: str) -> RegistrationResult:
        """Save a frame only when exactly one clear face is visible."""

        safe_name = sanitize_person_name(person_name)
        existing_count = self.count_samples(safe_name)
        prompt = self.next_prompt(existing_count)

        quality_ok, quality = assess_image_quality(frame_bgr)
        locations = self.detect_faces(frame_bgr)
        face_count = len(locations)

        if face_count == 0:
            return RegistrationResult(
                saved=False,
                status="no_face",
                message="No face detected",
                person_name=safe_name,
                image_path=None,
                face_count=0,
                face_location=None,
                quality=quality,
                prompt=prompt,
            )

        if face_count > 1:
            return RegistrationResult(
                saved=False,
                status="multiple_faces",
                message="Multiple faces detected",
                person_name=safe_name,
                image_path=None,
                face_count=face_count,
                face_location=None,
                quality=quality,
                prompt=prompt,
            )

        if not quality_ok:
            return RegistrationResult(
                saved=False,
                status="low_quality",
                message=f"Image rejected: {quality['reason']}",
                person_name=safe_name,
                image_path=None,
                face_count=face_count,
                face_location=locations[0],
                quality=quality,
                prompt=prompt,
            )

        image_path = self._next_image_path(safe_name)
        save_frame(frame_bgr, image_path)
        logger.info("Saved registration sample: %s", image_path)

        return RegistrationResult(
            saved=True,
            status="saved",
            message="Sample saved",
            person_name=safe_name,
            image_path=image_path,
            face_count=face_count,
            face_location=locations[0],
            quality=quality,
            prompt=self.next_prompt(existing_count + 1),
        )

    def capture_cli(
        self,
        person_name: str,
        samples: int = settings.default_samples_per_user,
        camera_index: int = settings.camera_index,
    ) -> int:
        """Open webcam and collect registration samples in an OpenCV window."""

        safe_name = sanitize_person_name(person_name)
        samples = max(settings.minimum_samples_per_user, min(settings.maximum_samples_per_user, samples))
        camera = open_camera(camera_index)
        saved_count = self.count_samples(safe_name)
        target_count = saved_count + samples
        last_capture = 0.0
        start_time = time.time()
        last_message = "Align one face inside the camera"

        try:
            while saved_count < target_count:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame could not be read.")

                now = time.time()
                if now - last_capture >= settings.registration_capture_interval_sec:
                    result = self.validate_and_save_frame(frame, safe_name)
                    last_capture = now
                    last_message = result.message
                    saved_count = self.count_samples(safe_name)

                    if result.face_location:
                        draw_face_box(
                            frame,
                            result.face_location,
                            result.status.replace("_", " ").title(),
                            (0, 180, 255) if not result.saved else (40, 180, 80),
                        )

                prompt = self.next_prompt(saved_count)
                cv2.putText(
                    frame,
                    f"{safe_name}: {saved_count}/{target_count} | {prompt}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    last_message,
                    (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Smart Attendance - Register", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if time.time() - start_time > settings.registration_timeout_sec:
                    logger.warning("Registration timed out for %s", safe_name)
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

        return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture face samples for a new user.")
    parser.add_argument("name", help="Person name, used as dataset folder name.")
    parser.add_argument("--samples", type=int, default=settings.default_samples_per_user)
    parser.add_argument("--camera", type=int, default=settings.camera_index)
    args = parser.parse_args()

    registrar = FaceRegistrar()
    count = registrar.capture_cli(args.name, args.samples, args.camera)
    print(f"Registration complete. Total samples for {sanitize_person_name(args.name)}: {count}")


if __name__ == "__main__":
    main()


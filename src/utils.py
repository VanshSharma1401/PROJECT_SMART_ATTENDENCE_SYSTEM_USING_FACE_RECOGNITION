"""Shared helpers for camera, dataset, drawing, and logging work."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .config import settings


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FaceLocation = Tuple[int, int, int, int]  # top, right, bottom, left


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging once."""

    logger = logging.getLogger("smart_attendance")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    target_log_file = log_file or settings.system_log_file
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target_log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


def sanitize_person_name(raw_name: str) -> str:
    """Return a filesystem-safe person identifier."""

    normalized = unicodedata.normalize("NFKD", raw_name or "")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe_name = re.sub(r"[^A-Za-z0-9_. -]+", "", ascii_name).strip()
    safe_name = re.sub(r"\s+", "_", safe_name)
    safe_name = safe_name.strip("._-")

    if not safe_name:
        raise ValueError("Person name must contain at least one letter or number.")

    return safe_name


def list_people(dataset_dir: Path = settings.dataset_dir) -> List[str]:
    """List known people from dataset folders."""

    if not dataset_dir.exists():
        return []

    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def list_image_files(root: Path) -> List[Path]:
    """Return supported image files below a path, sorted for deterministic output."""

    if not root.exists():
        return []

    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def dataset_signature(dataset_dir: Path = settings.dataset_dir) -> dict[str, dict[str, int]]:
    """Build a cheap cache signature from relative paths, sizes, and mtimes."""

    signature: dict[str, dict[str, int]] = {}
    for image_path in list_image_files(dataset_dir):
        stat = image_path.stat()
        signature[str(image_path.relative_to(dataset_dir))] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return signature


def now_local() -> datetime:
    """Central time helper for easier testing and future timezone hooks."""

    return datetime.now()


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def open_camera(camera_index: int = settings.camera_index) -> cv2.VideoCapture:
    """Open a webcam and fail with a clear error if unavailable."""

    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)

    if not camera.isOpened():
        raise RuntimeError(
            f"Camera index {camera_index} could not be opened. "
            "Check camera permissions or try another index."
        )

    return camera


def resize_for_detection(frame_bgr: np.ndarray, scale: float = settings.frame_scale) -> np.ndarray:
    """Downscale a frame before face detection."""

    if scale <= 0 or scale > 1:
        raise ValueError("Frame scale must be between 0 and 1.")

    if math.isclose(scale, 1.0):
        return frame_bgr

    return cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def scale_face_locations(locations: Sequence[FaceLocation], scale: float) -> List[FaceLocation]:
    """Scale face_recognition locations from resized frame coordinates."""

    if math.isclose(scale, 1.0):
        return list(locations)

    factor = 1.0 / scale
    return [
        (
            int(top * factor),
            int(right * factor),
            int(bottom * factor),
            int(left * factor),
        )
        for top, right, bottom, left in locations
    ]


def draw_face_box(
    frame_bgr: np.ndarray,
    location: FaceLocation,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a bounding box and readable label on a BGR frame."""

    top, right, bottom, left = location
    cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)

    label_height = 28
    label_top = max(0, top - label_height)
    cv2.rectangle(frame_bgr, (left, label_top), (right, top), color, cv2.FILLED)
    cv2.putText(
        frame_bgr,
        label,
        (left + 6, max(18, top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def overlay_timestamp(frame_bgr: np.ndarray, timestamp: datetime | None = None) -> None:
    """Add a timestamp overlay to the live frame."""

    current = timestamp or now_local()
    text = current.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame_bgr,
        text,
        (16, frame_bgr.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def classify_distance(
    distance: float | None,
    valid_threshold: float = settings.valid_distance_threshold,
    probable_threshold: float = settings.probable_distance_threshold,
) -> str:
    """Map an embedding distance to the required confidence class."""

    if distance is None or not np.isfinite(distance):
        return "Invalid"
    if distance < valid_threshold:
        return "Valid"
    if distance < probable_threshold:
        return "Probable"
    return "Invalid"


def distance_to_confidence(distance: float | None) -> float:
    """Convert Euclidean distance into a human-friendly percentage."""

    if distance is None or not np.isfinite(distance):
        return 0.0
    return round(max(0.0, min(100.0, (1.0 - distance) * 100.0)), 2)


def assess_image_quality(frame_bgr: np.ndarray) -> tuple[bool, dict[str, float | str]]:
    """Reject frames that are too dark, too bright, or too blurry."""

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons: list[str] = []
    if brightness < settings.min_brightness:
        reasons.append("too dark")
    if brightness > settings.max_brightness:
        reasons.append("too bright")
    if blur_variance < settings.min_blur_variance:
        reasons.append("too blurry")

    details: dict[str, float | str] = {
        "brightness": round(brightness, 2),
        "blur_variance": round(blur_variance, 2),
        "reason": ", ".join(reasons),
    }
    return not reasons, details


def save_frame(frame_bgr: np.ndarray, target_path: Path) -> None:
    """Save a frame and raise a clear error if OpenCV fails."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(target_path), frame_bgr)
    if not success:
        raise IOError(f"Could not write image to {target_path}")


def chunked(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


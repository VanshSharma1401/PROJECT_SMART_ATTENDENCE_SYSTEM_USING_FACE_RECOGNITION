"""Central configuration for the Smart Attendance System."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Runtime defaults.

    Values are collected in one place so the CLI modules, Streamlit UI, and
    future API layer all use the same behavior.
    """

    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = PROJECT_ROOT / "dataset"
    encodings_dir: Path = PROJECT_ROOT / "encodings"
    logs_dir: Path = PROJECT_ROOT / "logs"

    encodings_file: Path = PROJECT_ROOT / "encodings" / "known_faces.pkl"
    attendance_csv: Path = PROJECT_ROOT / "logs" / "attendance.csv"
    sqlite_db: Path = PROJECT_ROOT / "logs" / "attendance.db"
    system_log_file: Path = PROJECT_ROOT / "logs" / "system.log"

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    default_samples_per_user: int = 25
    minimum_samples_per_user: int = 20
    maximum_samples_per_user: int = 30

    detection_model: str = "hog"
    detection_upsample: int = 2
    encoding_model: str = "large"
    frame_scale: float = 0.5
    process_every_n_frames: int = 2

    recognition_threshold: float = 0.55
    valid_distance_threshold: float = 0.45
    probable_distance_threshold: float = 0.55
    cooldown_minutes: int = 60

    registration_capture_interval_sec: float = 0.35
    registration_timeout_sec: int = 180
    min_brightness: float = 35.0
    max_brightness: float = 225.0
    min_blur_variance: float = 35.0


settings = Settings()


def ensure_project_dirs() -> None:
    """Create runtime directories if they do not exist."""

    for directory in (settings.dataset_dir, settings.encodings_dir, settings.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


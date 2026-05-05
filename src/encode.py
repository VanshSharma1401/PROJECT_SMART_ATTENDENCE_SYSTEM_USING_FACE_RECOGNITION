"""Build and cache 128-dimensional face encodings from the dataset."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .utils import dataset_signature, list_image_files, logger


CACHE_VERSION = 1


def require_face_recognition():
    """Import face_recognition only when dataset encoding work is requested."""

    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError(
            "face_recognition/dlib is not available in the current environment. "
            "Use the Python 3.11 environment described in the README."
        ) from exc
    return face_recognition


@dataclass
class EncodingBuildReport:
    rebuilt: bool
    total_images: int
    encoded_faces: int
    skipped_images: int
    people: int
    cache_path: Path
    errors: list[str]


class FaceEncodingStore:
    """Manage cached face embeddings for known users."""

    def __init__(
        self,
        dataset_dir: Path = settings.dataset_dir,
        encodings_file: Path = settings.encodings_file,
        detection_model: str = settings.detection_model,
        encoding_model: str = settings.encoding_model,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.encodings_file = encodings_file
        self.detection_model = detection_model
        self.encoding_model = encoding_model
        self.encodings_file.parent.mkdir(parents=True, exist_ok=True)

    def load_cache(self) -> dict[str, Any] | None:
        """Load cache, returning None if it is missing or corrupted."""

        if not self.encodings_file.exists():
            return None

        try:
            with self.encodings_file.open("rb") as cache_file:
                cache = pickle.load(cache_file)
        except (pickle.PickleError, EOFError, OSError, AttributeError, ValueError) as exc:
            logger.warning("Encoding cache is unreadable and will be rebuilt: %s", exc)
            return None

        if cache.get("version") != CACHE_VERSION:
            logger.info("Encoding cache version changed; rebuilding.")
            return None

        return cache

    def cache_is_current(self) -> bool:
        cache = self.load_cache()
        if cache is None:
            return False
        return cache.get("dataset_signature") == dataset_signature(self.dataset_dir)

    def load(self) -> tuple[list[str], np.ndarray, dict[str, Any]]:
        """Load cached names and embeddings."""

        cache = self.load_cache()
        if cache is None:
            return [], np.empty((0, 128), dtype=np.float32), {}

        names = list(cache.get("names", []))
        encodings = np.asarray(cache.get("encodings", []), dtype=np.float32)
        if encodings.size == 0:
            encodings = np.empty((0, 128), dtype=np.float32)
        return names, encodings, cache

    def build(self, force: bool = False, num_jitters: int = 1) -> EncodingBuildReport:
        """Build encodings only when the dataset changed unless forced."""

        face_recognition = require_face_recognition()
        current_signature = dataset_signature(self.dataset_dir)
        current_cache = self.load_cache()
        if (
            not force
            and current_cache is not None
            and current_cache.get("dataset_signature") == current_signature
        ):
            names = current_cache.get("names", [])
            return EncodingBuildReport(
                rebuilt=False,
                total_images=len(current_signature),
                encoded_faces=len(names),
                skipped_images=0,
                people=len(set(names)),
                cache_path=self.encodings_file,
                errors=[],
            )

        image_paths = list_image_files(self.dataset_dir)
        names: list[str] = []
        encodings: list[np.ndarray] = []
        encoded_image_paths: list[str] = []
        errors: list[str] = []

        for image_path in image_paths:
            person_name = image_path.parent.name
            relative_path = str(image_path.relative_to(self.dataset_dir))

            try:
                image_rgb = face_recognition.load_image_file(str(image_path))
                locations = face_recognition.face_locations(
                    image_rgb,
                    model=self.detection_model,
                )

                if len(locations) == 0:
                    errors.append(f"{relative_path}: no face detected")
                    continue
                if len(locations) > 1:
                    errors.append(f"{relative_path}: multiple faces detected")
                    continue

                image_encodings = face_recognition.face_encodings(
                    image_rgb,
                    known_face_locations=locations,
                    num_jitters=num_jitters,
                    model=self.encoding_model,
                )

                if not image_encodings:
                    errors.append(f"{relative_path}: encoding failed")
                    continue

                names.append(person_name)
                encodings.append(np.asarray(image_encodings[0], dtype=np.float32))
                encoded_image_paths.append(relative_path)
            except Exception as exc:  # noqa: BLE001 - corrupted images can fail in many libraries.
                errors.append(f"{relative_path}: {exc}")

        encoding_matrix = (
            np.vstack(encodings).astype(np.float32)
            if encodings
            else np.empty((0, 128), dtype=np.float32)
        )

        cache = {
            "version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_signature": current_signature,
            "names": names,
            "encodings": encoding_matrix,
            "image_paths": encoded_image_paths,
            "stats": {
                "total_images": len(image_paths),
                "encoded_faces": len(names),
                "skipped_images": len(image_paths) - len(names),
                "people": len(set(names)),
            },
        }

        with self.encodings_file.open("wb") as cache_file:
            pickle.dump(cache, cache_file)

        report = EncodingBuildReport(
            rebuilt=True,
            total_images=len(image_paths),
            encoded_faces=len(names),
            skipped_images=len(image_paths) - len(names),
            people=len(set(names)),
            cache_path=self.encodings_file,
            errors=errors,
        )
        logger.info(
            "Encoding build complete: %s faces from %s images across %s people.",
            report.encoded_faces,
            report.total_images,
            report.people,
        )
        return report

    def validate_dataset(self, deep: bool = False) -> dict[str, Any]:
        """Return dataset health details.

        The lightweight mode counts images per person. Deep mode runs face
        detection and reports invalid samples without rebuilding encodings.
        """

        people: dict[str, dict[str, Any]] = {}
        image_paths = list_image_files(self.dataset_dir)

        for image_path in image_paths:
            person_name = image_path.parent.name
            people.setdefault(
                person_name,
                {"images": 0, "valid": 0, "invalid": 0, "issues": []},
            )
            people[person_name]["images"] += 1

        if deep:
            face_recognition = require_face_recognition()
            for image_path in image_paths:
                person_name = image_path.parent.name
                relative_path = str(image_path.relative_to(self.dataset_dir))
                try:
                    image_rgb = face_recognition.load_image_file(str(image_path))
                    locations = face_recognition.face_locations(
                        image_rgb,
                        model=self.detection_model,
                    )
                    if len(locations) == 1:
                        people[person_name]["valid"] += 1
                    else:
                        people[person_name]["invalid"] += 1
                        reason = "no face" if len(locations) == 0 else "multiple faces"
                        people[person_name]["issues"].append(f"{relative_path}: {reason}")
                except Exception as exc:  # noqa: BLE001
                    people[person_name]["invalid"] += 1
                    people[person_name]["issues"].append(f"{relative_path}: {exc}")

        return {
            "dataset_dir": str(self.dataset_dir),
            "people_count": len(people),
            "image_count": len(image_paths),
            "cache_current": self.cache_is_current(),
            "people": people,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build face encodings from dataset images.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when cache is current.")
    parser.add_argument("--jitters", type=int, default=1, help="Dlib encoding jitter count.")
    parser.add_argument("--validate", action="store_true", help="Validate dataset before building.")
    parser.add_argument("--deep", action="store_true", help="Run deep dataset validation.")
    args = parser.parse_args()

    store = FaceEncodingStore()

    if args.validate:
        print(store.validate_dataset(deep=args.deep))

    report = store.build(force=args.force, num_jitters=args.jitters)
    print(report)
    if report.errors:
        print("\nSkipped images:")
        for error in report.errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()

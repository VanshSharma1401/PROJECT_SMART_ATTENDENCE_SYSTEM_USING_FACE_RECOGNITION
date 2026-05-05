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
from .utils import (
    dataset_signature,
    detect_faces_robust,
    list_image_files,
    logger,
)


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
        import cv2

        names: list[str] = []
        encoded_image_paths: list[str] = []
        errors: list[str] = []
        
        # LBPH specific lists
        face_samples = []
        face_ids = []
        label_to_name = {}
        name_to_label = {}
        current_label_id = 0

        # We will still write to encodings_file for metadata/cache
        # but the actual model is saved to an XML file
        lbph_model_file = self.encodings_file.with_name("lbph_model.yml")

        for image_path in image_paths:
            person_name = image_path.parent.name
            relative_path = str(image_path.relative_to(self.dataset_dir))
            
            if person_name not in name_to_label:
                name_to_label[person_name] = current_label_id
                label_to_name[current_label_id] = person_name
                current_label_id += 1
                
            label_id = name_to_label[person_name]

            try:
                print(f"Processing: {relative_path}...")
                
                # Load with OpenCV directly to Grayscale (LBPH requires grayscale)
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    print(f"  - Failed to read file: {relative_path}")
                    errors.append(f"{relative_path}: could not read file")
                    continue
                
                # We need RGB for mediapipe detection
                bgr = cv2.imread(str(image_path))
                image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Detect face with mediapipe
                locations = detect_faces_robust(
                    image_rgb, 
                    upsample=settings.detection_upsample, 
                    model=self.detection_model
                )

                if len(locations) == 0:
                    print(f"  - No face found in {relative_path}")
                    errors.append(f"{relative_path}: no face detected")
                    continue
                
                # Crop the face from the grayscale image
                top, right, bottom, left = locations[0]
                face_crop = gray[top:bottom, left:right]
                
                if face_crop.size == 0:
                     errors.append(f"{relative_path}: invalid face crop")
                     continue
                     
                # Standardize face size for LBPH
                face_crop = cv2.resize(face_crop, (200, 200))

                face_samples.append(face_crop)
                face_ids.append(label_id)
                
                names.append(person_name)
                encoded_image_paths.append(relative_path)
                print(f"  - Success: {relative_path}")
                
            except Exception as exc:  # noqa: BLE001
                print(f"  - ERROR: {relative_path}: {exc}")
                errors.append(f"{relative_path}: {exc}")

        # Train LBPH Model
        if face_samples:
            print("Training LBPH model...")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(face_samples, np.array(face_ids))
            recognizer.write(str(lbph_model_file))
            print(f"Saved model to {lbph_model_file}")

        cache = {
            "version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_signature": current_signature,
            "names": names,
            "label_to_name": label_to_name,
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
                    
                    # Ensure image is uint8 RGB as required by dlib
                    if image_rgb.dtype != np.uint8:
                        image_rgb = image_rgb.astype(np.uint8)
                    if image_rgb.ndim == 2:
                        import cv2
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
                    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
                        image_rgb = image_rgb[:, :, :3]
                    
                    image_rgb = np.ascontiguousarray(image_rgb)

                    # Use robust detection
                    locations = detect_faces_robust(
                        image_rgb, 
                        upsample=settings.detection_upsample, 
                        model=self.detection_model
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

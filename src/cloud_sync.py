"""Cloud-sync extension points.

Production teams can implement the provider protocol for S3, Google Drive,
Firebase Storage, or an internal object store without changing attendance code.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import settings


class CloudSyncProvider(Protocol):
    def push_file(self, local_path: Path, remote_key: str) -> str:
        """Upload one file and return its remote URI."""


@dataclass
class LocalMirrorSync:
    """Simple filesystem mirror used for dry runs and office NAS backups."""

    mirror_root: Path

    def push_file(self, local_path: Path, remote_key: str) -> str:
        target = self.mirror_root / remote_key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)
        return str(target)


def sync_runtime_files(provider: CloudSyncProvider) -> list[str]:
    """Push core runtime artifacts through a provider."""

    uploaded: list[str] = []
    for path, key in (
        (settings.attendance_csv, "logs/attendance.csv"),
        (settings.sqlite_db, "logs/attendance.db"),
        (settings.encodings_file, "encodings/known_faces.pkl"),
    ):
        if path.exists():
            uploaded.append(provider.push_file(path, key))
    return uploaded


"""Optional FastAPI layer for integrations and cloud-ready deployments."""

from __future__ import annotations

from fastapi import FastAPI, Query

from .attendance import AttendanceLogger
from .encode import FaceEncodingStore
from .utils import list_people


app = FastAPI(title="Smart Attendance API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, object]:
    store = FaceEncodingStore()
    return {
        "status": "ok",
        "people": len(list_people()),
        "cache_current": store.cache_is_current(),
    }


@app.get("/attendance")
def attendance(limit: int = Query(100, ge=1, le=5000), storage: str = "csv") -> dict[str, object]:
    backend = "sqlite" if storage == "sqlite" else "csv"
    logger = AttendanceLogger(storage=backend)  # type: ignore[arg-type]
    return {"rows": logger.read_logs(limit=limit)}


@app.post("/encodings/rebuild")
def rebuild_encodings(force: bool = True) -> dict[str, object]:
    report = FaceEncodingStore().build(force=force)
    return {
        "rebuilt": report.rebuilt,
        "total_images": report.total_images,
        "encoded_faces": report.encoded_faces,
        "skipped_images": report.skipped_images,
        "people": report.people,
        "errors": report.errors,
    }


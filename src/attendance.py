"""Attendance persistence with CSV and SQLite backends."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal

from .config import settings
from .utils import logger, now_local


StorageBackend = Literal["csv", "sqlite"]


@dataclass
class AttendanceEvent:
    name: str
    date: str
    time: str
    timestamp: str
    confidence: float
    distance: float | None
    status: str
    source: str
    marked: bool
    reason: str


class AttendanceLogger:
    """Mark and query attendance while enforcing duplicate cooldowns."""

    columns = [
        "name",
        "date",
        "time",
        "timestamp",
        "confidence",
        "distance",
        "status",
        "source",
    ]

    def __init__(
        self,
        storage: StorageBackend = "csv",
        csv_path: Path = settings.attendance_csv,
        sqlite_path: Path = settings.sqlite_db,
        cooldown_minutes: int = settings.cooldown_minutes,
    ) -> None:
        self.storage = storage
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_seen: dict[str, datetime] = {}

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        if self.storage == "sqlite":
            self._init_sqlite()
        else:
            self._init_csv()

        self._load_last_seen()

    def _init_csv(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.columns)
                writer.writeheader()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    distance REAL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_attendance_name_timestamp "
                "ON attendance(name, timestamp)"
            )
            connection.commit()

    def _load_last_seen(self) -> None:
        rows = self.read_logs(limit=None)
        for row in rows:
            try:
                timestamp = datetime.fromisoformat(str(row["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue

            name = str(row.get("name", "")).strip()
            if not name:
                continue

            if name not in self.last_seen or timestamp > self.last_seen[name]:
                self.last_seen[name] = timestamp

    def _can_mark(self, name: str, timestamp: datetime) -> tuple[bool, str]:
        last_timestamp = self.last_seen.get(name)
        if last_timestamp is None:
            return True, "new attendance"

        elapsed = timestamp - last_timestamp
        if elapsed < self.cooldown:
            minutes_left = int((self.cooldown - elapsed).total_seconds() // 60) + 1
            return False, f"cooldown active ({minutes_left} min left)"

        return True, "cooldown elapsed"

    def mark(
        self,
        name: str,
        confidence: float,
        distance: float | None,
        status: str,
        source: str = "camera",
        timestamp: datetime | None = None,
    ) -> AttendanceEvent:
        """Record attendance unless the person is within cooldown."""

        current = timestamp or now_local()
        allowed, reason = self._can_mark(name, current)

        event = AttendanceEvent(
            name=name,
            date=current.strftime("%Y-%m-%d"),
            time=current.strftime("%H:%M:%S"),
            timestamp=current.isoformat(timespec="seconds"),
            confidence=round(float(confidence), 2),
            distance=None if distance is None else round(float(distance), 5),
            status=status,
            source=source,
            marked=allowed,
            reason=reason,
        )

        if not allowed:
            return event

        if self.storage == "sqlite":
            self._write_sqlite(event)
        else:
            self._write_csv(event)

        self.last_seen[name] = current
        logger.info("Attendance marked for %s (%s, %.2f%%)", name, status, confidence)
        return event

    def _write_csv(self, event: AttendanceEvent) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.columns)
            row = asdict(event)
            writer.writerow({column: row[column] for column in self.columns})

    def _write_sqlite(self, event: AttendanceEvent) -> None:
        row = asdict(event)
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO attendance
                    (name, date, time, timestamp, confidence, distance, status, source)
                VALUES
                    (:name, :date, :time, :timestamp, :confidence, :distance, :status, :source)
                """,
                {column: row[column] for column in self.columns},
            )
            connection.commit()

    def read_logs(self, limit: int | None = 100) -> list[dict[str, Any]]:
        """Read attendance logs newest first."""

        if self.storage == "sqlite":
            return self._read_sqlite(limit)
        return self._read_csv(limit)

    def _read_csv(self, limit: int | None) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            return []

        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        rows.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
        return rows if limit is None else rows[:limit]

    def _read_sqlite(self, limit: int | None) -> list[dict[str, Any]]:
        if not self.sqlite_path.exists():
            return []

        query = (
            "SELECT name, date, time, timestamp, confidence, distance, status, source "
            "FROM attendance ORDER BY timestamp DESC"
        )
        params: Iterable[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params = [limit]

        with sqlite3.connect(self.sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def export_csv(self, target_path: Path | None = None) -> Path:
        """Export the active backend to a CSV file."""

        target = target_path or self.csv_path
        rows = self.read_logs(limit=None)
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.columns)
            writer.writeheader()
            for row in reversed(rows):
                writer.writerow({column: row.get(column, "") for column in self.columns})

        return target


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect attendance logs.")
    parser.add_argument("--storage", choices=["csv", "sqlite"], default="csv")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    attendance = AttendanceLogger(storage=args.storage)
    for row in attendance.read_logs(limit=args.limit):
        print(row)


if __name__ == "__main__":
    main()


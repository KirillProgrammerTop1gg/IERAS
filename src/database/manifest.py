"""Export file manifests and verify registered checksums."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.schema import FileRecord, Observation
from src.utils.hashing import sha256_file
from src.utils.paths import ProjectPaths

MANIFEST_COLUMNS = (
    "file_path",
    "sha256",
    "file_size",
    "archive",
    "mission",
    "archive_id",
    "target_name",
    "product_filename",
    "product_subgroup",
    "downloaded_at",
    "validation_status",
)


def export_manifest(session: Session, out_path: Path) -> int:
    """Write a UTF-8 CSV manifest and return its row count."""
    statement = (
        select(
            FileRecord.file_path,
            FileRecord.sha256,
            FileRecord.file_size,
            Observation.archive,
            Observation.mission,
            Observation.archive_id,
            Observation.target_name,
            FileRecord.product_filename,
            FileRecord.product_subgroup,
            FileRecord.downloaded_at,
            FileRecord.validation_status,
        )
        .join(Observation, Observation.observation_id == FileRecord.observation_id)
        .order_by(FileRecord.file_path)
    )
    rows = session.execute(statement).mappings().all()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow([row[column] for column in MANIFEST_COLUMNS])
    return len(rows)


@dataclass(frozen=True)
class FileCheck:
    file_path: str
    status: str  # ok | missing | mismatch
    expected_sha256: str
    actual_sha256: str | None


def verify_files(session: Session, paths: ProjectPaths) -> list[FileCheck]:
    """Recompute SHA-256 for registered files and compare with the database."""
    results: list[FileCheck] = []
    rows = session.execute(
        select(FileRecord.file_path, FileRecord.sha256).order_by(FileRecord.file_path)
    )
    for file_path, expected_sha256 in rows:
        absolute = paths.absolute(file_path)
        if not absolute.exists():
            results.append(FileCheck(file_path, "missing", expected_sha256, None))
            continue
        actual = sha256_file(absolute)
        status = "ok" if actual == expected_sha256 else "mismatch"
        results.append(FileCheck(file_path, status, expected_sha256, actual))
    return results

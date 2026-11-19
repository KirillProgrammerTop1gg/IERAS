"""Project paths are defined here for use by every module.

Database and manifest paths are relative to the project root and use forward slashes
('data/raw/TESS/TOI-700/...'). This keeps records portable across platforms and locations.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# src/utils/paths.py -> src/utils -> src -> project root
DEFAULT_ROOT = Path(__file__).resolve().parents[2]

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """Convert an object name into a safe directory name: 'HD 189733' -> 'HD_189733'."""
    cleaned = _UNSAFE_CHARS.sub("_", name.strip()).strip("_")
    if not cleaned:
        raise ValueError(f"Empty name after sanitization: {name!r}")
    return cleaned


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        """Immutable source data; only initial downloads are written here."""
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def datasets(self) -> Path:
        return self.data / "datasets"

    @property
    def metadata(self) -> Path:
        return self.data / "metadata"

    @property
    def snapshots(self) -> Path:
        """Immutable timestamped archive-response snapshots."""
        return self.metadata / "snapshots"

    @property
    def db_path(self) -> Path:
        return self.metadata / "exoplanet.db"

    @property
    def manifest_path(self) -> Path:
        return self.metadata / "manifest_files.csv"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def fits_reports(self) -> Path:
        return self.reports / "fits_checks"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    def raw_dir(self, mission: str, target: str) -> Path:
        """Raw-file directory: data/raw/<mission>/<target>/."""
        return self.raw / safe_name(mission) / safe_name(target)

    def ensure_dirs(self) -> None:
        """Create required directories without modifying existing files."""
        for folder in (
            self.raw,
            self.interim,
            self.processed,
            self.datasets,
            self.metadata,
            self.snapshots,
            self.fits_reports,
            self.logs,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        """Return a root-relative 'data/raw/...' path with forward slashes."""
        resolved = Path(path).resolve()
        try:
            rel = resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"File is outside the project root: {resolved}") from exc
        return rel.as_posix()

    def absolute(self, relative_path: str) -> Path:
        return self.root / relative_path


def get_paths(root: str | os.PathLike[str] | None = None) -> ProjectPaths:
    """Return project paths.

    Priority: explicit root -> IERAS_ROOT environment variable -> project directory.
    """
    if root is None:
        root = os.environ.get("IERAS_ROOT") or DEFAULT_ROOT
    return ProjectPaths(root=Path(root).expanduser().resolve())

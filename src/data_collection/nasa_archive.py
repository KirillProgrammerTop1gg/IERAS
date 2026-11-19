"""NASA Exoplanet Archive client: TAP/ADQL through PyVO, using pscomppars.

pscomppars is a summary table with one row per planet and default parameters.
It is not ground-truth labeling: each parameter has its own *_reflink source.
Sources are retained for scientific labeling in later stages.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from astropy.table import Table
from pyvo.dal import TAPService

from src.utils.config import Settings, load_settings
from src.utils.hashing import sha256_file
from src.utils.paths import ProjectPaths

# Columns requested from the standard pscomppars schema.
# Update this list if the archive renames a column.
QUERY_COLUMNS = (
    "pl_name",
    "hostname",
    "ra",
    "dec",
    "discoverymethod",
    "disc_year",
    "pl_orbper",
    "pl_orbper_reflink",
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_rade_reflink",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    "pl_bmasse_reflink",
    "st_teff",
    "st_teff_reflink",
    "st_rad",
    "st_rad_reflink",
)

SERVICE_NAME = "NASA Exoplanet Archive"
_HTML_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class QueryInfo:
    service: str
    tap_url: str
    table: str
    query: str
    retrieved_at: str
    n_rows: int


def _escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def build_planets_query(
    *,
    table: str,
    limit: int | None = None,
    hostname: str | None = None,
    require_radius: bool = True,
) -> str:
    """Build a pscomppars ADQL query with validated and escaped parameters."""
    columns = ",\n    ".join(QUERY_COLUMNS)
    top = f"TOP {int(limit)} " if limit else ""
    conditions = []
    if require_radius:
        conditions.append("pl_rade IS NOT NULL")
    if hostname:
        conditions.append(f"hostname = '{_escape_sql_string(hostname)}'")
    where = f"\nWHERE {' AND '.join(conditions)}" if conditions else ""
    return f"SELECT {top}{columns}\nFROM {table}{where}"


def fetch_planets(
    limit: int | None = 20,
    *,
    hostname: str | None = None,
    require_radius: bool = True,
    settings: Settings | None = None,
) -> tuple[Table, QueryInfo]:
    """Load pscomppars rows and return the table and query provenance."""
    settings = settings or load_settings()
    query = build_planets_query(
        table=settings.nasa_table,
        limit=limit,
        hostname=hostname,
        require_radius=require_radius,
    )
    service = TAPService(settings.nasa_tap_url)
    result = service.search(query).to_table()
    info = QueryInfo(
        service=SERVICE_NAME,
        tap_url=settings.nasa_tap_url,
        table=settings.nasa_table,
        query=query,
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_rows=len(result),
    )
    return result, info


def strip_html(text: str | None) -> str | None:
    """Remove HTML markup from a *_reflink value."""
    if not text:
        return None
    cleaned = _HTML_TAG.sub("", text).strip()
    return cleaned or None


def save_snapshot(
    table: Table, info: QueryInfo, paths: ProjectPaths, *, update_latest: bool = True
) -> tuple[Path, Path]:
    """Save an immutable response snapshot and JSON query description.

    Timestamped filenames prevent snapshot overwrites. If update_latest is set,
    also update data/metadata/planets_sample.csv.
    """
    stamp = info.retrieved_at.replace(":", "").replace("-", "")
    csv_path = paths.snapshots / f"{info.table}_{stamp}.csv"
    counter = 1
    while csv_path.exists():  # Number simultaneous snapshots instead of overwriting them.
        counter += 1
        csv_path = paths.snapshots / f"{info.table}_{stamp}_{counter}.csv"
    meta_path = csv_path.with_suffix(".json")

    table.write(csv_path, format="csv", overwrite=False)
    meta = {
        **asdict(info),
        "csv_file": paths.relative(csv_path),
        "csv_sha256": sha256_file(csv_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if update_latest:
        table.write(paths.metadata / "planets_sample.csv", format="csv", overwrite=True)
    return csv_path, meta_path

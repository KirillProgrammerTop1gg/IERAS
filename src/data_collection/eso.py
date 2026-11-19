"""ESO Archive TAP client (ObsCore table). Retrieves metadata without bulk downloads.

Search HARPS and ESPRESSO radial-velocity observations by coordinate cone, since object
names differ across archives.
"""

from __future__ import annotations

from datetime import datetime, timezone

from astropy.table import Table
from pyvo.dal import TAPService

from src.data_collection.nasa_archive import QueryInfo
from src.utils.config import Settings, load_settings

SERVICE_NAME = "ESO Science Archive"
OBSCORE_TABLE = "ivoa.obscore"

OBSCORE_COLUMNS = (
    "obs_id",
    "obs_collection",
    "instrument_name",
    "target_name",
    "s_ra",
    "s_dec",
    "t_min",
    "t_exptime",
    "dataproduct_type",
)


def _quote_list(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def build_cone_query(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    instruments: tuple[str, ...] | list[str],
    limit: int = 50,
) -> str:
    radius_deg = float(radius_arcsec) / 3600.0
    columns = ", ".join(OBSCORE_COLUMNS)
    return f"""
SELECT TOP {int(limit)} {columns}
FROM {OBSCORE_TABLE}
WHERE instrument_name IN ({_quote_list(instruments)})
  AND 1 = CONTAINS(
      POINT('ICRS', s_ra, s_dec),
      CIRCLE('ICRS', {float(ra_deg)!r}, {float(dec_deg)!r}, {radius_deg!r})
  )
"""


def search_observations(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float = 10.0,
    settings: Settings | None = None,
    limit: int = 50,
) -> tuple[Table, QueryInfo]:
    settings = settings or load_settings()
    query = build_cone_query(ra_deg, dec_deg, radius_arcsec, settings.eso_instruments, limit)
    table = TAPService(settings.eso_tap_url).search(query).to_table()
    info = QueryInfo(
        service=SERVICE_NAME,
        tap_url=settings.eso_tap_url,
        table=OBSCORE_TABLE,
        query=query,
        retrieved_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_rows=len(table),
    )
    return table, info


def check_access(settings: Settings | None = None, limit: int = 5) -> Table:
    """Check access using a few HARPS ObsCore records without a spatial filter."""
    settings = settings or load_settings()
    query = (
        f"SELECT TOP {int(limit)} {', '.join(OBSCORE_COLUMNS)} FROM {OBSCORE_TABLE} "
        "WHERE instrument_name = 'HARPS'"
    )
    return TAPService(settings.eso_tap_url).search(query).to_table()

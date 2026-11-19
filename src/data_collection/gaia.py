"""Gaia Archive (ESA) client through astroquery. Stage 1 checks access only.

Gaia stellar parameters will be used in stage 4. Do not convert parallax directly to
1/parallax distance: low significance and systematics can make that estimate unreliable.
"""

from __future__ import annotations

from astropy.table import Table
from astroquery.gaia import Gaia

from src.utils.config import Settings, load_settings

SAMPLE_QUERY_TEMPLATE = """
SELECT TOP {limit}
    source_id, ra, dec, parallax,
    pmra, pmdec, phot_g_mean_mag
FROM {table}
WHERE parallax IS NOT NULL
"""


def build_cone_query(
    ra_deg: float, dec_deg: float, radius_arcsec: float, table: str, limit: int = 50
) -> str:
    """Build an ADQL cone search around ICRS coordinates in degrees for later stages."""
    radius_deg = float(radius_arcsec) / 3600.0
    return f"""
SELECT TOP {int(limit)}
    source_id, ra, dec, parallax, parallax_error, pmra, pmdec,
    phot_g_mean_mag, ruwe
FROM {table}
WHERE 1 = CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {float(ra_deg)!r}, {float(dec_deg)!r}, {radius_deg!r})
)
"""


def check_access(settings: Settings | None = None, limit: int = 10) -> Table:
    """Run a synchronous Gaia query and return results or raise an exception."""
    settings = settings or load_settings()
    query = SAMPLE_QUERY_TEMPLATE.format(limit=int(limit), table=settings.gaia_table)
    job = Gaia.launch_job(query)
    return job.get_results()


def cone_search(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    settings: Settings | None = None,
    limit: int = 50,
) -> Table:
    settings = settings or load_settings()
    query = build_cone_query(ra_deg, dec_deg, radius_arcsec, settings.gaia_table, limit)
    return Gaia.launch_job(query).get_results()

"""SIMBAD TAP/ADQL client: primary designation, coordinates, and cross-identifiers.

Objects are matched using catalog IDs and coordinates, not names. This client retrieves
SIMBAD coordinates and identifiers; the pipeline applies the configured coordinate tolerance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pyvo.dal import TAPService
from pyvo.dal.exceptions import DALProtocolError
from requests import Session as RequestsSession

from src.utils.tables import cell_float, cell_str

# CDS publishes these mirrors while its Strasbourg TAP service is degraded.
SIMBAD_TAP_URLS = (
    "https://simbad.idia.ac.za/simbad/sim-tap",
    "https://simbad.cfa.harvard.edu/simbad/sim-tap",
)
SIMBAD_TIMEOUT_SECONDS = 20

# Catalogs retained as external IDs; other designations are skipped.
_IDENTIFIER_PATTERN = re.compile(
    r"^(?P<catalog>TIC|KIC|EPIC|KOI|TOI|HD|HIP|TYC|Gaia DR[23]|2MASS)(?:\s+|-)(?P<ident>\S.*)$"
)


@dataclass(frozen=True)
class SimbadResult:
    query_name: str
    main_id: str
    ra_deg: float | None
    dec_deg: float | None
    identifiers: tuple[tuple[str, str], ...]
    tap_url: str | None = None


class _TimeoutSession(RequestsSession):
    def request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = SIMBAD_TIMEOUT_SECONDS
        return super().request(method, url, **kwargs)


def parse_identifier(raw: str) -> tuple[str, str] | None:
    """'TIC 150428135' -> ('TIC', '150428135'); 'Gaia DR3 123' -> ('Gaia DR3', '123')."""
    match = _IDENTIFIER_PATTERN.match(raw.strip())
    if match is None:
        return None
    return match["catalog"], match["ident"].strip()


def parse_identifiers(raw_ids: str | None) -> tuple[tuple[str, str], ...]:
    """Parse ident.id values from one string separated by empty entries."""
    if not raw_ids:
        return ()
    seen: dict[tuple[str, str], None] = {}
    for part in raw_ids.split("|"):
        parsed = parse_identifier(part)
        if parsed is not None:
            seen.setdefault(parsed, None)
    return tuple(seen)


_RESOLVE_QUERY = """
SELECT b.main_id, b.ra, b.dec, i2.id AS ident_id
FROM ident AS i1
JOIN basic AS b ON i1.oidref = b.oid
JOIN ident AS i2 ON i2.oidref = b.oid
WHERE i1.id = '{name}'
"""


def query_identifiers(name: str, tap_url: str | None = None) -> SimbadResult:
    """Find an object by a known designation and return coordinates and IDs."""
    escaped = name.replace("'", "''")
    query = _RESOLVE_QUERY.format(name=escaped)
    urls = (tap_url,) if tap_url else SIMBAD_TAP_URLS
    failures: list[str] = []
    successful_queries = 0
    for url in urls:
        try:
            with _TimeoutSession() as session:
                table = TAPService(url, session=session).search(query).to_table()
        except (DALProtocolError, OSError) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            continue
        successful_queries += 1
        if len(table):
            break
    else:
        if successful_queries:
            raise LookupError(f"SIMBAD has no object named {name!r}")
        raise ConnectionError("SIMBAD TAP mirrors failed: " + "; ".join(failures))

    first = table[0]
    idents: list[str] = []
    for row in table:
        value = cell_str(row, "ident_id")
        if value:
            idents.append(value)

    return SimbadResult(
        query_name=name,
        main_id=cell_str(first, "main_id") or name,
        ra_deg=cell_float(first, "ra"),
        dec_deg=cell_float(first, "dec"),
        identifiers=parse_identifiers("|".join(idents)),
        tap_url=url,
    )

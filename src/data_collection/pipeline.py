"""Stage 1 end-to-end data collection pipeline.

    1. NASA Exoplanet Archive: sample planets, host star, and response snapshot.
    2. MAST: star observations and coordinate-verified mission ID matching.
    3. Select and download FITS light curves, skipping existing files.
    4. Validate FITS HDUs, time, flux, errors, and quality flags with Astropy.
    5. Compute SHA-256 and register files in SQLite.
    6. Optional SIMBAD identifiers and ESO spectroscopy metadata.
    7. Generate a checksum manifest.

Repeated runs do not duplicate records or overwrite raw files; changed source files
stop the pipeline with a clear error.

Run:
    python -m src.data_collection.pipeline --host TOI-700 --mission TESS
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from sqlalchemy import func, select

from src.data_collection import eso, mast, nasa_archive, simbad
from src.data_collection.nasa_archive import QueryInfo
from src.database import repository as repo
from src.database.db import connect, init_db, utc_now
from src.database.manifest import export_manifest
from src.database.schema import Planet
from src.utils.config import Settings, load_settings
from src.utils.fits_utils import LightCurveCheck, inspect_lightcurve
from src.utils.hashing import sha256_file
from src.utils.logging_setup import get_logger, setup_logging
from src.utils.paths import ProjectPaths
from src.utils.tables import cell_float, cell_str

NASA_CATALOG = "NASA_HOSTNAME"
NASA_SOURCE = "NASA Exoplanet Archive"
MAST_SOURCE = "MAST"
SIMBAD_SOURCE = "SIMBAD"
ESO_SOURCE = "ESO Science Archive"
MISSION_ID_CATALOG = {"TESS": "TIC", "KEPLER": "KIC", "K2": "EPIC"}
OBSERVATION_TYPE_MAP = {"timeseries": "lightcurve", "spectrum": "spectrum", "image": "image"}
# Cone radius for ESO metadata near the star.
ESO_CONE_RADIUS_ARCSEC = 10.0

log = get_logger("pipeline")


class PipelineError(RuntimeError):
    """The pipeline cannot continue."""


class NoDataError(PipelineError):
    """The archive returned no data needed for the next step."""


@dataclass
class ArchiveClients:
    """Archive access functions, replaced by fakes in tests."""

    fetch_planets: Callable[..., tuple[Table, QueryInfo]] = nasa_archive.fetch_planets
    search_mast: Callable[[str, str], Table] = mast.search_observations
    list_products: Callable[..., Table] = mast.list_lightcurve_products
    download_products: Callable[[Table, Path], Table] = mast.download_products
    search_eso: Callable[..., tuple[Table, QueryInfo]] = eso.search_observations
    resolve_simbad: Callable[[str], simbad.SimbadResult] = simbad.query_identifiers


@dataclass
class Stage1Result:
    host: str
    mission: str
    started_at: str = field(default_factory=utc_now)
    star_id: int | None = None
    n_planets: int = 0
    snapshot_csv: str | None = None
    observation_ids: dict[str, int] = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)
    linked_identifiers: list[str] = field(default_factory=list)
    eso_observations: int = 0
    manifest_rows: int = 0
    db_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.star_id is not None
            and bool(self.files)
            and not self.errors
            and all(f["validation"] != "error" for f in self.files)
        )


# --- Helpers ---------------------------------------------------------


def _separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
    b = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")
    return float(a.separation(b).to(u.arcsec).value)


def _abs_or_none(value: float | None) -> float | None:
    return None if value is None else abs(value)


def _validation_notes(check: LightCurveCheck) -> str | None:
    notes = [*check.errors, *check.warnings]
    return "; ".join(notes) if notes else None


def _record_measurement(
    session,
    *,
    entity_type: str,
    entity_id: int,
    name: str,
    value: float | None,
    unit: str,
    err_plus: float | None,
    err_minus: float | None,
    reference: str | None,
    source_table: str,
) -> None:
    # Store the absolute value of NASA's negative lower uncertainty (pl_radeerr2).
    repo.insert_measurement(
        session,
        entity_type=entity_type,
        entity_id=entity_id,
        parameter_name=name,
        value=value,
        unit=unit,
        error_plus=_abs_or_none(err_plus),
        error_minus=_abs_or_none(err_minus),
        reference=reference,
        source_archive=NASA_SOURCE,
        source_table=source_table,
    )


def _register_nasa_table(session, table: Table, info: QueryInfo) -> dict[str, int]:
    """Store NASA stars, planets, and measurements; return {hostname: star_id}."""
    star_ids: dict[str, int] = {}
    for row in table:
        hostname = cell_str(row, "hostname")
        planet_name = cell_str(row, "pl_name")
        if not hostname or not planet_name:
            continue

        star_id = repo.get_or_create_star(
            session,
            catalog=NASA_CATALOG,
            external_id=hostname,
            name=hostname,
            ra=cell_float(row, "ra"),
            dec=cell_float(row, "dec"),
            coord_epoch=None,
            coord_source=f"{NASA_SOURCE}:{info.table}",
            source_archive=NASA_SOURCE,
            retrieved_at=info.retrieved_at,
        )
        star_ids[hostname] = star_id

        disc_year = cell_float(row, "disc_year")
        planet_id = repo.upsert_planet(
            session,
            star_id=star_id,
            planet_name=planet_name,
            status="CONFIRMED",  # pscomppars contains only confirmed planets
            status_source=f"{NASA_SOURCE}:{info.table}",
            discovery_method=cell_str(row, "discoverymethod"),
            discovery_year=int(disc_year) if disc_year is not None else None,
        )

        measurements = (
            (
                "planet",
                planet_id,
                "pl_rade",
                cell_float(row, "pl_rade"),
                "R_earth",
                cell_float(row, "pl_radeerr1"),
                cell_float(row, "pl_radeerr2"),
                "pl_rade_reflink",
            ),
            (
                "planet",
                planet_id,
                "pl_bmasse",
                cell_float(row, "pl_bmasse"),
                "M_earth",
                cell_float(row, "pl_bmasseerr1"),
                cell_float(row, "pl_bmasseerr2"),
                "pl_bmasse_reflink",
            ),
            (
                "planet",
                planet_id,
                "pl_orbper",
                cell_float(row, "pl_orbper"),
                "d",
                None,
                None,
                "pl_orbper_reflink",
            ),
            (
                "star",
                star_id,
                "st_teff",
                cell_float(row, "st_teff"),
                "K",
                None,
                None,
                "st_teff_reflink",
            ),
            (
                "star",
                star_id,
                "st_rad",
                cell_float(row, "st_rad"),
                "R_sun",
                None,
                None,
                "st_rad_reflink",
            ),
        )
        for (
            entity_type,
            entity_id,
            name,
            value,
            unit,
            err_plus,
            err_minus,
            ref_column,
        ) in measurements:
            _record_measurement(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                name=name,
                value=value,
                unit=unit,
                err_plus=err_plus,
                err_minus=err_minus,
                reference=nasa_archive.strip_html(cell_str(row, ref_column)),
                source_table=info.table,
            )
    return star_ids


_MISSION_NUMBER = re.compile(r"^(?:TIC|KIC|EPIC)?\s*(\d+)$", re.IGNORECASE)


def _mission_number(target_name: str | None) -> str | None:
    """'150428135' and 'TIC 150428135' -> '150428135'. Other names are not parsed."""
    if not target_name:
        return None
    match = _MISSION_NUMBER.match(target_name.strip())
    return match.group(1) if match else None


def _link_mission_identifier(
    session,
    *,
    star_id: int,
    mission: str,
    target_name: str | None,
    obs_ra: float | None,
    obs_dec: float | None,
    star_ra: float | None,
    star_dec: float | None,
    radius_arcsec: float,
    result: Stage1Result,
) -> None:
    """Link target_name's TIC/KIC/EPIC ID to a star only when coordinates match.

    The identifier alone is insufficient; coordinates confirm the match.
    """
    catalog = MISSION_ID_CATALOG.get(mission.upper())
    number = _mission_number(target_name)
    if catalog is None or number is None:
        return
    target_name = number
    if None in (obs_ra, obs_dec, star_ra, star_dec):
        result.warnings.append(f"{catalog} {target_name}: coordinates unavailable; no link created")
        return

    separation = _separation_arcsec(star_ra, star_dec, obs_ra, obs_dec)
    if separation > radius_arcsec:
        result.warnings.append(
            f'{catalog} {target_name}: MAST coordinates differ by {separation:.1f}", '
            f'no link created (tolerance {radius_arcsec}")'
        )
        return
    try:
        added = repo.link_external_id(
            session,
            star_id,
            catalog,
            target_name,
            method="mast_target",
            confidence="high",
            separation_arcsec=separation,
            source_archive=MAST_SOURCE,
            retrieved_at=utc_now(),
        )
    except repo.IdentityConflictError as exc:
        result.warnings.append(str(exc))
        return
    if added:
        result.linked_identifiers.append(f"{catalog} {target_name}")


# --- Pipeline steps -------------------------------------------------------------------


def _step_nasa(
    clients: ArchiveClients,
    settings: Settings,
    host: str,
    planet_limit: int,
    result: Stage1Result,
) -> None:
    paths = settings.paths
    log.info("Step 1/7: NASA Exoplanet Archive — sample planets (%d)", planet_limit)
    sample, info = clients.fetch_planets(limit=planet_limit, settings=settings)
    csv_path, _ = nasa_archive.save_snapshot(sample, info, paths)
    result.snapshot_csv = paths.relative(csv_path)

    with connect(paths.db_path) as session:
        star_ids = _register_nasa_table(session, sample, info)

        if host not in star_ids:
            log.info("Star %s is absent from the sample; querying it separately", host)
            targeted, targeted_info = clients.fetch_planets(
                limit=None, hostname=host, require_radius=False, settings=settings
            )
            nasa_archive.save_snapshot(targeted, targeted_info, paths, update_latest=False)
            star_ids.update(_register_nasa_table(session, targeted, targeted_info))

        if host not in star_ids:
            raise NoDataError(
                f"NASA Exoplanet Archive has no star with hostname = '{host}'. "
                "Check the spelling against the hostname column."
            )
        result.star_id = star_ids[host]
        result.n_planets = int(
            session.scalar(
                select(func.count()).select_from(Planet).where(Planet.star_id == result.star_id)
            )
        )
    log.info("Star %s: star_id=%s, planets in database: %d", host, result.star_id, result.n_planets)


def _step_mast_observations(
    clients: ArchiveClients,
    settings: Settings,
    host: str,
    mission: str,
    result: Stage1Result,
) -> Table:
    log.info("Step 2/7: MAST — observations of %s in mission %s", host, mission)
    observations = clients.search_mast(host, mission)
    if len(observations) == 0:
        raise NoDataError(f"MAST found no observations of {host} in mission {mission}.")

    rows = mast.observation_rows(observations)
    with connect(settings.paths.db_path) as session:
        star_ra, star_dec = repo.get_star_coords(session, result.star_id)
        for row in rows:
            if row["archive_id"] is None:
                continue
            obs_type = row["observation_type"]
            observation_id = repo.upsert_observation(
                session,
                star_id=result.star_id,
                archive=MAST_SOURCE,
                archive_id=row["archive_id"],
                mission=row["mission"] or mission,
                instrument=row["instrument"],
                observation_type=OBSERVATION_TYPE_MAP.get(obs_type or "", obs_type),
                sequence_number=row["sequence_number"],
                target_name=row["target_name"],
                t_min_mjd=row["t_min_mjd"],
                t_max_mjd=row["t_max_mjd"],
                exptime_s=row["exptime_s"],
            )
            result.observation_ids[row["archive_id"]] = observation_id
            _link_mission_identifier(
                session,
                star_id=result.star_id,
                mission=row["mission"] or mission,
                target_name=row["target_name"],
                obs_ra=row["ra"],
                obs_dec=row["dec"],
                star_ra=star_ra,
                star_dec=star_dec,
                radius_arcsec=settings.match_radius_arcsec,
                result=result,
            )
    log.info("MAST: %d observations, %d registered", len(rows), len(result.observation_ids))
    return observations


def _step_download(
    clients: ArchiveClients,
    settings: Settings,
    host: str,
    mission: str,
    observations: Table,
    max_products: int,
    result: Stage1Result,
) -> tuple[Table, set[str]]:
    """Return selected products and names of files downloaded in this run."""
    log.info("Step 3/7: select and download up to %d light curves", max_products)
    products = clients.list_products(
        observations,
        subgroups=settings.lightcurve_subgroups,
        max_products=max_products,
        settings=settings,
    )
    if len(products) == 0:
        raise NoDataError(
            f"No science FITS light curves for {host} "
            f"(subgroups: {', '.join(settings.lightcurve_subgroups)}, MRP)."
        )

    dest = settings.paths.raw_dir(mission, host)
    dest.mkdir(parents=True, exist_ok=True)

    names = [str(n) for n in products["productFilename"]]
    missing = [i for i, name in enumerate(names) if not (dest / name).exists()]
    downloaded: set[str] = set()

    if missing:
        log.info("Downloading %d files to %s", len(missing), dest)
        manifest = clients.download_products(products[missing], dest)
        for row in manifest:
            local = Path(str(row["Local Path"]))
            status = str(row["Status"])
            if status == "ERROR":
                result.errors.append(f"Download of {local.name} failed: {row['Message']}")
                continue
            if status == "COMPLETE" and local.exists():
                # Mark raw files read-only so accidental overwrites fail.
                os.chmod(local, 0o444)
                downloaded.add(local.name)
    else:
        log.info("All files already exist in %s; no download needed", dest)
    return products, downloaded


def _step_inspect_and_register(
    clients: ArchiveClients,
    settings: Settings,
    mission: str,
    host: str,
    products: Table,
    downloaded: set[str],
    result: Stage1Result,
) -> None:
    paths: ProjectPaths = settings.paths
    dest = paths.raw_dir(mission, host)

    for product in products:
        name = cell_str(product, "productFilename")
        observation_key = cell_str(product, "obsID")
        path = dest / name
        if not path.exists():
            result.errors.append(f"File missing after download: {name}")
            continue

        log.info("Step 4/7: validate FITS %s", name)
        check = inspect_lightcurve(path)
        report_path = paths.fits_reports / f"{path.stem}.json"
        report_path.write_text(
            json.dumps(check.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        for line in check.warnings:
            log.warning("%s: %s", name, line)
        for line in check.errors:
            log.error("%s: %s", name, line)

        log.info("Step 5/7: SHA-256 and register %s", name)
        digest = sha256_file(path)
        size = path.stat().st_size
        observation_id = result.observation_ids.get(observation_key or "")
        if observation_id is None:
            raise PipelineError(f"Product {name} references missing observation {observation_key}.")

        with connect(paths.db_path) as session:
            repo.upsert_file(
                session,
                observation_id=observation_id,
                file_path=paths.relative(path),
                file_format="fits",
                file_size=size,
                sha256=digest,
                product_filename=name,
                product_type=cell_str(product, "productType"),
                product_subgroup=cell_str(product, "productSubGroupDescription"),
                source_uri=cell_str(product, "dataURI"),
                downloaded_at=utc_now() if name in downloaded else None,
                validation_status=check.status,
                validation_notes=_validation_notes(check),
            )

        result.files.append(
            {
                "file_path": paths.relative(path),
                "product_filename": name,
                "sha256": digest,
                "size": size,
                "status": "downloaded" if name in downloaded else "reused",
                "validation": check.status,
                "time_start_jd": check.time_start_jd,
                "time_end_jd": check.time_end_jd,
                "n_cadences": check.n_cadences,
            }
        )
        if check.status == "error":
            result.errors.append(f"{name}: critical FITS errors; see report {report_path.name}")


def _step_simbad(
    clients: ArchiveClients,
    settings: Settings,
    host: str,
    star_ra: float | None,
    star_dec: float | None,
    result: Stage1Result,
) -> None:
    """Link SIMBAD identifiers only when coordinates match within tolerance."""
    log.info("Step 6/7: SIMBAD identifiers for %s", host)
    try:
        sim = clients.resolve_simbad(host)
    except Exception as exc:  # Optional step: failures do not stop the pipeline.
        result.warnings.append(f"SIMBAD skipped: {type(exc).__name__}: {exc}")
        return

    if None in (star_ra, star_dec) or sim.ra_deg is None or sim.dec_deg is None:
        result.warnings.append("SIMBAD: no coordinates for matching; identifiers were not linked")
        return
    separation = _separation_arcsec(star_ra, star_dec, sim.ra_deg, sim.dec_deg)
    if separation > settings.match_radius_arcsec:
        result.warnings.append(
            f"SIMBAD: {sim.main_id} is {separation:.1f} arcsec away; IDs were not linked"
        )
        return

    with connect(settings.paths.db_path) as session:
        for catalog, ident in sim.identifiers:
            try:
                added = repo.link_external_id(
                    session,
                    result.star_id,
                    catalog,
                    ident,
                    method="simbad",
                    confidence="high",
                    separation_arcsec=separation,
                    source_archive=SIMBAD_SOURCE,
                    retrieved_at=utc_now(),
                )
            except repo.IdentityConflictError as exc:
                result.warnings.append(str(exc))
                continue
            if added:
                result.linked_identifiers.append(f"{catalog} {ident}")


def _step_eso(
    clients: ArchiveClients,
    settings: Settings,
    host: str,
    star_ra: float | None,
    star_dec: float | None,
    result: Stage1Result,
) -> None:
    """Retrieve ESO spectrum metadata near the star without downloading files."""
    log.info("Step 6/7: ESO spectroscopy metadata near %s", host)
    if None in (star_ra, star_dec):
        result.warnings.append("ESO: no coordinates; search skipped")
        return
    try:
        table, _info = clients.search_eso(
            star_ra, star_dec, radius_arcsec=ESO_CONE_RADIUS_ARCSEC, settings=settings
        )
    except Exception as exc:  # Optional step: failures do not stop the pipeline.
        result.warnings.append(f"ESO skipped: {type(exc).__name__}: {exc}")
        return

    with connect(settings.paths.db_path) as session:
        for row in table:
            archive_id = cell_str(row, "obs_id")
            if archive_id is None:
                continue
            repo.upsert_observation(
                session,
                star_id=result.star_id,
                archive=ESO_SOURCE,
                archive_id=archive_id,
                mission=cell_str(row, "obs_collection") or "ESO",
                instrument=cell_str(row, "instrument_name"),
                observation_type=cell_str(row, "dataproduct_type"),
                target_name=cell_str(row, "target_name"),
                t_min_mjd=cell_float(row, "t_min"),
                exptime_s=cell_float(row, "t_exptime"),
            )
            result.eso_observations += 1
    log.info("ESO: %d metadata records (no downloads)", result.eso_observations)


def _step_optional_catalogs(
    clients: ArchiveClients,
    settings: Settings,
    host: str,
    with_simbad: bool,
    with_eso: bool,
    result: Stage1Result,
) -> None:
    if not (with_simbad or with_eso):
        return
    with connect(settings.paths.db_path) as session:
        star_ra, star_dec = repo.get_star_coords(session, result.star_id)
    if with_simbad:
        _step_simbad(clients, settings, host, star_ra, star_dec, result)
    if with_eso:
        _step_eso(clients, settings, host, star_ra, star_dec, result)


def _step_manifest(settings: Settings, result: Stage1Result) -> None:
    paths = settings.paths
    log.info("Step 7/7: manifest and final counts")
    with connect(paths.db_path) as session:
        result.manifest_rows = export_manifest(session, paths.manifest_path)
        result.db_counts = repo.table_counts(session)


def _write_run_report(result: Stage1Result, paths: ProjectPaths) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = paths.reports / f"stage1_run_{stamp}.json"
    out.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return out


# --- Entry point ---------------------------------------------------------------------


def run_stage1(
    host: str = "TOI-700",
    *,
    mission: str | None = None,
    planet_limit: int | None = None,
    max_products: int | None = None,
    with_simbad: bool = False,
    with_eso: bool = False,
    settings: Settings | None = None,
    clients: ArchiveClients | None = None,
) -> Stage1Result:
    settings = settings or load_settings()
    clients = clients or ArchiveClients()
    mission = mission or settings.default_mission
    planet_limit = planet_limit or settings.planet_sample_limit
    max_products = max_products or settings.max_products_per_target

    paths = settings.paths
    paths.ensure_dirs()
    init_db(paths.db_path)

    result = Stage1Result(host=host, mission=mission)
    log.info("Starting stage 1: target=%s, mission=%s", host, mission)

    _step_nasa(clients, settings, host, planet_limit, result)
    observations = _step_mast_observations(clients, settings, host, mission, result)
    products, downloaded = _step_download(
        clients, settings, host, mission, observations, max_products, result
    )
    _step_inspect_and_register(clients, settings, mission, host, products, downloaded, result)
    _step_optional_catalogs(clients, settings, host, with_simbad, with_eso, result)
    _step_manifest(settings, result)

    report = _write_run_report(result, paths)
    log.info("Run report: %s", paths.relative(report))
    return result


def format_summary(result: Stage1Result) -> str:
    lines = [
        f"Target: {result.host} | mission: {result.mission} | star_id: {result.star_id}",
        f"Planets in database for star: {result.n_planets}",
        f"MAST observations: {len(result.observation_ids)} | ESO: {result.eso_observations}",
    ]
    for f in result.files:
        lines.append(
            f"File {f['product_filename']}: {f['status']}, validation={f['validation']}, "
            f"cadences={f['n_cadences']}, sha256={f['sha256'][:16]}…"
        )
    if result.linked_identifiers:
        lines.append("New ID links: " + ", ".join(result.linked_identifiers))
    lines += [f"Warning: {w}" for w in result.warnings]
    lines += [f"Error: {e}" for e in result.errors]
    lines.append(f"Result: {'OK' if result.ok else 'ISSUES FOUND'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IERAS stage 1 end-to-end pipeline")
    parser.add_argument("--host", default="TOI-700", help="host star name as recorded in hostname")
    parser.add_argument(
        "--mission", default=None, help="mission: TESS, Kepler, K2 (default from settings)"
    )
    parser.add_argument("--planet-limit", type=int, default=None, help="NASA sample size")
    parser.add_argument(
        "--max-products", type=int, default=None, help="maximum light-curve files to download"
    )
    parser.add_argument("--with-simbad", action="store_true", help="add SIMBAD identifiers")
    parser.add_argument(
        "--with-eso", action="store_true", help="add ESO metadata without downloads"
    )
    parser.add_argument(
        "--root", default=None, help="project root (defaults to repository directory)"
    )
    args = parser.parse_args(argv)

    settings = load_settings(root=args.root)
    setup_logging(settings.paths.logs, settings.log_level)

    try:
        result = run_stage1(
            args.host,
            mission=args.mission,
            planet_limit=args.planet_limit,
            max_products=args.max_products,
            with_simbad=args.with_simbad,
            with_eso=args.with_eso,
            settings=settings,
        )
    except (PipelineError, repo.IdentityConflictError, repo.RawFileChangedError) as exc:
        log.error("Stopped: %s", exc)
        return 1
    except Exception as exc:  # Show the cause and guidance for network and other unexpected errors.
        log.error("Unexpected error: %s: %s", type(exc).__name__, exc)
        log.debug("Traceback:", exc_info=True)
        log.error(
            "If this is a network error, check internet access, VPN, and proxy settings, "
            "then retry. Previously stored data will remain intact."
        )
        return 2

    print(format_summary(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())

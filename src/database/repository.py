"""Idempotent metadata operations through SQLAlchemy sessions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from src.database.db import utc_now
from src.database.schema import (
    ExternalId,
    FileRecord,
    Observation,
    ParameterMeasurement,
    Planet,
    Star,
)


class IdentityConflictError(RuntimeError):
    """An identifier is linked to a different star and cannot be reassigned."""


class RawFileChangedError(RuntimeError):
    """A registered raw file has changed its SHA-256 checksum."""


def find_star_by_external_id(session: Session, catalog: str, external_id: str) -> int | None:
    return session.scalar(
        select(ExternalId.star_id).where(
            ExternalId.catalog_name == catalog,
            ExternalId.external_id == str(external_id),
        )
    )


def create_star(
    session: Session,
    *,
    name: str,
    ra: float | None,
    dec: float | None,
    coord_epoch: str | None,
    coord_source: str | None,
) -> int:
    now = utc_now()
    star = Star(
        name=name,
        ra=ra,
        dec=dec,
        coord_epoch=coord_epoch,
        coord_source=coord_source,
        created_at=now,
        updated_at=now,
    )
    session.add(star)
    session.flush()
    return star.star_id


def link_external_id(
    session: Session,
    star_id: int,
    catalog: str,
    external_id: str,
    *,
    method: str,
    confidence: str | None = None,
    separation_arcsec: float | None = None,
    source_archive: str | None = None,
    retrieved_at: str | None = None,
) -> bool:
    """Link an external ID to a star; return True if the link is new."""
    external_id = str(external_id)
    existing = find_star_by_external_id(session, catalog, external_id)
    if existing is not None:
        if existing != star_id:
            raise IdentityConflictError(
                f"{catalog} {external_id} is already linked to star_id={existing}, "
                f"but was requested for star_id={star_id}. Review the cross-match manually."
            )
        return False
    session.add(
        ExternalId(
            star_id=star_id,
            catalog_name=catalog,
            external_id=external_id,
            match_method=method,
            match_confidence=confidence,
            match_separation_arcsec=separation_arcsec,
            source_archive=source_archive,
            retrieved_at=retrieved_at or utc_now(),
        )
    )
    session.flush()
    return True


def get_or_create_star(
    session: Session,
    *,
    catalog: str,
    external_id: str,
    name: str,
    ra: float | None,
    dec: float | None,
    coord_epoch: str | None,
    coord_source: str | None,
    source_archive: str,
    retrieved_at: str,
) -> int:
    """Find a star by catalog ID or create one."""
    star_id = find_star_by_external_id(session, catalog, external_id)
    if star_id is not None:
        return star_id
    star_id = create_star(
        session, name=name, ra=ra, dec=dec, coord_epoch=coord_epoch, coord_source=coord_source
    )
    link_external_id(
        session,
        star_id,
        catalog,
        external_id,
        method="native",
        confidence="high",
        source_archive=source_archive,
        retrieved_at=retrieved_at,
    )
    return star_id


def get_star_coords(session: Session, star_id: int) -> tuple[float | None, float | None]:
    star = session.get(Star, star_id)
    if star is None:
        raise KeyError(f"No star with star_id={star_id}")
    return star.ra, star.dec


def upsert_planet(
    session: Session,
    *,
    star_id: int,
    planet_name: str,
    status: str | None,
    status_source: str | None,
    discovery_method: str | None,
    discovery_year: int | None,
) -> int:
    now = utc_now()
    planet = session.scalar(select(Planet).where(Planet.planet_name == planet_name))
    if planet is None:
        planet = Planet(
            star_id=star_id,
            planet_name=planet_name,
            status=status,
            status_source=status_source,
            discovery_method=discovery_method,
            discovery_year=discovery_year,
            created_at=now,
            updated_at=now,
        )
        session.add(planet)
        session.flush()
        return planet.planet_id
    if planet.star_id != star_id:
        raise IdentityConflictError(
            f"Planet {planet_name} belongs to star_id={planet.star_id}, not {star_id}."
        )
    planet.status = status
    planet.status_source = status_source
    planet.discovery_method = discovery_method
    planet.discovery_year = discovery_year
    planet.updated_at = now
    return planet.planet_id


def upsert_observation(
    session: Session,
    *,
    star_id: int,
    archive: str,
    archive_id: str,
    mission: str,
    instrument: str | None = None,
    observation_type: str | None = None,
    sequence_number: str | None = None,
    target_name: str | None = None,
    t_min_mjd: float | None = None,
    t_max_mjd: float | None = None,
    exptime_s: float | None = None,
) -> int:
    archive_id = str(archive_id)
    now = utc_now()
    observation = session.scalar(
        select(Observation).where(
            Observation.archive == archive, Observation.archive_id == archive_id
        )
    )
    if observation is None:
        observation = Observation(
            star_id=star_id,
            archive=archive,
            archive_id=archive_id,
            mission=mission,
            instrument=instrument,
            observation_type=observation_type,
            sequence_number=sequence_number,
            target_name=target_name,
            t_min_mjd=t_min_mjd,
            t_max_mjd=t_max_mjd,
            exptime_s=exptime_s,
            retrieved_at=now,
        )
        session.add(observation)
        session.flush()
        return observation.observation_id
    if observation.star_id != star_id:
        raise IdentityConflictError(
            f"Observation {archive}:{archive_id} belongs to star_id={observation.star_id}."
        )
    observation.mission = mission
    observation.instrument = instrument
    observation.observation_type = observation_type
    observation.sequence_number = sequence_number
    observation.target_name = target_name
    observation.t_min_mjd = t_min_mjd
    observation.t_max_mjd = t_max_mjd
    observation.exptime_s = exptime_s
    observation.retrieved_at = now
    return observation.observation_id


def find_file_by_path(session: Session, file_path: str) -> FileRecord | None:
    return session.scalar(select(FileRecord).where(FileRecord.file_path == file_path))


def upsert_file(
    session: Session,
    *,
    observation_id: int,
    file_path: str,
    file_format: str | None,
    file_size: int,
    sha256: str,
    product_filename: str | None,
    product_type: str | None,
    product_subgroup: str | None,
    source_uri: str | None,
    downloaded_at: str | None,
    validation_status: str,
    validation_notes: str | None,
) -> int:
    """Register a file, refusing changes to an existing raw file's checksum."""
    existing = find_file_by_path(session, file_path)
    if existing is not None:
        if existing.sha256 != sha256:
            raise RawFileChangedError(
                f"File {file_path} changed: stored SHA-256 {existing.sha256}; "
                f"disk SHA-256 {sha256}. Raw files must not be overwritten. "
                "Restore the file or register a new version."
            )
        existing.validation_status = validation_status
        existing.validation_notes = validation_notes
        existing.file_size = file_size
        return existing.file_id

    record = FileRecord(
        observation_id=observation_id,
        file_path=file_path,
        file_format=file_format,
        file_size=file_size,
        sha256=sha256,
        product_filename=product_filename,
        product_type=product_type,
        product_subgroup=product_subgroup,
        source_uri=source_uri,
        downloaded_at=downloaded_at,
        registered_at=utc_now(),
        validation_status=validation_status,
        validation_notes=validation_notes,
    )
    session.add(record)
    session.flush()
    return record.file_id


def insert_measurement(
    session: Session,
    *,
    entity_type: str,
    entity_id: int,
    parameter_name: str,
    value: float | None,
    unit: str | None,
    error_plus: float | None,
    error_minus: float | None,
    reference: str | None,
    source_archive: str,
    source_table: str | None,
) -> bool:
    """Add a measurement unless the same value and source are already stored."""
    if value is None:
        return False
    numeric_value = float(value)
    existing = session.scalar(
        select(ParameterMeasurement.measurement_id).where(
            ParameterMeasurement.entity_type == entity_type,
            ParameterMeasurement.entity_id == entity_id,
            ParameterMeasurement.parameter_name == parameter_name,
            ParameterMeasurement.source_archive == source_archive,
            func.coalesce(ParameterMeasurement.source_table, "") == (source_table or ""),
            func.coalesce(ParameterMeasurement.reference, "") == (reference or ""),
            ParameterMeasurement.value == numeric_value,
        )
    )
    if existing is not None:
        return False
    session.add(
        ParameterMeasurement(
            entity_type=entity_type,
            entity_id=entity_id,
            parameter_name=parameter_name,
            value=numeric_value,
            unit=unit,
            error_plus=error_plus,
            error_minus=error_minus,
            reference=reference,
            source_archive=source_archive,
            source_table=source_table,
            retrieved_at=utc_now(),
        )
    )
    session.flush()
    return True


TABLES = {
    "stars": Star,
    "planets": Planet,
    "external_ids": ExternalId,
    "observations": Observation,
    "files": FileRecord,
    "parameter_measurements": ParameterMeasurement,
}


def table_counts(session: Session) -> dict[str, int]:
    return {
        name: int(session.scalar(select(func.count()).select_from(model)))
        for name, model in TABLES.items()
    }


def as_dict(row: Any | None) -> dict[str, Any] | None:
    if row is None:
        return None
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    if isinstance(row, Mapping):
        return dict(row)
    mapper = inspect(row).mapper
    return {column.key: getattr(row, column.key) for column in mapper.columns}

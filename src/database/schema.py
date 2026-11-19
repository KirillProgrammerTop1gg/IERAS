"""SQLAlchemy ORM schema for the local metadata database.

Table and column names match the original SQLite schema so existing databases remain
readable. Measurement identity uses the same SQLite expression index as stage 1.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    pass


class SchemaMeta(Base):
    __tablename__ = "schema_meta"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class Star(Base):
    __tablename__ = "stars"

    star_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    ra: Mapped[float | None] = mapped_column(Float)
    dec: Mapped[float | None] = mapped_column(Float)
    coord_epoch: Mapped[str | None] = mapped_column(Text)
    coord_source: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


class Planet(Base):
    __tablename__ = "planets"

    planet_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    star_id: Mapped[int] = mapped_column(ForeignKey("stars.star_id"), nullable=False)
    planet_name: Mapped[str | None] = mapped_column(Text, unique=True)
    status: Mapped[str | None] = mapped_column(Text)
    status_source: Mapped[str | None] = mapped_column(Text)
    discovery_method: Mapped[str | None] = mapped_column(Text)
    discovery_year: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


class ExternalId(Base):
    __tablename__ = "external_ids"
    __table_args__ = (
        UniqueConstraint("catalog_name", "external_id"),
        Index("idx_external_ids_star", "star_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    star_id: Mapped[int] = mapped_column(ForeignKey("stars.star_id"), nullable=False)
    catalog_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    match_method: Mapped[str] = mapped_column(
        Text, nullable=False, default="native", server_default="native"
    )
    match_confidence: Mapped[str | None] = mapped_column(Text)
    match_separation_arcsec: Mapped[float | None] = mapped_column(Float)
    source_archive: Mapped[str | None] = mapped_column(Text)
    retrieved_at: Mapped[str | None] = mapped_column(Text)


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("archive", "archive_id"),
        Index("idx_observations_star", "star_id"),
    )

    observation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    star_id: Mapped[int] = mapped_column(ForeignKey("stars.star_id"), nullable=False)
    archive: Mapped[str] = mapped_column(Text, nullable=False)
    archive_id: Mapped[str] = mapped_column(Text, nullable=False)
    mission: Mapped[str] = mapped_column(Text, nullable=False)
    instrument: Mapped[str | None] = mapped_column(Text)
    observation_type: Mapped[str | None] = mapped_column(Text)
    sequence_number: Mapped[str | None] = mapped_column(Text)
    target_name: Mapped[str | None] = mapped_column(Text)
    t_min_mjd: Mapped[float | None] = mapped_column(Float)
    t_max_mjd: Mapped[float | None] = mapped_column(Float)
    exptime_s: Mapped[float | None] = mapped_column(Float)
    retrieved_at: Mapped[str] = mapped_column(Text, nullable=False)


class FileRecord(Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("observation_id", "product_filename"),
        Index("idx_files_observation", "observation_id"),
    )

    file_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("observations.observation_id"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    file_format: Mapped[str | None] = mapped_column(Text)
    file_size: Mapped[int | None] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    product_filename: Mapped[str | None] = mapped_column(Text)
    product_type: Mapped[str | None] = mapped_column(Text)
    product_subgroup: Mapped[str | None] = mapped_column(Text)
    source_uri: Mapped[str | None] = mapped_column(Text)
    downloaded_at: Mapped[str | None] = mapped_column(Text)
    registered_at: Mapped[str] = mapped_column(Text, nullable=False)
    validation_status: Mapped[str | None] = mapped_column(Text)
    validation_notes: Mapped[str | None] = mapped_column(Text)


class ParameterMeasurement(Base):
    __tablename__ = "parameter_measurements"

    measurement_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    parameter_name: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(Text)
    error_plus: Mapped[float | None] = mapped_column(Float)
    error_minus: Mapped[float | None] = mapped_column(Float)
    reference: Mapped[str | None] = mapped_column(Text)
    source_archive: Mapped[str] = mapped_column(Text, nullable=False)
    source_table: Mapped[str | None] = mapped_column(Text)
    retrieved_at: Mapped[str] = mapped_column(Text, nullable=False)


Index(
    "idx_measurements_entity",
    ParameterMeasurement.entity_type,
    ParameterMeasurement.entity_id,
    ParameterMeasurement.parameter_name,
)
Index(
    "ux_measurements_dedup",
    ParameterMeasurement.entity_type,
    ParameterMeasurement.entity_id,
    ParameterMeasurement.parameter_name,
    ParameterMeasurement.source_archive,
    func.coalesce(ParameterMeasurement.source_table, ""),
    func.coalesce(ParameterMeasurement.reference, ""),
    ParameterMeasurement.value,
    unique=True,
)

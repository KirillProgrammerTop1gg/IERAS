"""SQLite tests for schema, constraints, idempotent writes, manifests, and checksums."""

import sqlite3

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError, InvalidRequestError

from src.database import repository as repo
from src.database.db import connect, init_db, schema_version
from src.database.manifest import export_manifest, verify_files
from src.database.schema import ExternalId, ParameterMeasurement, Star
from src.utils.hashing import sha256_file
from tests.fakes import make_tess_like_lc

EXPECTED_TABLES = {
    "schema_meta",
    "stars",
    "planets",
    "external_ids",
    "observations",
    "files",
    "parameter_measurements",
}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "metadata" / "exoplanet.db"
    init_db(path)
    return path


def _star(session, name="TOI-700", catalog="NASA_HOSTNAME", ext=None):
    return repo.get_or_create_star(
        session,
        catalog=catalog,
        external_id=ext or name,
        name=name,
        ra=97.1,
        dec=-65.6,
        coord_epoch=None,
        coord_source="test",
        source_archive="test",
        retrieved_at="2026-01-01T00:00:00Z",
    )


def test_init_creates_all_tables_and_version(db):
    with connect(db) as session:
        names = set(inspect(session.bind).get_table_names())
    assert EXPECTED_TABLES <= names
    assert schema_version(db) == 1


def test_init_is_idempotent_and_keeps_data(db):
    with connect(db) as session:
        _star(session)
    init_db(db)
    init_db(db)
    with connect(db) as session:
        assert repo.table_counts(session)["stars"] == 1


def test_init_preserves_existing_stage1_rows(tmp_path):
    path = tmp_path / "existing.db"
    with sqlite3.connect(path) as legacy:
        legacy.executescript(
            """
            CREATE TABLE stars (
                star_id INTEGER PRIMARY KEY, name TEXT NOT NULL, ra REAL, dec REAL,
                coord_epoch TEXT, coord_source TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO stars VALUES (
                7, 'TOI-700', 97.1, -65.6, NULL, 'NASA', '2026-01-01T00:00:00Z',
                '2026-01-01T00:00:00Z'
            );
            """
        )

    init_db(path)
    with connect(path) as session:
        star = session.get(Star, 7)
        assert star.name == "TOI-700"
        assert repo.table_counts(session)["stars"] == 1


def test_connection_closes_and_enables_foreign_keys(db):
    with connect(db) as session:
        assert session.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    with pytest.raises(InvalidRequestError):
        session.execute(select(1))  # session is closed


def test_foreign_key_is_enforced(db):
    with pytest.raises(IntegrityError), connect(db) as session:
        session.execute(
            text(
                "INSERT INTO planets (star_id, planet_name, created_at, updated_at) "
                "VALUES (999, 'ghost', 'x', 'x')"
            )
        )


def test_failed_transaction_is_rolled_back(db):
    with pytest.raises(RuntimeError), connect(db) as session:
        _star(session, name="temp")
        raise RuntimeError("boom")
    with connect(db) as session:
        assert repo.table_counts(session)["stars"] == 0


def test_get_or_create_star_is_idempotent(db):
    with connect(db) as session:
        first = _star(session)
        second = _star(session)
        assert first == second
        assert repo.table_counts(session)["stars"] == 1
        assert repo.table_counts(session)["external_ids"] == 1


def test_same_name_different_catalog_ids_are_different_stars(db):
    """Names do not merge objects: distinct IDs remain distinct stars even with the same name."""
    with connect(db) as session:
        a = _star(session, name="Alpha", ext="A-1")
        b = _star(session, name="Alpha", ext="A-2")
        assert a != b


def test_external_id_cannot_point_to_two_stars(db):
    with connect(db) as session:
        a = _star(session, name="Alpha", ext="A-1")
        b = _star(session, name="Beta", ext="B-1")
        repo.link_external_id(session, a, "TIC", "123", method="mast_target", confidence="high")
        with pytest.raises(repo.IdentityConflictError):
            repo.link_external_id(session, b, "TIC", "123", method="simbad", confidence="high")


def test_link_external_id_keeps_provenance(db):
    with connect(db) as session:
        star = _star(session)
        added = repo.link_external_id(
            session,
            star,
            "TIC",
            "150428135",
            method="mast_target",
            confidence="high",
            separation_arcsec=0.4,
            source_archive="MAST",
            retrieved_at="2026-01-01T00:00:00Z",
        )
        row = session.scalar(
            select(ExternalId).where(
                ExternalId.catalog_name == "TIC", ExternalId.external_id == "150428135"
            )
        )
    assert added is True
    assert row.match_method == "mast_target"
    assert row.match_confidence == "high"
    assert row.match_separation_arcsec == 0.4
    assert row.source_archive == "MAST"


def test_planet_upsert_updates_without_duplicates(db):
    with connect(db) as session:
        star = _star(session)
        p1 = repo.upsert_planet(
            session,
            star_id=star,
            planet_name="TOI-700 d",
            status="CONFIRMED",
            status_source="NASA",
            discovery_method="Transit",
            discovery_year=2020,
        )
        p2 = repo.upsert_planet(
            session,
            star_id=star,
            planet_name="TOI-700 d",
            status="CONFIRMED",
            status_source="NASA v2",
            discovery_method="Transit",
            discovery_year=2020,
        )
        assert p1 == p2
        assert repo.table_counts(session)["planets"] == 1


def test_planet_cannot_move_to_other_star(db):
    with connect(db) as session:
        a = _star(session, name="A", ext="A")
        b = _star(session, name="B", ext="B")
        repo.upsert_planet(
            session,
            star_id=a,
            planet_name="p",
            status=None,
            status_source=None,
            discovery_method=None,
            discovery_year=None,
        )
        with pytest.raises(repo.IdentityConflictError):
            repo.upsert_planet(
                session,
                star_id=b,
                planet_name="p",
                status=None,
                status_source=None,
                discovery_method=None,
                discovery_year=None,
            )


def test_measurement_dedup_and_history(db):
    kwargs = dict(
        entity_type="planet",
        entity_id=1,
        parameter_name="pl_rade",
        unit="R_earth",
        error_plus=0.1,
        error_minus=0.09,
        reference="Gilbert 2020",
        source_archive="NASA",
        source_table="pscomppars",
    )
    with connect(db) as session:
        assert repo.insert_measurement(session, value=1.19, **kwargs) is True
        # The same value from the same source does not create a duplicate.
        assert repo.insert_measurement(session, value=1.19, **kwargs) is False
        # A changed value adds a row and preserves history.
        assert repo.insert_measurement(session, value=1.25, **kwargs) is True
        assert repo.table_counts(session)["parameter_measurements"] == 2


def test_measurement_unique_index_treats_null_as_empty(db):
    common = dict(
        entity_type="planet",
        entity_id=1,
        parameter_name="pl_rade",
        value=1.19,
        source_archive="NASA",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(IntegrityError), connect(db) as session:
        session.add_all(
            [
                ParameterMeasurement(**common, source_table=None, reference=None),
                ParameterMeasurement(**common, source_table="", reference=""),
            ]
        )
        session.flush()


def test_measurement_none_value_is_skipped(db):
    with connect(db) as session:
        assert (
            repo.insert_measurement(
                session,
                entity_type="star",
                entity_id=1,
                parameter_name="st_teff",
                value=None,
                unit="K",
                error_plus=None,
                error_minus=None,
                reference=None,
                source_archive="NASA",
                source_table=None,
            )
            is False
        )
        assert repo.table_counts(session)["parameter_measurements"] == 0


def _observation(session, star):
    return repo.upsert_observation(
        session, star_id=star, archive="MAST", archive_id="1001", mission="TESS"
    )


def test_file_registration_is_idempotent(db, tmp_path):
    lc = make_tess_like_lc(tmp_path / "lc.fits")
    digest = sha256_file(lc)
    kwargs = dict(
        file_path="data/raw/TESS/TOI-700/lc.fits",
        file_format="fits",
        file_size=lc.stat().st_size,
        sha256=digest,
        product_filename="lc.fits",
        product_type="SCIENCE",
        product_subgroup="LC",
        source_uri="mast:TESS/product/lc.fits",
        downloaded_at=None,
        validation_status="ok",
        validation_notes=None,
    )
    with connect(db) as session:
        star = _star(session)
        obs = _observation(session, star)
        f1 = repo.upsert_file(session, observation_id=obs, **kwargs)
        f2 = repo.upsert_file(session, observation_id=obs, **kwargs)
        assert f1 == f2
        assert repo.table_counts(session)["files"] == 1


def test_changed_raw_file_is_refused(db):
    base = dict(
        file_path="data/raw/x.fits",
        file_format="fits",
        file_size=1,
        product_filename="x.fits",
        product_type=None,
        product_subgroup=None,
        source_uri=None,
        downloaded_at=None,
        validation_status="ok",
        validation_notes=None,
    )
    with connect(db) as session:
        star = _star(session)
        obs = _observation(session, star)
        repo.upsert_file(session, observation_id=obs, sha256="a" * 64, **base)
        with pytest.raises(repo.RawFileChangedError):
            repo.upsert_file(session, observation_id=obs, sha256="b" * 64, **base)


def test_manifest_export_and_verification(settings, lc_fixture):
    init_db(settings.paths.db_path)
    raw_dir = settings.paths.raw_dir("TESS", "TOI-700")
    raw_dir.mkdir(parents=True)
    target = raw_dir / "lc.fits"
    target.write_bytes(lc_fixture.read_bytes())
    rel = settings.paths.relative(target)

    with connect(settings.paths.db_path) as session:
        star = _star(session)
        obs = _observation(session, star)
        repo.upsert_file(
            session,
            observation_id=obs,
            file_path=rel,
            file_format="fits",
            file_size=target.stat().st_size,
            sha256=sha256_file(target),
            product_filename="lc.fits",
            product_type="SCIENCE",
            product_subgroup="LC",
            source_uri=None,
            downloaded_at=None,
            validation_status="ok",
            validation_notes=None,
        )
        rows = export_manifest(session, settings.paths.manifest_path)
        assert rows == 1
        assert [c.status for c in verify_files(session, settings.paths)] == ["ok"]

        # A corrupted file must be detected.
        target.write_bytes(b"corrupted")
        assert [c.status for c in verify_files(session, settings.paths)] == ["mismatch"]

        target.unlink()
        assert [c.status for c in verify_files(session, settings.paths)] == ["missing"]

    header = settings.paths.manifest_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("file_path,sha256,file_size")

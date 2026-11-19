"""Stage 1 end-to-end pipeline tests using fake archive clients.

NASA, MAST, SIMBAD, and ESO are replaced with local data from tests/fakes.py.
Live archive checks are in tests/network/ (run: pytest -m network).
"""

import os
import stat

import pytest
from sqlalchemy import select

from src.data_collection import pipeline as pl
from src.database import repository as repo
from src.database.db import connect
from src.database.manifest import verify_files
from src.database.schema import ExternalId, ParameterMeasurement, Planet
from src.utils.hashing import sha256_file
from tests.fakes import LC_NAME, make_clients, make_tess_like_lc


def _run(settings, clients, **kwargs):
    return pl.run_stage1(
        "TOI-700",
        settings=settings,
        clients=clients,
        planet_limit=20,
        max_products=1,
        **kwargs,
    )


def _counts(settings):
    with connect(settings.paths.db_path) as session:
        return repo.table_counts(session)


def test_full_run_satisfies_acceptance_criteria(settings, lc_fixture):
    result = _run(settings, make_clients(lc_fixture), with_simbad=True, with_eso=True)

    # The NASA star is found and MAST observations are stored.
    assert result.star_id is not None
    assert result.n_planets == 1
    assert len(result.observation_ids) == 2

    # The FITS product is downloaded to the target directory.
    assert len(result.files) == 1
    file_info = result.files[0]
    assert file_info["status"] == "downloaded"
    assert file_info["product_filename"] == LC_NAME

    # HDUs, time, flux, errors, and quality flags are validated.
    assert file_info["validation"] == "ok"
    assert file_info["n_cadences"] == 2000
    assert file_info["time_start_jd"] > 2456000

    # The recorded SHA-256 matches the file.
    raw = settings.paths.absolute(file_info["file_path"])
    assert file_info["sha256"] == sha256_file(raw)
    assert file_info["file_path"] == f"data/raw/TESS/TOI-700/{LC_NAME}"

    # Metadata is stored in SQLite.
    counts = _counts(settings)
    assert counts["stars"] >= 2  # TOI-700 and HD 209458 from the sample
    assert counts["planets"] == 2
    assert counts["observations"] == 4  # 2 MAST + 2 ESO
    assert counts["files"] == 1
    assert result.ok, result.errors


def test_identifiers_linked_with_method_and_confidence(settings, lc_fixture):
    _run(settings, make_clients(lc_fixture), with_simbad=True)
    with connect(settings.paths.db_path) as session:
        rows = session.scalars(
            select(ExternalId)
            .where(ExternalId.catalog_name.in_(["TIC", "Gaia DR3", "HD"]))
            .order_by(ExternalId.catalog_name)
        ).all()
    methods = {(r.catalog_name, r.match_method, r.match_confidence) for r in rows}
    assert ("TIC", "mast_target", "high") in methods
    assert ("Gaia DR3", "simbad", "high") in methods
    assert ("HD", "simbad", "high") in methods


def test_planet_measurements_keep_provenance_and_positive_errors(settings, lc_fixture):
    _run(settings, make_clients(lc_fixture))
    with connect(settings.paths.db_path) as session:
        row = session.scalar(
            select(ParameterMeasurement)
            .join(
                Planet,
                (ParameterMeasurement.entity_type == "planet")
                & (ParameterMeasurement.entity_id == Planet.planet_id),
            )
            .where(
                Planet.planet_name == "TOI-700 d", ParameterMeasurement.parameter_name == "pl_rade"
            )
        )
        period = session.scalar(
            select(ParameterMeasurement).where(
                ParameterMeasurement.entity_type == "planet",
                ParameterMeasurement.entity_id == row.entity_id,
                ParameterMeasurement.parameter_name == "pl_orbper",
            )
        )
        star_id = session.get(Planet, row.entity_id).star_id
        temperature = session.scalar(
            select(ParameterMeasurement).where(
                ParameterMeasurement.entity_type == "star",
                ParameterMeasurement.entity_id == star_id,
                ParameterMeasurement.parameter_name == "st_teff",
            )
        )
    assert row.value == pytest.approx(1.19)
    assert row.unit == "R_earth"
    assert row.error_plus == pytest.approx(0.1)
    assert row.error_minus == pytest.approx(
        0.09
    )  # archive value -0.09 is stored as an absolute magnitude
    assert row.source_archive == "NASA Exoplanet Archive"
    assert row.source_table == "pscomppars"
    assert row.reference == "Gilbert et al. 2020"
    assert period.reference == "Orbital period source"
    assert temperature.reference == "Stellar temperature source"


def test_planet_status_is_recorded_not_assumed_for_candidates(settings, lc_fixture):
    _run(settings, make_clients(lc_fixture))
    with connect(settings.paths.db_path) as session:
        row = session.scalar(select(Planet))
    assert row.status == "CONFIRMED"
    assert "pscomppars" in row.status_source


def test_rerun_creates_no_duplicates_and_keeps_raw_file(settings, lc_fixture):
    clients = make_clients(lc_fixture)
    first = _run(settings, clients, with_simbad=True, with_eso=True)
    counts_1 = _counts(settings)
    raw = settings.paths.absolute(first.files[0]["file_path"])
    sha_1 = sha256_file(raw)

    second = _run(settings, clients, with_simbad=True, with_eso=True)
    counts_2 = _counts(settings)

    assert counts_1 == counts_2, "a repeated run must not add rows"
    assert second.files[0]["status"] == "reused"  # file was not downloaded again
    assert sha256_file(raw) == sha_1
    assert second.ok


def test_rerun_writes_new_snapshot_without_overwriting_old(settings, lc_fixture):
    clients = make_clients(lc_fixture)
    _run(settings, clients)
    _run(settings, clients)
    snapshots = sorted(settings.paths.snapshots.glob("pscomppars_*.csv"))
    assert len(snapshots) == 2
    assert snapshots[0] != snapshots[1]


def test_raw_file_is_read_only_after_download(settings, lc_fixture):
    result = _run(settings, make_clients(lc_fixture))
    raw = settings.paths.absolute(result.files[0]["file_path"])
    mode = stat.S_IMODE(os.stat(raw).st_mode)
    assert mode & stat.S_IWUSR == 0, "raw files must be read-only"


def test_changed_raw_file_stops_the_run(settings, lc_fixture):
    clients = make_clients(lc_fixture)
    first = _run(settings, clients)
    raw = settings.paths.absolute(first.files[0]["file_path"])

    # Simulate corruption by changing a file on disk.
    os.chmod(raw, 0o644)
    make_tess_like_lc(raw, seed=7, n=1500)

    with pytest.raises(repo.RawFileChangedError):
        _run(settings, clients)


def test_deleted_raw_file_is_redownloaded(settings, lc_fixture):
    clients = make_clients(lc_fixture)
    first = _run(settings, clients)
    raw = settings.paths.absolute(first.files[0]["file_path"])
    os.chmod(raw, 0o644)
    raw.unlink()

    second = _run(settings, clients)
    assert second.files[0]["status"] == "downloaded"
    assert second.files[0]["sha256"] == first.files[0]["sha256"]
    assert _counts(settings)["files"] == 1


def test_unknown_host_stops_with_clear_message(settings, lc_fixture):
    clients = make_clients(lc_fixture)
    with pytest.raises(pl.NoDataError, match="hostname"):
        pl.run_stage1(
            "NO SUCH STAR", settings=settings, clients=clients, planet_limit=5, max_products=1
        )


def test_download_failure_is_reported_not_registered(settings, lc_fixture):
    clients = make_clients(lc_fixture, download_status="ERROR")
    result = _run(settings, clients)
    assert not result.ok
    assert any("failed" in e for e in result.errors)
    assert _counts(settings)["files"] == 0


def test_mismatched_coordinates_do_not_link_mission_id(settings, lc_fixture):
    clients = make_clients(lc_fixture, mast_ra=10.0, mast_dec=10.0)
    result = _run(settings, clients)
    assert not any(i.startswith("TIC") for i in result.linked_identifiers)
    assert any("no link created" in w for w in result.warnings)


def test_manifest_and_verification_after_run(settings, lc_fixture):
    result = _run(settings, make_clients(lc_fixture))
    assert result.manifest_rows == 1
    assert settings.paths.manifest_path.exists()
    with connect(settings.paths.db_path) as session:
        checks = verify_files(session, settings.paths)
    assert [c.status for c in checks] == ["ok"]


def test_run_report_is_written(settings, lc_fixture):
    _run(settings, make_clients(lc_fixture))
    reports = list(settings.paths.reports.glob("stage1_run_*.json"))
    assert len(reports) == 1


def test_damaged_fits_is_registered_as_error_and_run_fails(settings, lc_fixture):
    # A file without TIME must be registered with error status.
    broken = lc_fixture.parent / "broken.fits"
    make_tess_like_lc(broken, include_time=False)
    clients = make_clients(broken)
    result = _run(settings, clients)
    assert not result.ok
    assert result.files[0]["validation"] == "error"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("150428135", "150428135"),
        ("TIC 150428135", "150428135"),
        ("tic150428135", "150428135"),
        ("TOI-700", None),
        ("", None),
        (None, None),
    ],
)
def test_mission_number_extraction(raw, expected):
    assert pl._mission_number(raw) == expected


def test_optional_simbad_failure_does_not_stop_run(settings, lc_fixture):
    clients = make_clients(lc_fixture)

    def boom(name):
        raise LookupError("SIMBAD has no object named 'TOI-700'")

    clients.resolve_simbad = boom
    result = _run(settings, clients, with_simbad=True)
    assert result.ok, result.errors
    assert any("SIMBAD skipped" in w for w in result.warnings)


def test_optional_eso_failure_does_not_stop_run(settings, lc_fixture):
    clients = make_clients(lc_fixture)

    def boom(*args, **kwargs):
        raise OSError("ESO unavailable")

    clients.search_eso = boom
    result = _run(settings, clients, with_eso=True)
    assert result.ok, result.errors
    assert any("ESO skipped" in w for w in result.warnings)

"""Live archive checks. Run: python -m pytest -m network

These tests are excluded from the default run because archives may be unavailable,
rate-limit requests, or change responses. Network errors produce clear failure messages.
"""

from contextlib import contextmanager

import pytest

from src.data_collection import access_check, eso, gaia, mast, nasa_archive, pipeline, simbad
from src.utils.config import load_settings

pytestmark = pytest.mark.network


@contextmanager
def archive(name: str):
    """Convert network exceptions into clear service-specific failures."""
    try:
        yield
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.fail(
            f"{name}: network access unavailable ({type(exc).__name__}: {exc}). "
            "Check internet access, VPN, and proxy settings."
        )
    except Exception as exc:  # pyvo.DALServiceError and similar exceptions
        if "DALServiceError" in type(exc).__name__ or "Timeout" in type(exc).__name__:
            pytest.fail(f"{name}: service unavailable or returned an error ({exc}).")
        raise


@pytest.fixture(scope="module")
def live_settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("live_project")
    settings = load_settings(root=root, env_file=root / "none.env")
    settings.paths.ensure_dirs()
    return settings


def test_nasa_returns_planets(live_settings):
    with archive("NASA Exoplanet Archive"):
        table, info = nasa_archive.fetch_planets(limit=5, settings=live_settings)
    assert len(table) > 0
    assert "pl_rade" in table.colnames and "hostname" in table.colnames
    assert info.n_rows == len(table)


def test_mast_finds_toi700_lightcurve(live_settings):
    with archive("MAST"):
        observations = mast.search_observations("TOI-700", "TESS")
        assert len(observations) > 0, "MAST returned no TESS observations for TOI-700"
        products = mast.list_lightcurve_products(
            observations, max_products=1, settings=live_settings
        )
    assert len(products) == 1
    assert str(products["productFilename"][0]).endswith(".fits")


def test_gaia_access(live_settings):
    with archive("Gaia Archive"):
        stars = gaia.check_access(live_settings, limit=3)
    assert len(stars) > 0


def test_eso_access(live_settings):
    with archive("ESO Archive"):
        rows = eso.check_access(live_settings, limit=3)
    assert len(rows) > 0


def test_simbad_resolves_toi700(live_settings):
    with archive("SIMBAD"):
        result = simbad.query_identifiers("TOI-700")
    assert result.ra_deg is not None and result.dec_deg is not None
    assert any(cat == "TIC" for cat, _ in result.identifiers)


def test_access_check_all_services_pass(live_settings):
    results = access_check.run_checks(live_settings)
    failed = [f"{r.name}: {r.message}" for r in results if not r.ok]
    assert not failed, "Archive access checks failed:\n" + "\n".join(failed)


def test_end_to_end_stage1_twice(live_settings):
    """Run the full pipeline on live data; repeated runs must not duplicate records."""
    with archive("Stage 1 pipeline"):
        first = pipeline.run_stage1(
            "TOI-700", settings=live_settings, planet_limit=20, max_products=1, with_simbad=True
        )
        second = pipeline.run_stage1(
            "TOI-700", settings=live_settings, planet_limit=20, max_products=1, with_simbad=True
        )
    assert first.ok, first.errors
    assert second.ok, second.errors
    assert second.db_counts == first.db_counts
    assert second.files[0]["status"] == "reused"
    assert second.files[0]["sha256"] == first.files[0]["sha256"]

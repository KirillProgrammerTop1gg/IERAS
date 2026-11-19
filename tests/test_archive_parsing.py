"""Offline tests for parsing archive responses and building queries."""

import pytest
from astropy.table import MaskedColumn, Table

from src.data_collection import eso, gaia, mast, nasa_archive, simbad
from src.data_collection.nasa_archive import QueryInfo, build_planets_query, strip_html
from src.utils.paths import ProjectPaths

# --- NASA Exoplanet Archive ---------------------------------------------------------


def test_planets_query_default_matches_instructions():
    query = build_planets_query(table="pscomppars", limit=20)
    assert query.startswith("SELECT TOP 20 ")
    assert "FROM pscomppars" in query
    assert "WHERE pl_rade IS NOT NULL" in query
    assert "pl_rade" in query and "hostname" in query
    assert "pl_refname" not in query
    for column in (
        "pl_rade_reflink",
        "pl_orbper_reflink",
        "pl_bmasse_reflink",
        "st_teff_reflink",
        "st_rad_reflink",
    ):
        assert column in query


def test_planets_query_by_host_without_radius_filter():
    query = build_planets_query(table="pscomppars", hostname="TOI-700", require_radius=False)
    assert "TOP" not in query
    assert "hostname = 'TOI-700'" in query
    assert "pl_rade IS NOT NULL" not in query


def test_hostname_quotes_are_escaped():
    query = build_planets_query(table="pscomppars", hostname="O'Hara 1")
    assert "hostname = 'O''Hara 1'" in query


def test_limit_is_coerced_to_int():
    query = build_planets_query(table="pscomppars", limit="5")
    assert query.startswith("SELECT TOP 5 ")


def test_strip_html_removes_tags_and_keeps_text():
    raw = '<a refstr=GILBERT_ET_AL__2020 href="/x">Gilbert et al. 2020</a>'
    assert strip_html(raw) == "Gilbert et al. 2020"
    assert strip_html(None) is None
    assert strip_html("<a></a>") is None


def test_snapshot_never_overwrites(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    paths.ensure_dirs()
    table = Table({"pl_name": ["b"]})
    info = QueryInfo("svc", "url", "pscomppars", "q", "2026-01-01T10:00:00Z", 1)

    first_csv, _ = nasa_archive.save_snapshot(table, info, paths)
    second_csv, _ = nasa_archive.save_snapshot(table, info, paths)

    assert first_csv != second_csv
    assert first_csv.exists() and second_csv.exists()
    assert second_csv.name.endswith("_2.csv")


def test_snapshot_latest_copy_can_be_skipped(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    paths.ensure_dirs()
    table = Table({"pl_name": ["b"]})
    info = QueryInfo("svc", "url", "pscomppars", "q", "2026-01-01T10:00:00Z", 1)
    nasa_archive.save_snapshot(table, info, paths, update_latest=False)
    assert not (paths.metadata / "planets_sample.csv").exists()


# --- SIMBAD -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TIC 150428135", ("TIC", "150428135")),
        ("Gaia DR3 5489213108398512000", ("Gaia DR3", "5489213108398512000")),
        ("HD 235088", ("HD", "235088")),
        ("TOI-700", ("TOI", "700")),
        ("2MASS J06285147-6537101", ("2MASS", "J06285147-6537101")),
        ("NGC 1234", None),  # skip catalogs outside the supported list
        ("  ", None),
    ],
)
def test_simbad_identifier_parsing(raw, expected):
    assert simbad.parse_identifier(raw) == expected


def test_simbad_identifier_list_is_deduplicated_and_ordered():
    parsed = simbad.parse_identifiers("TIC 1 | HD 2 | TIC 1 | Foo 9 | HD 2")
    assert parsed == (("TIC", "1"), ("HD", "2"))


def test_simbad_empty_identifiers():
    assert simbad.parse_identifiers(None) == ()
    assert simbad.parse_identifiers("") == ()


def test_simbad_uses_mirror_when_first_endpoint_fails(monkeypatch):
    urls = []

    class FakeService:
        def __init__(self, url, **kwargs):
            urls.append(url)
            self.url = url

        def search(self, query):
            assert "TOI-700" in query
            if self.url == simbad.SIMBAD_TAP_URLS[0]:
                raise OSError("primary mirror unavailable")
            return self

        def to_table(self):
            return Table(
                {
                    "main_id": ["TOI-700"],
                    "ra": [97.1],
                    "dec": [-65.6],
                    "ident_id": ["TIC 150428135"],
                }
            )

    monkeypatch.setattr(simbad, "TAPService", FakeService)
    result = simbad.query_identifiers("TOI-700")
    assert urls == list(simbad.SIMBAD_TAP_URLS)
    assert ("TIC", "150428135") in result.identifiers


# --- MAST ---------------------------------------------------------------------------


def test_mast_observation_rows_handle_masks():
    table = Table(
        {
            "obsid": [1001, 1002],
            "obs_collection": ["TESS", "TESS"],
            "dataproduct_type": ["timeseries", "timeseries"],
            "instrument_name": ["TESS/Photometer", "TESS/Photometer"],
            "target_name": ["150428135", "150428135"],
            "s_ra": [97.1, 97.1],
            "s_dec": MaskedColumn([-65.6, -65.6], mask=[False, True]),
            "t_min": [58000.0, 58028.0],
            "t_max": [58027.0, 58055.0],
            "t_exptime": [120.0, 120.0],
            "sequence_number": MaskedColumn([4, 5], mask=[False, True]),
        }
    )
    rows = mast.observation_rows(table)
    assert rows[0]["archive_id"] == "1001"
    assert rows[0]["dec"] == -65.6
    assert rows[1]["dec"] is None
    assert rows[1]["sequence_number"] is None


def test_mast_empty_observations_yield_no_products():
    assert len(mast.list_lightcurve_products(Table(), subgroups=["LC"])) == 0


def test_mast_product_lookup_stops_after_first_matching_batch(monkeypatch, settings):
    observations = Table(
        {
            "obsid": list(range(1, 21)),
            "dataproduct_type": ["image"] + ["timeseries"] * 19,
        }
    )
    calls = []

    def get_product_list(batch):
        ids = list(batch["obsid"])
        calls.append(ids)
        return Table({"productFilename": ["lc.fits"]}) if 9 in ids else Table()

    monkeypatch.setattr(mast.Observations, "get_product_list", get_product_list)
    monkeypatch.setattr(mast.Observations, "filter_products", lambda products, **kwargs: products)
    products = mast.list_lightcurve_products(observations, max_products=1, settings=settings)
    assert len(products) == 1
    assert len(calls) == 1
    assert calls[0][0] == 2  # timeseries observations are checked before images
    assert len(calls[0]) <= mast._PRODUCT_BATCH_SIZE


def test_mast_manifest_status_counts():
    manifest = Table({"Status": ["COMPLETE", "SKIPPED", "COMPLETE"]})
    assert mast.manifest_status_counts(manifest) == {"COMPLETE": 2, "SKIPPED": 1}
    assert mast.manifest_status_counts(Table({"Status": []})) == {}


# --- Gaia and ESO ---------------------------------------------------------------------


def test_gaia_cone_query_converts_radius_to_degrees():
    query = gaia.build_cone_query(97.1, -65.6, 3600.0, "gaiadr3.gaia_source", limit=10)
    assert "CIRCLE('ICRS', 97.1, -65.6, 1.0)" in query
    assert "TOP 10" in query


def test_eso_cone_query_filters_instruments():
    query = eso.build_cone_query(97.1, -65.6, 10.0, ("HARPS", "ESPRESSO"), limit=5)
    assert "instrument_name IN ('HARPS', 'ESPRESSO')" in query
    assert "TOP 5" in query
    assert "CONTAINS" in query


def test_eso_instrument_names_are_quoted_safely():
    query = eso.build_cone_query(1.0, 1.0, 1.0, ("HAR'PS",))
    assert "'HAR''PS'" in query

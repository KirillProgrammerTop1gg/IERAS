"""FITS inspector tests using synthetic TESS and Kepler light curves."""

import hashlib

from astropy.io import fits

from src.utils.fits_utils import (
    format_report,
    inspect_lightcurve,
    read_time_reference,
)
from tests.fakes import make_tess_like_lc


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_clean_tess_like_file_is_ok(tmp_path):
    path = make_tess_like_lc(tmp_path / "clean.fits")
    check = inspect_lightcurve(path)

    assert check.status == "ok", check.errors + check.warnings
    assert check.mission == "TESS"
    assert check.target_id == "150428135"
    assert check.sequence == "4"
    assert check.lc_hdu_name == "LIGHTCURVE"
    assert check.n_cadences == 2000
    assert check.flux_column == "PDCSAP_FLUX"
    assert check.error_column == "PDCSAP_FLUX_ERR"
    assert check.quality_column_present
    assert check.n_quality_flagged == 15
    assert check.time_monotonic is True
    assert check.n_nan_time == 0 and check.n_nan_flux == 0


def test_time_reference_is_read_from_header_and_converted_to_jd(tmp_path):
    path = make_tess_like_lc(tmp_path / "t.fits", bjdrefi=2457000)
    check = inspect_lightcurve(path)

    assert check.time_reference.timesys == "TDB"
    assert check.time_reference.reference_jd == 2457000.0
    assert check.time_start_jd == 2457000.0 + check.time_min
    assert check.time_end_jd == 2457000.0 + check.time_max


def test_inspection_does_not_modify_file(tmp_path):
    path = make_tess_like_lc(tmp_path / "readonly.fits")
    before = _sha(path)
    inspect_lightcurve(path)
    assert _sha(path) == before


def test_missing_time_column_is_error(tmp_path):
    path = make_tess_like_lc(tmp_path / "no_time.fits", include_time=False)
    check = inspect_lightcurve(path)
    assert check.status == "error"
    assert any("TIME" in e for e in check.errors)


def test_missing_lc_table_is_error(tmp_path):
    path = make_tess_like_lc(tmp_path / "no_lc.fits", include_lc_hdu=False)
    check = inspect_lightcurve(path)
    assert check.status == "error"
    assert check.lc_hdu_name is None


def test_all_nan_flux_is_error(tmp_path):
    path = make_tess_like_lc(tmp_path / "nan.fits", all_nan_flux=True)
    check = inspect_lightcurve(path)
    assert check.status == "error"
    assert any("NaN" in e for e in check.errors)


def test_partial_nan_flux_is_warning_with_count(tmp_path):
    path = make_tess_like_lc(tmp_path / "partial.fits", nan_flux_every=10)
    check = inspect_lightcurve(path)
    assert check.status == "warning"
    assert check.n_nan_flux == 200
    assert any("200" in w for w in check.warnings)


def test_non_monotonic_time_is_warning(tmp_path):
    path = make_tess_like_lc(tmp_path / "rev.fits", reverse_time=True)
    check = inspect_lightcurve(path)
    assert check.time_monotonic is False
    assert any("monotonic" in w for w in check.warnings)


def test_missing_timesys_is_warning(tmp_path):
    path = make_tess_like_lc(tmp_path / "nots.fits", timesys=None)
    check = inspect_lightcurve(path)
    assert any("TIMESYS" in w for w in check.warnings)


def test_unexpected_reference_epoch_is_warning(tmp_path):
    # TESS expects BJDREFI=2457000; report a different value without correcting it.
    path = make_tess_like_lc(tmp_path / "wrongref.fits", bjdrefi=2454833)
    check = inspect_lightcurve(path)
    assert any("BJDREFI" in w for w in check.warnings)


def test_kepler_reference_epoch_is_accepted(tmp_path):
    path = make_tess_like_lc(tmp_path / "kepler.fits", telescope="Kepler", bjdrefi=2454833)
    check = inspect_lightcurve(path)
    assert check.mission == "KEPLER"
    assert not any("BJDREFI" in w for w in check.warnings)


def test_read_time_reference_mjdref_fallback():
    header = fits.Header({"MJDREF": 58000.0, "TIMESYS": "TT"})
    ref = read_time_reference(header)
    assert ref.reference_jd == 58000.0 + 2400000.5
    assert ref.timesys == "TT"


def test_read_time_reference_without_keys():
    ref = read_time_reference(fits.Header())
    assert ref.reference_jd is None


def test_report_text_contains_status(tmp_path):
    path = make_tess_like_lc(tmp_path / "rep.fits")
    text = format_report(inspect_lightcurve(path))
    assert "Status: ok" in text
    assert "TESS" in text

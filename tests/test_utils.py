"""Utility tests for paths, settings, checksums, name sanitization, and tables."""

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from src.utils.config import load_settings
from src.utils.hashing import sha256_file
from src.utils.paths import ProjectPaths, get_paths, safe_name
from src.utils.tables import cell, cell_float, cell_str


def test_sha256_known_value(tmp_path):
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_large_file_chunked(tmp_path):
    path = tmp_path / "big.bin"
    payload = b"x" * (3 * 1024 * 1024 + 17)
    path.write_bytes(payload)
    import hashlib

    assert sha256_file(path, chunk_size=4096) == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("TOI-700", "TOI-700"), ("HD 189733", "HD_189733"), ("  Kepler/22 ", "Kepler_22")],
)
def test_safe_name(raw, expected):
    assert safe_name(raw) == expected


def test_safe_name_rejects_empty():
    with pytest.raises(ValueError):
        safe_name("   ")


def test_relative_paths_use_forward_slashes(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    target = paths.raw_dir("TESS", "TOI-700") / "file.fits"
    assert paths.relative(target) == "data/raw/TESS/TOI-700/file.fits"


def test_relative_rejects_outside_root(tmp_path):
    paths = ProjectPaths(root=tmp_path / "proj")
    with pytest.raises(ValueError):
        paths.relative(tmp_path / "elsewhere.fits")


def test_get_paths_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IERAS_ROOT", str(tmp_path))
    assert get_paths().root == tmp_path.resolve()


def test_load_settings_reads_toml(tmp_path):
    settings = load_settings(root=tmp_path, env_file=tmp_path / "none.env")
    assert settings.nasa_table == "pscomppars"
    assert "LC" in settings.lightcurve_subgroups
    assert settings.match_radius_arcsec == 5.0
    assert settings.paths.root == tmp_path.resolve()


def test_load_settings_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(root=tmp_path, config_file=tmp_path / "missing.toml")


def test_cell_helpers_handle_masks_and_nan():
    table = Table(
        {
            "a": MaskedColumn([1.5, 2.0], mask=[True, False]),
            "b": ["  text  ", ""],
            "c": np.array([np.nan, 3.0]),
            "d": [7, 8],
        }
    )
    assert cell(table[0], "a") is None
    assert cell_float(table[1], "a") == 2.0
    assert cell(table[0], "c") is None
    assert cell_str(table[0], "b") == "text"
    assert cell(table[1], "b") is None
    assert cell_str(table[1], "d") == "8"
    assert cell(table[0], "missing_column") is None

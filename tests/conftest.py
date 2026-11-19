"""Shared fixtures use temporary paths to avoid changing project data."""

from __future__ import annotations

import pytest

from src.utils.config import load_settings


@pytest.fixture
def project_root(tmp_path):
    """Isolated project root with its own database, data/, logs/, and reports/."""
    return tmp_path / "project"


@pytest.fixture
def settings(project_root):
    project_root.mkdir(parents=True, exist_ok=True)
    cfg = load_settings(root=project_root, env_file=project_root / "absent.env")
    cfg.paths.ensure_dirs()
    return cfg


@pytest.fixture
def lc_fixture(tmp_path):
    """Synthetic TESS light curve outside the project root, like an archive source file."""
    from tests.fakes import make_tess_like_lc

    src_dir = tmp_path / "archive_source"
    src_dir.mkdir()
    return make_tess_like_lc(src_dir / "source_lc.fits")

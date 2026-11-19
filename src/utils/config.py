"""Load settings from configs/archives.toml and .env variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.utils.paths import DEFAULT_ROOT, ProjectPaths, get_paths

DEFAULT_CONFIG_FILE = DEFAULT_ROOT / "configs" / "archives.toml"


@dataclass(frozen=True)
class Settings:
    paths: ProjectPaths
    log_level: str
    # NASA Exoplanet Archive
    nasa_tap_url: str
    nasa_table: str
    # MAST
    default_mission: str
    lightcurve_subgroups: tuple[str, ...]
    # Gaia
    gaia_table: str
    # ESO
    eso_tap_url: str
    eso_instruments: tuple[str, ...]
    # Cross-matching and pipeline
    match_radius_arcsec: float
    max_products_per_target: int
    planet_sample_limit: int


def _read_toml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_settings(
    root: str | os.PathLike[str] | None = None,
    config_file: Path = DEFAULT_CONFIG_FILE,
    env_file: Path | None = None,
) -> Settings:
    """Assemble settings. Values from .env override TOML defaults.

    .env is read from the project root when present. Existing environment variables
    are not overwritten.
    """
    paths = get_paths(root)
    env_path = env_file if env_file is not None else paths.root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    cfg = _read_toml(config_file)
    nasa = cfg["nasa_exoplanet_archive"]
    mast = cfg["mast"]
    gaia = cfg["gaia"]
    eso = cfg["eso"]
    matching = cfg["matching"]
    pipeline = cfg["pipeline"]

    return Settings(
        paths=paths,
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        nasa_tap_url=nasa["tap_url"],
        nasa_table=nasa["table"],
        default_mission=mast["default_mission"],
        lightcurve_subgroups=tuple(mast["lightcurve_subgroups"]),
        gaia_table=gaia["table"],
        eso_tap_url=eso["tap_url"],
        eso_instruments=tuple(eso["instruments"]),
        match_radius_arcsec=float(matching["radius_arcsec"]),
        max_products_per_target=int(pipeline["max_products_per_target"]),
        planet_sample_limit=int(pipeline["planet_sample_limit"]),
    )

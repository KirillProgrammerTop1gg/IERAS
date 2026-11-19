# Stage 1 — Infrastructure and Data Cataloging

This report documents the first stage of IERAS: obtaining real astronomical data and cataloging it through NASA Exoplanet Archive → MAST → FITS validation → SHA-256 → SQLite. It does not cover training neural networks, producing a final dataset, or claiming discoveries.

## Implementation map

| Area | Implementation |
|---|---|
| Environment and dependencies | `requirements.txt`, `pyproject.toml` |
| FITS, units, and time | `src/utils/fits_utils.py`, `notebooks/01_fits_units_time.ipynb` |
| NASA Exoplanet Archive | `src/data_collection/nasa_archive.py` |
| MAST light curves | `src/data_collection/mast.py` |
| Gaia, ESO, SIMBAD | `src/data_collection/gaia.py`, `eso.py`, `simbad.py` |
| Cross-matching | `src/database/repository.py`, `src/data_collection/pipeline.py` |
| SQLAlchemy ORM and SQLite metadata | `src/database/schema.py`, `db.py`, `repository.py`, `manifest.py` |
| Offline and live tests | `tests/`, `tests/network/` |

Empty data, log, and report directories are retained with `.gitkeep`.

## Installation on Windows (PowerShell)

Use Python 3.11 or newer. The examples below assume the repository is already checked out:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Install PyTorch separately using the [official selector](https://pytorch.org/get-started/locally/) for your CPU or CUDA configuration. Check it with `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`. For notebooks, install a kernel with `python -m ipykernel install --user --name ieras --display-name IERAS`, then run `jupyter lab`.

Copy `.env.example` to `.env` and adjust `LOG_LEVEL` if needed. Archive endpoints and matching settings are in `configs/archives.toml`. On Windows, set `$env:PYTHONUTF8 = "1"` if the console does not display UTF-8 output correctly.

## Commands

Run these from the repository root with the virtual environment active:

```powershell
python -m src.database.maintenance init
python -m src.data_collection.access_check
python -m src.data_collection.access_check --only mast gaia
python -m src.data_collection.pipeline --host TOI-700 --mission TESS
python -m src.data_collection.pipeline --host TOI-700 --with-simbad --with-eso
python -m src.database.maintenance counts
python -m src.database.maintenance manifest
python -m src.database.maintenance verify
python -m pytest
python -m pytest -m network
python -m ruff check .
python -m ruff format --check .
```

The access check, pipeline, and network-marked tests require internet access. `--host` must match the NASA Exoplanet Archive `hostname` value, including case and spaces. The pipeline returns exit code 0 on success, 1 for data errors, and 2 for unexpected errors.

## End-to-end data flow

1. **NASA Exoplanet Archive:** request a small `pscomppars` sample (20 planets by default), query a missing target separately, and save a response snapshot under `data/metadata/snapshots/`.
2. **MAST:** find observations for the selected mission. Match TIC, KIC, or EPIC identifiers to the star only if coordinates agree within `radius_arcsec` (5 arcseconds by default).
3. **Download:** select supported science FITS light curves (LC, LLC, SLC, FAST-LC), up to `max_products_per_target` files (one by default). Store them under `data/raw/<mission>/<target>/` as read-only files.
4. **FITS validation:** inspect HDUs, time headers, TIME, flux, error, and QUALITY columns. Write a report under `reports/fits_checks/` with `ok`, `warning`, or `error` status.
5. **Checksums:** record each raw file's SHA-256 in the `files` table.
6. **Optional sources:** add SIMBAD identifiers after coordinate matching and ESO HARPS/ESPRESSO metadata. ESO spectra are not downloaded.
7. **Manifest:** write `data/metadata/manifest_files.csv` and a run report under `reports/`.

Repeated runs avoid duplicate records and downloads. A changed raw file causes `RawFileChangedError`.

NASA `pscomppars` provenance comes from each parameter's `*_reflink` column. When a file limit is set, MAST product lookup checks small groups of likely light-curve observations and stops once enough products are found.

## Storage rules

- `data/raw/` contains immutable source files. Derived files belong in `data/interim/` or `data/processed/`.
- `data/metadata/` contains `exoplanet.db`, manifests, and timestamped archive-response snapshots.
- `reports/` and `logs/` contain generated output and are excluded from Git, except this report.
- Database and manifest paths are relative to the repository root and use forward slashes on every platform.
- SQLite stores metadata and paths, not large arrays or FITS contents.

## Database and provenance

SQLAlchemy ORM models define the original SQLite table and column names, so existing stage 1 databases remain readable. Sessions commit successful operations and roll back failures. The schema tracks coordinate epochs and sources, planet status sources, archive identifiers, observation times, product metadata, validation status, and retrieval timestamps. Measurements retain their source table and retrieval date.

Star names are not unique identifiers. External IDs are unique by catalog and ID; conflicting links raise `IdentityConflictError`. Repeated measurements from the same source do not create duplicates, while changed values add history. NASA's negative lower-radius uncertainty is stored as its absolute magnitude. The polymorphic `entity_type`/`entity_id` relationship is checked by the repository layer.

## Time and units

For TESS, TIME is BTJD = BJD_TDB − 2457000; for Kepler, it is BKJD = BJD_TDB − 2454833. The reference epoch comes from FITS headers (`BJDREFI`, `BJDREFF`, `MJDREF`), rather than a hard-coded offset. Unexpected mission epochs produce a warning. MAST and ObsCore `t_min` values are stored as MJD in `t_min_mjd`. NASA radii use Earth radii, masses use Earth masses, periods use days, and temperatures use kelvin.

## Connected archives and attribution

| Archive | Current use | Website |
|---|---|---|
| NASA Exoplanet Archive | `pscomppars`, TAP/ADQL | https://exoplanetarchive.ipac.caltech.edu/ |
| MAST | TESS/Kepler observations and light curves | https://archive.stsci.edu/ |
| Gaia Archive | Gaia DR3 access check | https://gea.esac.esa.int/archive/ |
| ESO Science Archive | ObsCore HARPS/ESPRESSO metadata | https://archive.eso.org/ |
| SIMBAD | Identifiers and coordinates through CDS-published TAP mirrors | https://simbad.cds.unistra.fr/ |

Check each archive's current attribution requirements before publication. Relevant references include NASA Exoplanet Archive DOI 10.26133/NEA12, Ricker et al. (2015) for TESS, Lightkurve Collaboration (2018), Astropy Collaboration (2013, 2018, 2022), Gaia Collaboration/Vallenari et al. (2023), and Wenger et al. (2000) for SIMBAD.

The SIMBAD client currently prefers the IDIA mirror and falls back to the CfA Harvard mirror while the Strasbourg service is degraded. Mirror data may lag the main service; check its status before scientific publication.

## Verification and limitations

The offline suite covers synthetic TESS and Kepler FITS files, invalid time and flux data, database transactions and idempotency, identifier conflicts, raw-file immutability, manifests, archive-response parsing, and the full pipeline with fake archives. Run `python -m pytest` to see the current result.

Live archive requests depend on external services. On a connected machine, run `python -m src.data_collection.access_check` and `python -m pytest -m network`. If an archive schema changes, check NASA `QUERY_COLUMNS` and the ESO TAP settings against their current documentation.

Current limitations: Gaia parallax is not converted to distance; proper motion is not used in coordinate matching; HARPS/ESPRESSO spectra are metadata only; and a default run downloads one light-curve file per star. Mass collection and scientific inference belong to later stages.

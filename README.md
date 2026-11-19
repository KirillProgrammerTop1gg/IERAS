# Independent Exoplanet Research and Analysis System (IERAS)

## Overview

**IERAS** is a research project for finding and studying exoplanets using open astronomical observations. Its goal is to combine reproducible data processing, astrophysical methods, and machine learning: from collecting observations to identifying candidates, estimating physical properties, and analyzing atmospheres when suitable spectra exist.

The project is developed in stages. This repository currently implements the **data collection and cataloging foundation**. Detection, characterization, and atmospheric models are planned; the project does not yet claim scientific discoveries of new planets.

## Goals

- Collect and cross-match objects and observations from multiple archives while recording data provenance.
- Search for transit signals and vet candidates for false positives.
- Estimate radii, masses, orbital parameters, and their uncertainties when the required measurements are available.
- Analyze atmospheric spectra and combine evidence from different observation types.

## Data and sources

The current stage handles star and planet catalogs, observation metadata, and Kepler and TESS FITS light curves. It integrates the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/), [MAST](https://archive.stsci.edu/), [Gaia Archive](https://gea.esac.esa.int/archive/), [ESO Science Archive](https://archive.eso.org/), and [SIMBAD](https://simbad.cds.unistra.fr/). Archive availability is checked separately.

Future stages may use radial velocities, transmission spectra, thermal phase curves, and imaging. The available observations depend on the target and instrument.

## Current architecture

At this stage, `src/data_collection/` retrieves data and runs the pipeline, `src/utils/` checks FITS files, units, and paths, and `src/database/` uses SQLAlchemy ORM models and sessions to store metadata and provenance in SQLite. Raw files are saved under `data/raw/` with SHA-256 checksums. Tests live in `tests/`, and archive settings are in `configs/archives.toml`.

```text
Open archives → collection and matching → FITS validation → SQLite + files + manifest
                                                        ↓
                Future stages: detection → vetting → characterization → atmospheres
```

The current stack includes Python 3.11+, Astropy, Astroquery, Lightkurve, NumPy, SciPy, pandas, SQLAlchemy with SQLite, and pytest. PyTorch is installed separately for future models. Dependencies for the first stage are listed in `requirements.txt`.

An experimental BLS transit-search prototype is available in `ieras_transit_search.py`. It can run on synthetic data with `python ieras_transit_search.py --selftest --no-plots`; its candidates require independent vetting and are not scientific discoveries.

## Quick start

Create a Python 3.11+ virtual environment in the repository root, install the dependencies, and copy `.env.example` to `.env` if local settings are needed. Then run:

```bash
python -m pip install -r requirements.txt
python -m src.database.maintenance init
python -m pytest
python -m src.data_collection.access_check
python -m src.data_collection.pipeline --host TOI-700 --mission TESS
```

The last two commands contact external archives. The [stage 1 data foundation report](reports/stage-1-data-foundation.md) covers Windows setup, the database, and current limitations. Run live archive tests separately with `python -m pytest -m network`.

## Roadmap

1. **Data foundation:** collect, validate, and catalog observations; expand reproducible datasets.
2. **Candidate detection:** search for transits, classify signals, and evaluate results on independent star systems.
3. **Physical characterization:** add radial velocities and estimate parameters and uncertainties.
4. **Atmospheres and multimodal analysis:** process suitable spectra and jointly interpret available observations.
5. **Scientific validation:** test on independent data, compare with published results, and release a reproducible catalog.

**Status:** research and development; the first-stage infrastructure is implemented. **Current focus:** Kepler and TESS data for future exoplanet searches.

## Language and contributions

Contributors may discuss the project in Ukrainian, Russian, or English. Repository content—including documentation, code comments, user-facing messages, tests, notebooks, issues, and pull requests—should be written in English. See [AGENTS.md](AGENTS.md) for contributor guidelines.

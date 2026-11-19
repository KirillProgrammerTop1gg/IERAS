# Repository Guidelines

## Project Structure & Module Organization

`src/data_collection/` contains archive clients and the ingestion pipeline; `src/database/` contains SQLAlchemy ORM models, sessions, repository operations, and maintenance commands for SQLite; `src/utils/` holds shared configuration, FITS, and path helpers. Tests live in `tests/`, with live archive checks in `tests/network/`. Archive settings are in `configs/archives.toml`, exploratory work in `notebooks/`, and generated data, logs, and reports under `data/`, `logs/`, and `reports/`.

`ieras_transit_search.py` is a standalone experimental BLS candidate-search prototype.

## Build, Test, and Development Commands

Use Python 3.11 or newer. Create a virtual environment and install dependencies with `python -m pip install -r requirements.txt`; PyTorch is installed separately when needed (see `README.md`). There is no build step.

- `python -m src.database.maintenance init` creates the local database tables.
- `python -m src.data_collection.pipeline --host TOI-700 --mission TESS` runs an ingestion example; it needs archive access.
- `python -m src.data_collection.access_check` checks connectivity to the configured archives.
- `python -m pytest` runs the default offline test suite.
- `python -m ruff check .` and `python -m ruff format --check .` check lint and formatting.

## Coding Style & Naming Conventions

Use four spaces for Python indentation, `snake_case` for modules and functions, and `PascalCase` for classes. Keep imports sorted and lines within Ruff's 100-character limit. Ruff targets Python 3.11 and checks `E`, `F`, `W`, `I`, and `B` rules. Run `python -m ruff format .` before submitting code.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and `test_*` functions. Add offline tests for parsing, database, and pipeline changes; use the temporary-path fixtures in `tests/conftest.py` so tests do not alter project data. Mark real archive requests with `@pytest.mark.network` and run them explicitly using `python -m pytest -m network`. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Git history contains only `Initial commit`, so no commit message convention is established. Use short, imperative subjects that describe the change. In pull requests, summarize the behavior changed, include the test commands and results, link any related issue, and attach screenshots only for notebook or visual output changes.

## Configuration & Data

Copy `.env.example` to `.env` for local settings. Keep `.env`, databases, downloaded archive files, and generated outputs out of commits; `.gitignore` excludes them. Update `configs/archives.toml` when archive endpoints or matching rules change.

## Project Language

Discuss work in Ukrainian, Russian, or English as convenient. Write repository content in English: documentation, code comments and docstrings, CLI output, plots, notebooks, tests, issues, and pull requests. Preserve official names and astronomical identifiers in their original form. Translate external material into English when adding it to the repository.

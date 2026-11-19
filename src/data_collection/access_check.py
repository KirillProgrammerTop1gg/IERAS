"""Check stage 1 archive access with one query per service.

    python -m src.data_collection.access_check            # all checks
    python -m src.data_collection.access_check --only mast gaia

Network checks require internet access. An unavailable service is reported while the
remaining checks continue. Exit code 0 means all checks passed; 1 means at least one failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from src.data_collection import eso, gaia, mast, nasa_archive, simbad
from src.utils.config import Settings, load_settings
from src.utils.logging_setup import get_logger, setup_logging

log = get_logger("access")

# TOI-700 is a planetary system with TESS observations used to check MAST and SIMBAD.
CHECK_TARGET = "TOI-700"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    seconds: float
    message: str


def check_nasa(settings: Settings) -> str:
    table, info = nasa_archive.fetch_planets(limit=3, settings=settings)
    if len(table) == 0:
        raise RuntimeError("query succeeded but returned an empty table")
    return f"{info.table}: rows received {len(table)}, columns {len(table.colnames)}"


def check_mast(settings: Settings) -> str:
    started = time.perf_counter()
    observations = mast.search_observations(CHECK_TARGET, settings.default_mission)
    observation_seconds = time.perf_counter() - started
    if len(observations) == 0:
        raise RuntimeError(f"no observations for {CHECK_TARGET} in {settings.default_mission}")
    started = time.perf_counter()
    products = mast.list_lightcurve_products(observations, max_products=1, settings=settings)
    product_seconds = time.perf_counter() - started
    return (
        f"observations: {len(observations)}, light curves in sample: {len(products)} "
        f"(observation query {observation_seconds:.1f}s, product lookup {product_seconds:.1f}s)"
    )


def check_gaia(settings: Settings) -> str:
    stars = gaia.check_access(settings, limit=5)
    return f"{settings.gaia_table}: rows received {len(stars)}"


def check_eso(settings: Settings) -> str:
    rows = eso.check_access(settings, limit=3)
    return f"ObsCore HARPS: rows received {len(rows)}"


def check_simbad(settings: Settings) -> str:
    result = simbad.query_identifiers(CHECK_TARGET)
    return (
        f"primary name: {result.main_id}, identifiers: {len(result.identifiers)}, "
        f"sample: {result.identifiers[:3]}, TAP: {result.tap_url}"
    )


CHECKS: dict[str, Callable[[Settings], str]] = {
    "nasa": check_nasa,
    "mast": check_mast,
    "gaia": check_gaia,
    "eso": check_eso,
    "simbad": check_simbad,
}


def run_checks(settings: Settings, names: list[str] | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name in names or list(CHECKS):
        func = CHECKS[name]
        started = time.perf_counter()
        try:
            message = func(settings)
            ok = True
        except Exception as exc:  # Check every service instead of stopping at the first failure.
            message = f"{type(exc).__name__}: {exc}"
            ok = False
        elapsed = time.perf_counter() - started
        results.append(CheckResult(name=name, ok=ok, seconds=elapsed, message=message))
        log.info("%-7s %s (%.1f s): %s", name, "OK" if ok else "FAIL", elapsed, message)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check access to stage 1 archives")
    parser.add_argument("--only", nargs="+", choices=list(CHECKS), help="check only these services")
    parser.add_argument("--root", default=None, help="project root")
    args = parser.parse_args(argv)

    settings = load_settings(root=args.root)
    setup_logging(settings.paths.logs, settings.log_level)

    results = run_checks(settings, args.only)
    print("\nService    Status  Time, s  Details")
    for r in results:
        print(f"{r.name:9s} {'OK' if r.ok else 'FAIL':6s}  {r.seconds:8.1f}  {r.message}")
    failed = [r.name for r in results if not r.ok]
    print("\nAll checks passed." if not failed else f"Failed: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

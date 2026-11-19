"""MAST (STScI) astroquery client: find observations, select products, and download.

Observations and products are distinct: one observation may have raw, calibrated, and
science products. Select only science FITS light curves in the minimum recommended set.

Files are stored flat under data/raw/<mission>/<target>/<filename>, making paths
independent of mastDownload's internal directory layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from astroquery.mast import Observations

from src.utils.config import Settings, load_settings
from src.utils.tables import cell_float, cell_str

_PRODUCT_BATCH_SIZE = 8


def search_observations(target: str, mission: str) -> Table:
    """Find target observations for a mission; an empty table is a valid result."""
    return Observations.query_criteria(objectname=target, obs_collection=mission)


def list_lightcurve_products(
    observations: Table,
    subgroups: Sequence[str] | None = None,
    max_products: int | None = None,
    settings: Settings | None = None,
) -> Table:
    """Select science FITS light curves, stopping early when a file limit is set."""
    if len(observations) == 0:
        return Table()
    settings = settings or load_settings()
    wanted = list(subgroups or settings.lightcurve_subgroups)

    if max_products is None:
        products = Observations.get_product_list(observations)
        selected = Observations.filter_products(
            products,
            productType="SCIENCE",
            productSubGroupDescription=wanted,
            extension="fits",
            mrp_only=True,
        )
        if len(selected):
            selected.sort("productFilename")
        return selected

    if max_products is not None and int(max_products) <= 0:
        return Table()

    # Query likely light-curve observations first. Fetching products for every observation
    # can be expensive even when the caller needs just one FITS file.
    timeseries = [
        i
        for i, row in enumerate(observations)
        if (cell_str(row, "dataproduct_type") or "").lower() == "timeseries"
    ]
    timeseries_ids = set(timeseries)
    other = [i for i in range(len(observations)) if i not in timeseries_ids]
    ordered = observations[timeseries + other]

    matches: list[Table] = []
    count = 0
    for start in range(0, len(ordered), _PRODUCT_BATCH_SIZE):
        products = Observations.get_product_list(ordered[start : start + _PRODUCT_BATCH_SIZE])
        selected = Observations.filter_products(
            products,
            productType="SCIENCE",
            productSubGroupDescription=wanted,
            extension="fits",
            mrp_only=True,
        )
        if len(selected):
            matches.append(selected)
            count += len(selected)
        if max_products is not None and count >= int(max_products):
            break

    if not matches:
        return Table()
    selected = vstack(matches) if len(matches) > 1 else matches[0]
    selected.sort("productFilename")
    return selected[: int(max_products)] if max_products is not None else selected


def download_products(products: Table, dest_dir: Path) -> Table:
    """Download products to dest_dir and return the astroquery manifest.

    Manifest columns: 'Local Path', 'Status' (COMPLETE, SKIPPED, or ERROR), 'Message', 'URL'.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Observations.download_products(
        products,
        download_dir=str(dest_dir),
        flat=True,
        cache=True,
        verbose=False,
    )


def observation_rows(observations: Table) -> list[dict]:
    """Normalize MAST observation fields for database storage."""
    rows = []
    for row in observations:
        rows.append(
            {
                "archive_id": cell_str(row, "obsid"),
                "mission": cell_str(row, "obs_collection"),
                "instrument": cell_str(row, "instrument_name"),
                "observation_type": cell_str(row, "dataproduct_type"),
                "sequence_number": cell_str(row, "sequence_number"),
                "target_name": cell_str(row, "target_name"),
                "ra": cell_float(row, "s_ra"),
                "dec": cell_float(row, "s_dec"),
                "t_min_mjd": cell_float(row, "t_min"),
                "t_max_mjd": cell_float(row, "t_max"),
                "exptime_s": cell_float(row, "t_exptime"),
            }
        )
    return rows


def manifest_status_counts(manifest: Table) -> dict[str, int]:
    if len(manifest) == 0:
        return {}
    statuses, counts = np.unique(np.asarray(manifest["Status"]).astype(str), return_counts=True)
    return {str(s): int(c) for s, c in zip(statuses, counts, strict=True)}

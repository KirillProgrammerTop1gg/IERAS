"""Synthetic data and fake archive clients for offline tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import MaskedColumn, Table

from src.data_collection.nasa_archive import QueryInfo
from src.data_collection.pipeline import ArchiveClients
from src.data_collection.simbad import SimbadResult
from src.database.db import utc_now

TOI700_RA = 97.1
TOI700_DEC = -65.6
TIC_ID = "150428135"
LC_NAME = "tess2019-s0004-0000000150428135-0001-s_lc.fits"


def make_tess_like_lc(
    path: Path,
    *,
    n: int = 2000,
    tic: int = int(TIC_ID),
    sector: int = 4,
    bjdrefi: float = 2457000,
    timesys: str | None = "TDB",
    telescope: str = "TESS",
    nan_flux_every: int = 0,
    all_nan_flux: bool = False,
    reverse_time: bool = False,
    include_time: bool = True,
    include_lc_hdu: bool = True,
    seed: int = 42,
) -> Path:
    """Write a FITS file with TESS SPOC light-curve HDUs and TIME/FLUX/QUALITY columns."""
    rng = np.random.default_rng(seed)
    time = np.linspace(1325.0, 1352.0, n)
    if reverse_time:
        time = time[::-1].copy()
    flux = 1.0 + 1e-3 * rng.standard_normal(n)
    if nan_flux_every:
        flux[::nan_flux_every] = np.nan
    if all_nan_flux:
        flux[:] = np.nan
    err = np.full(n, 1e-3)
    quality = np.zeros(n, dtype=np.int32)
    quality[:15] = 1  # several flagged cadences

    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = telescope
    primary.header["TICID"] = tic
    primary.header["SECTOR"] = sector
    primary.header["OBJECT"] = f"TIC {tic}"

    hdus = [primary]
    if include_lc_hdu:
        columns = []
        if include_time:
            columns.append(fits.Column(name="TIME", format="D", unit="BTJD", array=time))
        columns += [
            fits.Column(name="SAP_FLUX", format="E", array=flux * 0.99),
            fits.Column(name="SAP_FLUX_ERR", format="E", array=err),
            fits.Column(name="PDCSAP_FLUX", format="E", array=flux),
            fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=err),
            fits.Column(name="QUALITY", format="J", array=quality),
        ]
        lc = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="LIGHTCURVE")
        if timesys is not None:
            lc.header["TIMESYS"] = timesys
        lc.header["BJDREFI"] = bjdrefi
        lc.header["BJDREFF"] = 0.0
        lc.header["TIMEUNIT"] = "d"
        lc.header["TIMEREF"] = "SOLARSYSTEM"
        lc.header["TIMEZERO"] = 0.0
        hdus.append(lc)

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def table_from_rows(rows: list[dict], columns: list[str]) -> Table:
    """Table with masked missing values, as returned by astropy/pyvo."""
    data = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        mask = [v is None for v in values]
        if all(v is None or isinstance(v, str) for v in values):
            filled = ["" if v is None else v for v in values]
        else:
            filled = [0.0 if v is None else v for v in values]
        data[column] = MaskedColumn(filled, mask=mask)
    return Table(data)


PLANET_COLUMNS = [
    "pl_name",
    "hostname",
    "ra",
    "dec",
    "discoverymethod",
    "disc_year",
    "pl_orbper",
    "pl_orbper_reflink",
    "pl_rade",
    "pl_radeerr1",
    "pl_radeerr2",
    "pl_rade_reflink",
    "pl_bmasse",
    "pl_bmasseerr1",
    "pl_bmasseerr2",
    "pl_bmasse_reflink",
    "st_teff",
    "st_teff_reflink",
    "st_rad",
    "st_rad_reflink",
]

PLANET_ROWS = [
    {
        "pl_name": "TOI-700 d",
        "hostname": "TOI-700",
        "ra": TOI700_RA,
        "dec": TOI700_DEC,
        "discoverymethod": "Transit",
        "disc_year": 2020,
        "pl_orbper": 37.42,
        "pl_orbper_reflink": "Orbital period source",
        "pl_rade": 1.19,
        "pl_radeerr1": 0.1,
        "pl_radeerr2": -0.09,
        "pl_rade_reflink": '<a refstr=GILBERT_ET_AL__2020 href="/x">Gilbert et al. 2020</a>',
        "pl_bmasse": None,
        "pl_bmasseerr1": None,
        "pl_bmasseerr2": None,
        "pl_bmasse_reflink": None,
        "st_teff": 3480.0,
        "st_teff_reflink": "Stellar temperature source",
        "st_rad": 0.42,
        "st_rad_reflink": "Stellar radius source",
    },
    {
        "pl_name": "HD 209458 b",
        "hostname": "HD 209458",
        "ra": 330.8,
        "dec": 18.9,
        "discoverymethod": "Transit",
        "disc_year": 1999,
        "pl_orbper": 3.52,
        "pl_orbper_reflink": "Henry et al. 2000",
        "pl_rade": 1.38,
        "pl_radeerr1": 0.05,
        "pl_radeerr2": -0.05,
        "pl_rade_reflink": "Henry et al. 2000",
        "pl_bmasse": 0.69,
        "pl_bmasseerr1": 0.02,
        "pl_bmasseerr2": -0.02,
        "pl_bmasse_reflink": "Henry et al. 2000",
        "st_teff": 6065.0,
        "st_teff_reflink": "Henry et al. 2000",
        "st_rad": 1.15,
        "st_rad_reflink": "Henry et al. 2000",
    },
]


def fake_fetch_planets(rows: list[dict] | None = None):
    rows = PLANET_ROWS if rows is None else rows

    def fetch(limit=None, *, hostname=None, require_radius=True, settings=None):
        selected = [r for r in rows if hostname is None or r["hostname"] == hostname]
        if require_radius:
            selected = [r for r in selected if r["pl_rade"] is not None]
        if limit:
            selected = selected[:limit]
        table = table_from_rows(selected, PLANET_COLUMNS)
        info = QueryInfo(
            service="fake NASA",
            tap_url="fake://nasa",
            table="pscomppars",
            query=f"fake hostname={hostname} limit={limit}",
            retrieved_at=utc_now(),
            n_rows=len(table),
        )
        return table, info

    return fetch


def fake_search_mast(ra: float = TOI700_RA, dec: float = TOI700_DEC, target_name: str = TIC_ID):
    def search(target, mission):
        return Table(
            {
                "obsid": [1001, 1002],
                "obs_id": ["tess-s0004", "tess-s0005"],
                "obs_collection": [mission, mission],
                "dataproduct_type": ["timeseries", "timeseries"],
                "instrument_name": ["TESS/Photometer", "TESS/Photometer"],
                "target_name": [target_name, target_name],
                "s_ra": [ra, ra],
                "s_dec": [dec, dec],
                "t_min": [58000.0, 58028.0],
                "t_max": [58027.0, 58055.0],
                "t_exptime": [120.0, 120.0],
                "sequence_number": [4, 5],
            }
        )

    return search


def fake_list_products(lc_source: Path):
    def list_products(observations, subgroups=None, max_products=None, settings=None):
        rows = [
            {
                "obsID": 1001,
                "productFilename": LC_NAME,
                "productType": "SCIENCE",
                "productSubGroupDescription": "LC",
                "dataURI": f"mast:TESS/product/{LC_NAME}",
            }
        ]
        table = table_from_rows(rows, list(rows[0]))
        return table[:max_products] if max_products else table

    return list_products


def fake_download(lc_source: Path, status: str = "COMPLETE"):
    def download(products, dest_dir):
        dest_dir = Path(dest_dir)
        manifest_rows = []
        for name in products["productFilename"]:
            local = dest_dir / str(name)
            if status == "COMPLETE":
                shutil.copy2(lc_source, local)
            message = "" if status == "COMPLETE" else "simulated failure"
            manifest_rows.append((str(local), status, message, ""))
        return Table(rows=manifest_rows, names=("Local Path", "Status", "Message", "URL"))

    return download


def fake_simbad(identifiers=None, ra: float = TOI700_RA, dec: float = TOI700_DEC):
    idents = (
        (("TIC", TIC_ID), ("Gaia DR3", "5489213108398512000"), ("HD", "235088"))
        if identifiers is None
        else identifiers
    )

    def resolve(name):
        return SimbadResult(
            query_name=name,
            main_id=name,
            ra_deg=ra,
            dec_deg=dec,
            identifiers=tuple(idents),
        )

    return resolve


def fake_search_eso(ra: float = TOI700_RA, dec: float = TOI700_DEC):
    def search(ra_deg, dec_deg, radius_arcsec=10.0, settings=None, limit=50):
        rows = [
            {
                "obs_id": "2019-ESO-HARPS-0001",
                "obs_collection": "HARPS",
                "instrument_name": "HARPS",
                "target_name": "TOI-700",
                "s_ra": ra,
                "s_dec": dec,
                "t_min": 58500.0,
                "t_exptime": 900.0,
                "dataproduct_type": "spectrum",
            },
            {
                "obs_id": "2020-ESO-HARPS-0002",
                "obs_collection": "HARPS",
                "instrument_name": "HARPS",
                "target_name": "TOI-700",
                "s_ra": ra,
                "s_dec": dec,
                "t_min": 58800.0,
                "t_exptime": 900.0,
                "dataproduct_type": "spectrum",
            },
        ]
        table = table_from_rows(
            rows,
            [
                "obs_id",
                "obs_collection",
                "instrument_name",
                "target_name",
                "s_ra",
                "s_dec",
                "t_min",
                "t_exptime",
                "dataproduct_type",
            ],
        )
        info = QueryInfo("fake ESO", "fake://eso", "ivoa.obscore", "fake", utc_now(), len(table))
        return table, info

    return search


def make_clients(
    lc_source: Path,
    *,
    planet_rows: list[dict] | None = None,
    mast_ra: float = TOI700_RA,
    mast_dec: float = TOI700_DEC,
    download_status: str = "COMPLETE",
    simbad_identifiers=None,
) -> ArchiveClients:
    """Fake clients provide local data for all normally networked calls."""
    return ArchiveClients(
        fetch_planets=fake_fetch_planets(planet_rows),
        search_mast=fake_search_mast(mast_ra, mast_dec),
        list_products=fake_list_products(lc_source),
        download_products=fake_download(lc_source, download_status),
        search_eso=fake_search_eso(),
        resolve_simbad=fake_simbad(simbad_identifiers),
    )

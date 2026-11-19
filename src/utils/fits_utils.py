"""Inspect FITS light curves (TESS, Kepler, K2): HDUs, headers, time, flux, errors, flags.

Files are opened read-only. The inspector does not delete, normalize, or change data;
it records contents and reports errors and warnings. Later stages decide how to process them.

For TESS, TIME is BTJD = BJD_TDB − 2457000; for Kepler, TIME is BKJD = BJD_TDB − 2454833.
Read the reference epoch from BJDREFI/BJDREFF or MJDREF headers; do not apply a universal offset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

TIME_COLUMN = "TIME"
QUALITY_COLUMN = "QUALITY"
FLUX_CANDIDATES = ("PDCSAP_FLUX", "SAP_FLUX")
ERROR_FOR_FLUX = {"PDCSAP_FLUX": "PDCSAP_FLUX_ERR", "SAP_FLUX": "SAP_FLUX_ERR"}

# Expected BJDREFI by mission, used for warnings rather than corrections.
EXPECTED_BJD_REFI = {"TESS": 2457000, "KEPLER": 2454833}

LC_EXTNAME = "LIGHTCURVE"
JD_MINUS_MJD = 2400000.5


@dataclass(frozen=True)
class HDUInfo:
    index: int
    name: str
    kind: str
    n_rows: int | None
    columns: tuple[str, ...]


@dataclass(frozen=True)
class TimeReference:
    """Time-scale keywords from a header, or None when absent."""

    timesys: str | None
    timeunit: str | None
    timeref: str | None
    bjdrefi: float | None
    bjdreff: float | None
    mjdref: float | None
    timezero: float | None

    @property
    def reference_jd(self) -> float | None:
        """Reference epoch in Julian days (JD), or None if unavailable."""
        if self.bjdrefi is not None:
            return self.bjdrefi + (self.bjdreff or 0.0)
        if self.mjdref is not None:
            return self.mjdref + JD_MINUS_MJD
        return None


@dataclass
class LightCurveCheck:
    file_name: str
    mission: str | None
    target_id: str | None
    sequence: str | None
    hdus: list[HDUInfo] = field(default_factory=list)
    lc_hdu_name: str | None = None
    lc_columns: list[str] = field(default_factory=list)
    n_cadences: int | None = None
    time_reference: TimeReference | None = None
    flux_column: str | None = None
    error_column: str | None = None
    quality_column_present: bool = False
    time_min: float | None = None
    time_max: float | None = None
    time_start_jd: float | None = None
    time_end_jd: float | None = None
    time_monotonic: bool | None = None
    n_nan_time: int = 0
    n_nan_flux: int = 0
    n_nan_error: int = 0
    n_quality_flagged: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.errors:
            return "error"
        if self.warnings:
            return "warning"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status
        return data


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_time_reference(header: fits.Header) -> TimeReference:
    """Read time-scale keywords from a header."""
    return TimeReference(
        timesys=_as_str(header.get("TIMESYS")),
        timeunit=_as_str(header.get("TIMEUNIT")),
        timeref=_as_str(header.get("TIMEREF")),
        bjdrefi=_as_float(header.get("BJDREFI")),
        bjdreff=_as_float(header.get("BJDREFF")),
        mjdref=_as_float(header.get("MJDREF")),
        timezero=_as_float(header.get("TIMEZERO")),
    )


def _find_lc_hdu(hdul: fits.HDUList) -> int | None:
    """Find the light-curve HDU by EXTNAME, then by its TIME column."""
    for i, hdu in enumerate(hdul):
        if hdu.name.upper() == LC_EXTNAME and isinstance(hdu, fits.BinTableHDU):
            return i
    for i, hdu in enumerate(hdul):
        if isinstance(hdu, fits.BinTableHDU) and hdu.columns is not None:
            if TIME_COLUMN in [n.upper() for n in hdu.columns.names]:
                return i
    return None


def _describe_hdus(hdul: fits.HDUList) -> list[HDUInfo]:
    infos = []
    for i, hdu in enumerate(hdul):
        columns: tuple[str, ...] = ()
        n_rows = None
        if isinstance(hdu, fits.BinTableHDU):
            columns = tuple(hdu.columns.names)
            n_rows = int(hdu.header.get("NAXIS2", 0))
        infos.append(
            HDUInfo(index=i, name=hdu.name, kind=type(hdu).__name__, n_rows=n_rows, columns=columns)
        )
    return infos


def inspect_lightcurve(path: str | Path) -> LightCurveCheck:
    """Validate a FITS light curve without changing the source file."""
    path = Path(path)
    result = LightCurveCheck(file_name=path.name, mission=None, target_id=None, sequence=None)

    with fits.open(path, mode="readonly", memmap=False) as hdul:
        result.hdus = _describe_hdus(hdul)
        primary = hdul[0].header

        mission = _as_str(primary.get("TELESCOP")) or _as_str(primary.get("MISSION"))
        result.mission = mission.upper() if mission else None
        result.target_id = (
            _as_str(primary.get("TICID"))
            or _as_str(primary.get("KEPLERID"))
            or _as_str(primary.get("OBJECT"))
        )
        result.sequence = next(
            (
                _as_str(primary.get(key))
                for key in ("SECTOR", "QUARTER", "CAMPAIGN")
                if _as_str(primary.get(key)) is not None
            ),
            None,
        )

        lc_index = _find_lc_hdu(hdul)
        if lc_index is None:
            result.errors.append(
                "Light-curve table not found: no HDU with EXTNAME=LIGHTCURVE and a TIME column."
            )
            return result

        lc_hdu = hdul[lc_index]
        result.lc_hdu_name = lc_hdu.name
        result.lc_columns = list(lc_hdu.columns.names)
        columns_upper = {name.upper(): name for name in result.lc_columns}

        # Read time keys from the light-curve HDU, falling back to the primary header.
        time_ref = read_time_reference(lc_hdu.header)
        if time_ref.reference_jd is None:
            time_ref = read_time_reference(primary)
        result.time_reference = time_ref
        if time_ref.timesys is None:
            result.warnings.append("TIMESYS is missing: time scale is unknown.")
        if time_ref.reference_jd is None:
            result.warnings.append(
                "BJDREFI/BJDREFF or MJDREF is missing: reference epoch is unknown."
            )
        expected = EXPECTED_BJD_REFI.get(result.mission or "")
        if (
            expected is not None
            and time_ref.bjdrefi is not None
            and int(time_ref.bjdrefi) != expected
        ):
            result.warnings.append(
                f"BJDREFI={int(time_ref.bjdrefi)} differs from the expected {result.mission} value "
                f"({expected}). Check product documentation before using its time values."
            )

        if TIME_COLUMN not in columns_upper:
            result.errors.append("The light-curve table has no TIME column.")
            return result

        # Cast TIME to float for reliable NaN and ordering checks.
        data = lc_hdu.data
        result.n_cadences = int(len(data))
        if result.n_cadences == 0:
            result.errors.append("The light-curve table is empty (0 rows).")
            return result

        time = np.asarray(data[columns_upper[TIME_COLUMN]], dtype=float)
        result.n_nan_time = int(np.count_nonzero(~np.isfinite(time)))
        finite_time = time[np.isfinite(time)]
        if finite_time.size == 0:
            result.errors.append("All TIME values are non-finite (NaN/inf).")
            return result

        result.time_min = float(finite_time.min())
        result.time_max = float(finite_time.max())
        if time_ref.reference_jd is not None:
            result.time_start_jd = time_ref.reference_jd + result.time_min
            result.time_end_jd = time_ref.reference_jd + result.time_max
        result.time_monotonic = bool(np.all(np.diff(finite_time) >= 0))
        if not result.time_monotonic:
            result.warnings.append("TIME values are not monotonic; rows may be out of order.")
        if result.n_nan_time:
            result.warnings.append(
                f"Rows with NaN in TIME: {result.n_nan_time} of {result.n_cadences}."
            )

        # Prefer systematics-corrected PDCSAP flux, then SAP.
        flux_name = next((c for c in FLUX_CANDIDATES if c in columns_upper), None)
        if flux_name is None:
            result.errors.append("Flux column missing (expected PDCSAP_FLUX or SAP_FLUX).")
        else:
            result.flux_column = columns_upper[flux_name]
            flux = np.asarray(data[result.flux_column], dtype=float)
            result.n_nan_flux = int(np.count_nonzero(~np.isfinite(flux)))
            if result.n_nan_flux == result.n_cadences:
                result.errors.append(f"Column {result.flux_column} contains only NaN values.")
            elif result.n_nan_flux:
                pct = 100.0 * result.n_nan_flux / result.n_cadences
                result.warnings.append(
                    f"NaN in {result.flux_column}: {result.n_nan_flux} ({pct:.1f}%)."
                )

            err_key = ERROR_FOR_FLUX[flux_name]
            if err_key in columns_upper:
                result.error_column = columns_upper[err_key]
                err = np.asarray(data[result.error_column], dtype=float)
                result.n_nan_error = int(np.count_nonzero(~np.isfinite(err)))
            else:
                result.warnings.append(f"Error column {err_key} is missing for {flux_name}.")

        # Count quality flags without removing data; nonzero flags are not errors.
        if QUALITY_COLUMN in columns_upper:
            result.quality_column_present = True
            quality = np.asarray(data[columns_upper[QUALITY_COLUMN]])
            result.n_quality_flagged = int(np.count_nonzero(quality))
        else:
            result.warnings.append("QUALITY column is missing: quality flags were not checked.")

    return result


def format_report(check: LightCurveCheck) -> str:
    """Short text report for logs and console output."""
    lines = [
        f"File: {check.file_name}",
        f"Mission: {check.mission} | target: {check.target_id} | sector/quarter: {check.sequence}",
        f"Table: {check.lc_hdu_name} | rows: {check.n_cadences} | "
        f"columns: {', '.join(check.lc_columns)}",
    ]
    if check.time_reference is not None:
        ref = check.time_reference
        lines.append(
            f"Time: TIMESYS={ref.timesys}, TIMEUNIT={ref.timeunit}, "
            f"reference JD={ref.reference_jd}, TIME range={check.time_min}..{check.time_max}"
        )
    if check.flux_column:
        lines.append(
            f"Flux: {check.flux_column} (NaN: {check.n_nan_flux}) | "
            f"errors: {check.error_column or '—'} | QUALITY flags != 0: {check.n_quality_flagged}"
        )
    lines.append(f"Status: {check.status}")
    lines += [f"  error: {e}" for e in check.errors]
    lines += [f"  warning: {w}" for w in check.warnings]
    return "\n".join(lines)

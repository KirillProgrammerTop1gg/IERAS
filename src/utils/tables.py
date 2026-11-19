"""Helpers for astropy tables where missing values are often masked or NaN."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _has_column(row: Any, column: str) -> bool:
    colnames = getattr(row, "colnames", None)
    if colnames is None:
        dtype = getattr(row, "dtype", None)
        colnames = dtype.names if dtype is not None else None
    return colnames is None or column in colnames


def cell(row: Any, column: str) -> Any:
    """Return a cell as a plain Python value or None for missing, masked, or NaN values.

    Return None when a column is absent instead of raising KeyError.
    """
    if not _has_column(row, column):
        return None
    try:
        value = row[column]
    except (KeyError, IndexError, ValueError):
        return None

    if value is np.ma.masked:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def cell_float(row: Any, column: str) -> float | None:
    value = cell(row, column)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cell_str(row: Any, column: str) -> str | None:
    value = cell(row, column)
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

"""Configure console and UTF-8 file logging under logs/stage1_<date>.log."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

LOGGER_NAME = "exoplanet_ai"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(logs_dir: Path, level: str = "INFO") -> logging.Logger:
    """Return the project logger without adding duplicate handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    file_handler = logging.FileHandler(logs_dir / f"stage1_{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger, for example get_logger('mast')."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")

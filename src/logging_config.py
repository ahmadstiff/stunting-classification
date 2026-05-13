"""
Logging configuration for the stunting classification pipeline.

Usage:
    from src.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")

A single top-level call to `configure_logging()` (performed lazily on first
call to `get_logger`) attaches a human-readable console handler plus an
optional file handler.  Re-importing the modules does not add duplicate
handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED: bool = False


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[os.PathLike | str] = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """Configure the root logger for the pipeline.

    Parameters
    ----------
    level : int, default logging.INFO
        Minimum log level for the console handler.
    log_file : path-like, optional
        If provided, additionally write logs to this file (append mode).
    fmt, datefmt : str
        Standard ``logging.Formatter`` arguments.

    Notes
    -----
    Safe to call multiple times: the function is idempotent so subsequent
    calls with the same parameters will not attach duplicate handlers.
    """
    global _CONFIGURED

    root = logging.getLogger()
    # Avoid adding duplicate handlers on repeated import.
    if _CONFIGURED and root.handlers:
        return

    # Clear any pre-existing handlers that may have been configured by
    # libraries (e.g. notebook runtimes) so formatting stays consistent.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(logging.DEBUG)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the root logger on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)

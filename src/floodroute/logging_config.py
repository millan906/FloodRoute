"""Structured logging configuration for FloodRoute."""

from __future__ import annotations

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: LogLevel = "INFO") -> None:
    """Configure root logger with a consistent format.

    Call once at application startup (CLI entrypoint).
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stderr,
    )
    # Silence overly verbose third-party loggers at WARNING+
    for noisy in ("urllib3", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the floodroute namespace."""
    return logging.getLogger(f"floodroute.{name}")

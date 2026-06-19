"""Logging setup for the OVIOVO system."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str = "") -> logging.Logger:
    """Configure root logger for the oviovo system.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file. Empty string = console only.

    Returns:
        Configured root logger.
    """
    root_logger = logging.getLogger("oviovo")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger

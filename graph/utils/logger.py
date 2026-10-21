"""Centralized logging configuration for the :mod:`graph` package.

Provides a single, idempotent console handler for the ``graph`` logger
namespace so every node emits consistent, timestamped records at ``INFO`` level
or above (e.g. ``graph.nodes.supervisor``, ``graph.nodes.market_data``).

Usage::

    from graph.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys

#: Console log format: timestamp, level, logger name, message.
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"

#: Namespace logger that acts as the single configuration point.
NAMESPACE = "graph"

#: Default level for the graph namespace.
DEFAULT_LEVEL = logging.INFO

_configured = False


def setup_logging(level: int = DEFAULT_LEVEL) -> None:
    """Configure the ``graph`` namespace logger with a console handler.

    Idempotent: repeated calls (re-imports, multiple test runs) do not add
    duplicate handlers.
    """
    global _configured
    if _configured:
        return

    logger = logging.getLogger(NAMESPACE)
    logger.setLevel(level)
    logger.propagate = False  # keep graph logs off the root/last-resort handler

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``graph`` namespace, configuring logging first.

    ``name`` may be a full dotted name (e.g. ``graph.nodes.supervisor``) or a
    short name (e.g. ``nodes.supervisor``); the ``graph.`` prefix is added
    automatically when missing.
    """
    setup_logging()
    if name != NAMESPACE and not name.startswith(f"{NAMESPACE}."):
        name = f"{NAMESPACE}.{name}"
    return logging.getLogger(name)


__all__ = ["LOG_FORMAT", "DEFAULT_LEVEL", "setup_logging", "get_logger"]

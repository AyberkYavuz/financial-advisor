"""Shared pytest fixtures for the financial-advisor test suite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure the root-level ``graph`` package is importable regardless of how
# pytest is invoked.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def compiled_graph():
    """Load the compiled graph from the root ``graph.py``.

    Mirrors how ``langgraph dev`` resolves ``./graph.py:graph`` (via
    ``spec_from_file_location``), so we exercise the real production entry point.
    """
    path = PROJECT_ROOT / "graph.py"
    modname = "graph_module_under_test"
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module.graph

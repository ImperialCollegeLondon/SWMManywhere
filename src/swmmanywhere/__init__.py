"""The main module for MyProject."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("swmmanywhere")
except PackageNotFoundError:
    __version__ = "999"

# Importing module to register the graphfcns and made them available
from . import graphfcns  # noqa: F401

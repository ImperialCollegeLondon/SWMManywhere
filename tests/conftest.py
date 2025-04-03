from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pytest
from pywbt_source.tests import test_pywbt

pytest_plugins = ["pywbt_source.conftest"]


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with downloads."""
    if not config.getoption("markexpr", "False"):
        config.option.markexpr = "not downloads"


@pytest.fixture()
def wbt_path(wbt_zipfile):
    """Fixture to provide the path to the wbt zip file."""
    return str(Path(test_pywbt.__file__).parent / "wbt_zip" / wbt_zipfile)

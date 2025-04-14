from __future__ import annotations

from pathlib import Path

import pytest
from pywbt.pywbt import _get_platform_suffix


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with downloads."""
    if not config.getoption("markexpr", "False"):
        config.option.markexpr = "not downloads"


@pytest.fixture
def wbt_zip_path() -> Path:
    """Determine the platform specific binary for whiteboxtools for testing.

    All WBT binaries are stored in `wbt_zip` within the `tests` directory.

    Based on implementation in [`pywbt`](https://github.com/cheginit/pywbt).
    """
    _, suffix, _ = _get_platform_suffix()

    return Path(__file__).parent / "wbt_zip" / f"WhiteboxTools_{suffix}.zip"

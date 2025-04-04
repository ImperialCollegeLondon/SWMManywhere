from __future__ import annotations

import platform
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with downloads."""
    if not config.getoption("markexpr", "False"):
        config.option.markexpr = "not downloads"


@pytest.fixture
def wbt_path() -> Path:
    """Determine the platform specific binary for whiteboxtools for testing.

    All WBT binaries are stored in `wbt_zip` within the `tests` directory.

    Based on implementation in [`pywbt`](https://github.com/cheginit/pywbt).
    """
    system = platform.system()
    base_name = "WhiteboxTools_{}.zip"
    if system not in ("Windows", "Darwin", "Linux"):
        raise ValueError(f"Unsupported operating system: {system}")

    if system == "Windows":
        suffix = "win_amd64"
    elif system == "Darwin":
        suffix = "darwin_m_series" if platform.machine() == "arm64" else "darwin_amd64"
    else:
        suffix = (
            "linux_musl" if "musl" in platform.libc_ver()[0].lower() else "linux_amd64"
        )
    return Path(__file__).parent / "wbt_zip" / base_name.format(suffix)

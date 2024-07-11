from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with downloads."""
    if not config.getoption("markexpr", "False"):
        config.option.markexpr = "not downloads"

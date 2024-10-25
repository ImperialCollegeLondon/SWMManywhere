"""Integration test for the swmmanywhere command-line interface."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

from swmmanywhere import __main__
from swmmanywhere.logging import logger


def test_swmmanywhere_cli(tmp_path):
    """Test that the CLI can successfully run with an actual configuration."""
    base_dir = Path(tmp_path)

    # Define minimum viable config
    config = {
        "base_dir": str(base_dir),
        "project": "my_first_swmm",
        "bbox": [1.52740, 42.50524, 1.54273, 42.51259],
    }

    config_path = base_dir / "config.yml"
    with config_path.open("w") as config_file:
        yaml.dump(config, config_file)

    # Mock sys.argv to simulate command-line arguments
    sys.argv = [
        "swmmanywhere",
        "--config_path",
        str(config_path),
        "--verbose",
        "True",
    ]

    expected = b"No real network provided, returning SWMM .inp file."

    with tempfile.NamedTemporaryFile(
        suffix=".log", mode="w+b", delete=False
    ) as temp_file:
        fid = Path(temp_file.name)
        logger.add(fid)

        # Run the CLI entry point
        __main__.run()

        # Capture the output
        assert expected in temp_file.read()
        logger.remove()

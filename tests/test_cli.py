"""Integration test for the swmmanywhere command-line interface."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

from swmmanywhere import __main__


def test_swmmanywhere_cli(capsys):
    """Test that the CLI can successfully run with an actual configuration."""
    with tempfile.TemporaryDirectory() as tempdir:
        base_dir = Path(tempdir)
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

        # Run the CLI entry point
        __main__.run()

        # Capture the output
        captured = capsys.readouterr()
        expected = "No real network provided, returning SWMM .inp file."
        assert expected in captured.out

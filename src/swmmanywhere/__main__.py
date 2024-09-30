"""The entry point for the swmmanywhere program."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from swmmanywhere import swmmanywhere
from swmmanywhere.logging import logger


def parse_arguments():
    """Parse the command line arguments.

    Returns:
        Path: The path to the configuration file.
    """
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path(__file__).parent.parent
        / "tests"
        / "test_data"
        / "demo_config.yml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Configuration verbosity"
    )
    args = parser.parse_args()
    return args.config_path, args.verbose


def run():
    """Run a swmmanywhere config file."""
    # Parse the arguments
    config_path, verbose = parse_arguments()
    os.environ["SWMMANYWHERE_VERBOSE"] = str(verbose).lower()

    config = swmmanywhere.load_config(config_path)

    # Run the model
    inp, metrics = swmmanywhere.swmmanywhere(config)
    logger.info(f"Model run complete. Results saved to {inp}")
    logger.info(f"Metrics:\n {metrics}")


if __name__ == "__main__":
    run()

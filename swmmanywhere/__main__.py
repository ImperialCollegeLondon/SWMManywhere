"""The entry point for the swmmanywhere program."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas
import shapely
from swmmanywhere import swmmanywhere
from swmmanywhere.parameter import FilePaths
from swmmanywhere.post_processing import synthetic_write
from swmmanywhere.custom_logging import logger


def parse_arguments():
    """Parse the command line arguments.

    Returns:
        Path: The path to the configuration file.
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--config_path', 
                        type=Path, 
                        default=Path(__file__).parent.parent / 'tests' /\
                                    'test_data' / 'demo_config.yml',
                        help='Configuration file path')
    parser.add_argument('--verbose', 
                        type=bool, 
                        default=False,
                        help='Configuration verbosity')
    parser.add_argument('--write', 
                        type=Path, 
                        default=False,
                        help='Configuration verbosity')
    args = parser.parse_args()
    return args.config_path, args.verbose, args.write

def write_wrap(write):
    """Write the nodes, subs, and edges to the config file."""
    edges = write / 'edges.geojson'
    nodes = write / 'nodes.geojson'
    subs = write / 'subcatchments.geojson'
    precip = write / 'storm.dat'
    
    if not precip.exists():
        raise FileNotFoundError(f"File not found: {precip}")
    if not edges.exists():
        raise FileNotFoundError(f"File not found: {edges}")
    if not nodes.exists():
        raise FileNotFoundError(f"File not found: {nodes}")
    if not subs.exists():
        raise FileNotFoundError(f"File not found: {subs}")
    
    addresses = FilePaths(base_dir = None,project_name = None,
                    bbox_number = None,model_number = None,extension = 'json')
    addresses.edges = edges
    addresses.nodes = nodes
    addresses.subcatchments = subs
    addresses.precipitation = precip
    addresses.inp = write / 'model.inp'
    synthetic_write(addresses)
    return addresses.inp

def run():
    """Run a swmmanywhere config file."""
    # Parse the arguments
    config_path, verbose, write = parse_arguments()
    os.environ["SWMMANYWHERE_VERBOSE"] = str(verbose).lower()
    if write:
        inp = write_wrap(write)
        logger.info(f"Nodes, subs, and edges written to {inp}")
        return

    config = swmmanywhere.load_config(config_path)

    # Run the model
    inp, metrics = swmmanywhere.swmmanywhere(config)
    logger.info(f"Model run complete. Results saved to {inp}")
    logger.info(f"Metrics:\n {metrics}")

if __name__ == '__main__':
    run()

"""Copy the test data files to a given directory."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from swmmanywhere.geospatial_utilities import graph_to_geojson
from swmmanywhere.graph_utilities import load_graph


def copy_test_data(fid: Path):
    """Copy the test data files to a given directory.

    Args:
        fid (Path): Directory to copy the test data files to.
    """
    if not fid.exists():
        raise FileNotFoundError(f"Directory does not exist: {fid}")

    defs_dir = Path(__file__).parent
    files = [
        "bellinge_small.inp",
        "bellinge_small_graph.json",
        "bellinge_small_subcatchments.geojson",
        "storm.dat",
    ]
    for filename in files:
        copyfile(defs_dir / filename, fid / filename)

    G = load_graph(fid / "bellinge_small_graph.json")
    graph_to_geojson(G, fid / "nodes.geojson", fid / "edges.geojson", G.graph["crs"])

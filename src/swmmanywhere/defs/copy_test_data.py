"""Copy the test data files to a given directory."""
from pathlib import Path
from shutil import copyfile

from swmmanywhere.graph_utilities import load_graph
from swmmanywhere.geospatial_utilities import graph_to_geojson

def copy_test_data(fid: Path):
    """Copy the test data files to a given directory.
    
    Args:
        fid (Path): Directory to copy the test data files to.
    """

    if not fid.exists():
        raise FileNotFoundError(f"Directory does not exist: {fid}")
    
    defs_dir = Path(__file__).parent
    copyfile(defs_dir / "bellinge_small.inp", fid / "bellinge_small.inp")
    copyfile(defs_dir / "bellinge_small_graph.json", fid / "bellinge_small_graph.json")
    copyfile(defs_dir / "bellinge_small_subcatchments.geojson", fid / "bellinge_small_subcatchments.geojson")
    copyfile(defs_dir / "storm.dat", fid / "storm.dat")

    G = load_graph(fid / "bellinge_small_graph.json")
    graph_to_geojson(G, fid / "nodes.geojson", fid / "edges.geojson", G.graph["crs"])
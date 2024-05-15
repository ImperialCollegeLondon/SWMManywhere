"""Misc line to trigger workflow."""
from __future__ import annotations

from pathlib import Path

from swmmanywhere import graph_utilities as ge


def load_street_network():
    """Load a street network."""
    G = ge.load_graph(Path(__file__).parent / 'test_data' / 'street_graph.json')
    return G

def test_load():
    """Test load."""
    G = load_street_network()
    assert G is not None
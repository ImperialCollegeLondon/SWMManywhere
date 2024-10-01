"""Load example data."""

from __future__ import annotations

from pathlib import Path

from swmmanywhere.graph_utilities import load_graph

demo_graph = load_graph(Path(__file__).parent / "demo_graph.json")

from __future__ import annotations

import networkx as nx
import pytest

from swmmanywhere.shortest_path_utils import dijkstra_pq, tarjans_pq


def test_simple_graph():
    """Test case 1: Simple connected graph."""
    G = nx.MultiDiGraph()
    G.add_edges_from([(2, 1), (3, 1), (4, 2), (4, 3), (1, "waste")])
    nx.set_edge_attributes(
        G,
        {
            (2, 1, 0): {"weight": 1, "edge_type": "outfall"},
            (3, 1, 0): {"weight": 2, "edge_type": "street"},
            (4, 2, 0): {"weight": 3, "edge_type": "street"},
            (4, 3, 0): {"weight": 4, "edge_type": "street"},
            (1, "waste", 0): {"weight": 0, "edge_type": "outfall"},
        },
    )
    root = "waste"
    mst = tarjans_pq(G, root)
    assert set(mst.edges) == {(1, "waste", 0), (2, 1, 0), (4, 2, 0), (3, 1, 0)}
    assert mst.size() == 4

    djg = dijkstra_pq(G, [root])
    assert set(djg.edges) == {(1, "waste", 0), (2, 1, 0), (4, 2, 0), (3, 1, 0)}
    assert djg.size() == 4


def test_disconnected():
    """Test case 2: Disconnected graph."""
    G = nx.MultiDiGraph()
    G.add_edges_from(
        [(1, "waste"), (2, 1), (3, 1), (5, 4), (6, 5)], weight=1, edge_type="street"
    )
    root = "waste"
    with pytest.raises(ValueError):
        tarjans_pq(G, root)

    djg = dijkstra_pq(G, [root])
    assert set(djg.edges) == {(1, "waste", 0), (2, 1, 0), (3, 1, 0)}


def test_parallel():
    """Test case 3: Graph with parallel edges."""
    G = nx.MultiDiGraph()
    G.add_edges_from(
        [(2, 1, 0), (2, 1, 1), (3, 1, 0), (4, 2, 0), (4, 3, 0)],
        edge_type="street",
        weight=1,
    )
    root = 1
    mst = tarjans_pq(G, root)
    assert set(mst.edges) == {(2, 1, 0), (4, 2, 0), (3, 1, 0)}
    assert mst.size() == 3

    djg = dijkstra_pq(G, [root])
    # Currently paths are defined as node-to-node and so ignore keys .. TODO?
    assert set(djg.edges) == {(2, 1, 0), (2, 1, 1), (3, 1, 0), (4, 2, 0)}


def test_selfloop():
    """Test case 4: Graph with self-loops."""
    G = nx.MultiDiGraph()
    G.add_edges_from(
        [(2, 1, 0), (3, 1, 0), (4, 2, 0), (4, 3, 0), (2, 4, 0)],
        edge_type="street",
        weight=1,
    )
    G.add_edge(3, 4, weight=1, edge_type="street")
    root = 1
    mst = tarjans_pq(G, root)
    assert set(mst.edges) == {(2, 1, 0), (4, 2, 0), (3, 1, 0)}
    assert mst.size() == 3

    djg = dijkstra_pq(G, [root])
    assert set(djg.edges) == {(2, 1, 0), (4, 2, 0), (3, 1, 0)}
    assert djg.size() == 3


def test_custom_weight():
    """Test case 5: Graph with custom weight attribute."""
    G = nx.MultiDiGraph()
    G.add_edges_from(
        [(2, 1, 0), (3, 1, 0), (4, 2, 0), (4, 3, 0)], edge_type="street", cost=1
    )
    root = 1
    mst = tarjans_pq(G, root, weight_attr="cost")
    assert set(mst.edges) == {(2, 1, 0), (4, 2, 0), (3, 1, 0)}
    assert mst.size() == 3

    djg = dijkstra_pq(G, [root], weight_attr="cost")
    assert set(djg.edges) == {(2, 1, 0), (4, 2, 0), (3, 1, 0)}
    assert djg.size() == 3


def test_empty():
    """Test case 6: Empty graph."""
    G = nx.MultiDiGraph()
    root = 1
    with pytest.raises(KeyError):
        tarjans_pq(G, root)
    with pytest.raises(nx.NetworkXError):
        dijkstra_pq(G, [root])

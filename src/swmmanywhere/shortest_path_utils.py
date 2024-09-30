"""Utility functions for shortest path algorithms."""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Hashable

import networkx as nx


def tarjans_pq(
    G: nx.MultiDiGraph, root: int | str, weight_attr: str = "weight"
) -> nx.MultiDiGraph:
    """Tarjan's algorithm for a directed minimum spanning tree.

    Also known as a minimum spanning arborescence, this algorithm finds the
    minimum directed spanning tree rooted at a given vertex (root) in a directed
    graph.

    Args:
        G (nx.MultiDiGraph): The input graph.
        root (int | str): The root node (i.e., that all vertices in the graph
            should flow to).
        weight_attr (str): The name of the edge attribute containing the edge
            weights. Defaults to 'weight'.

    Returns:
        nx.MultiDiGraph: The directed minimum spanning tree.
    """
    # Copy the graph and relabel the nodes
    G_ = G.copy()
    new_nodes = {node: i for i, node in enumerate(G.nodes)}
    node_mapping = {i: node for i, node in enumerate(G.nodes)}
    G_ = nx.relabel_nodes(G_, new_nodes)

    # Extract the new root label, edges and weights
    root = new_nodes[root]
    edges = [(u, v, d[weight_attr]) for u, v, d in G_.edges(data=True)]

    # Initialize data structures
    graph = defaultdict(list)
    for u, v, weight in edges:
        graph[v].append((u, weight))

    n = len(G.nodes)
    parent = {}  # Parent pointers for the MST
    in_edge_pq: list = []  # Priority queue to store incoming edges

    # Initialize the priority queue with edges incoming to the root
    for u, weight in graph[root]:
        heapq.heappush(in_edge_pq, (weight, u, root))

    mst_edges = []
    mst_weight = 0
    outfalls: dict = {}
    while in_edge_pq:
        weight, u, v = heapq.heappop(in_edge_pq)

        if u not in parent:
            # If v is not in the MST yet, add the edge (u, v) to the MST
            parent[u] = v
            mst_edges.append((u, v))
            mst_weight += weight

            if v in outfalls:
                outfalls[u] = outfalls[v]

            elif G_.get_edge_data(u, v)[0]["edge_type"] == "outfall":
                outfalls[u] = node_mapping[u]

            # Add incoming edges to v to the priority queue
            for w, weight_new in graph[u]:
                heapq.heappush(in_edge_pq, (weight_new, w, u))

    # Check if all vertices are reachable from the root
    if len(parent) != n - 1:
        raise ValueError("Graph is not connected or has multiple roots.")

    new_graph = nx.MultiDiGraph()

    for u, v in mst_edges:
        d = G_.get_edge_data(u, v)[0]
        new_graph.add_edge(u, v, **d)

    for u, d in G_.nodes(data=True):
        new_graph.nodes[u].update(d)

    nx.set_node_attributes(new_graph, outfalls, "outfall")
    new_graph = nx.relabel_nodes(new_graph, node_mapping)
    new_graph.graph = G.graph.copy()
    return new_graph


def dijkstra_pq(
    G: nx.MultiDiGraph, outfalls: list, weight_attr: str = "weight"
) -> nx.MultiDiGraph:
    """Dijkstra's algorithm for shortest paths to outfalls.

    This function calculates the shortest paths from each node in the graph to
    the nearest outfall. The graph is modified to include the outfall
    and the shortest path length.

    Args:
        G (nx.MultiDiGraph): The input graph.
        outfalls (list): A list of outfall nodes.
        weight_attr (str): The name of the edge attribute containing the edge
            weights. Defaults to 'weight'.

    Returns:
        nx.MultiDiGraph: The graph with the shortest paths to outfalls.
    """
    G = G.copy()
    # Initialize the dictionary with infinity for all nodes
    shortest_paths = {node: float("inf") for node in G.nodes}

    # Initialize the dictionary to store the paths
    paths: dict[Hashable, list] = {node: [] for node in G.nodes}

    # Set the shortest path length to 0 for outfalls
    for outfall in outfalls:
        shortest_paths[outfall] = 0
        paths[outfall] = [outfall]

    # Initialize a min-heap with (distance, node) tuples
    heap = [(0, outfall) for outfall in outfalls]
    while heap:
        # Pop the node with the smallest distance
        dist, node = heapq.heappop(heap)

        # For each neighbor of the current node
        for neighbor, _, edge_data in G.in_edges(node, data=True):
            # Calculate the distance through the current node
            alt_dist = dist + edge_data[weight_attr]
            # If the alternative distance is shorter

            if alt_dist >= shortest_paths[neighbor]:
                continue

            # Update the shortest path length
            shortest_paths[neighbor] = alt_dist
            # Update the path
            paths[neighbor] = paths[node] + [neighbor]
            # Push the neighbor to the heap
            heapq.heappush(heap, (alt_dist, neighbor))

    # Remove nodes with no path to an outfall
    for node in [node for node, path in paths.items() if not path]:
        G.remove_node(node)
        del paths[node], shortest_paths[node]

    if len(G.nodes) == 0:
        raise ValueError("""No nodes with path to outfall, """)

    edges_to_keep: set = set()

    for path in paths.values():
        # Assign outfall
        outfall = path[0]
        for node in path:
            G.nodes[node]["outfall"] = outfall
            G.nodes[node]["shortest_path"] = shortest_paths[node]

        # Store path
        edges_to_keep.update(zip(path[1:], path[:-1]))

    # Remove edges not on paths
    new_graph = G.copy()
    for u, v in G.edges():
        if (u, v) not in edges_to_keep:
            new_graph.remove_edge(u, v)

    return new_graph

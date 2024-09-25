"""Module for graphfcns that change subcatchments."""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters, shortest_path_utils
from swmmanywhere.filepaths import FilePaths
from swmmanywhere.graph_utilities import (
    BaseGraphFunction,
    filter_streets,
    register_graphfcn,
)
from swmmanywhere.logging import logger


@register_graphfcn
class set_elevation(
    BaseGraphFunction,
    required_node_attributes=["x", "y"],
    adds_node_attributes=["surface_elevation"],
):
    """set_elevation class."""

    def __call__(self, G: nx.Graph, addresses: FilePaths, **kwargs) -> nx.Graph:
        """Set the elevation for each node.

        This function sets the elevation for each node. The elevation is
        calculated from the elevation data.

        Args:
            G (nx.Graph): A graph
            addresses (FilePaths): An FilePaths parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        x = [d["x"] for x, d in G.nodes(data=True)]
        y = [d["y"] for x, d in G.nodes(data=True)]
        elevations = go.interpolate_points_on_raster(
            x, y, addresses.bbox_paths.elevation
        )
        elevations_dict = {id_: elev for id_, elev in zip(G.nodes, elevations)}
        nx.set_node_attributes(G, elevations_dict, "surface_elevation")
        return G


@register_graphfcn
class set_surface_slope(
    BaseGraphFunction,
    required_node_attributes=["surface_elevation"],
    adds_edge_attributes=["surface_slope"],
):
    """set_surface_slope class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """Set the surface slope for each edge.

        This function sets the surface slope for each edge. The surface slope is
        calculated from the elevation data.

        Args:
            G (nx.Graph): A graph
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        # Compute the slope for each edge
        slope_dict = {
            (u, v, k): (
                G.nodes[u]["surface_elevation"] - G.nodes[v]["surface_elevation"]
            )
            / d["length"]
            for u, v, k, d in G.edges(data=True, keys=True)
        }

        # Set the 'surface_slope' attribute for all edges
        nx.set_edge_attributes(G, slope_dict, "surface_slope")
        return G


@register_graphfcn
class set_chahinian_slope(
    BaseGraphFunction,
    required_edge_attributes=["surface_slope"],
    adds_edge_attributes=["chahinian_slope"],
):
    """set_chahinian_slope class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """set_chahinian_slope class.

        This function sets the Chahinian slope for each edge. The Chahinian slope is
        calculated from the surface slope and weighted according to the slope
        (based on: https://doi.org/10.1016/j.compenvurbsys.2019.101370)

        Args:
            G (nx.Graph): A graph
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()

        # Values where the weight of the slope can be matched to the values
        # in weights - e.g., a slope of 0.3% has 0 weight (preferred), while
        # a slope of <=-1% has a weight of 1 (not preferred)
        slope_points = [-1, 0.3, 0.7, 10]
        weights = [1, 0, 0, 1]

        # Calculate weights
        slope = nx.get_edge_attributes(G, "surface_slope")
        weights = np.interp(
            np.asarray(list(slope.values())) * 100,
            slope_points,
            weights,
            left=1,
            right=1,
        )
        nx.set_edge_attributes(G, dict(zip(slope, weights)), "chahinian_slope")

        return G


@register_graphfcn
class set_chahinian_angle(
    BaseGraphFunction,
    required_node_attributes=["x", "y"],
    adds_edge_attributes=["chahinian_angle"],
):
    """set_chahinian_angle class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """Set the Chahinian angle for each edge.

        This function sets the Chahinian angle for each edge. The Chahinian angle is
        calculated from the geometry of the edge and weighted according to the
        angle (based on: https://doi.org/10.1016/j.compenvurbsys.2019.101370)

        Args:
            G (nx.Graph): A graph
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        # TODO - in a double directed graph, not sure how meaningful this is
        # TODO could probably refactor
        G = G.copy()
        for u, v, d in G.edges(data=True):
            min_weight = float("inf")
            for node in G.successors(v):
                if node == u:
                    continue

                p1 = (G.nodes[u]["x"], G.nodes[u]["y"])
                p2 = (G.nodes[v]["x"], G.nodes[v]["y"])
                p3 = (G.nodes[node]["x"], G.nodes[node]["y"])
                angle = go.calculate_angle(p1, p2, p3)
                chahinian_weight = np.interp(
                    angle,
                    [0, 90, 135, 180, 225, 270, 360],
                    [1, 0.2, 0.7, 0, 0.7, 0.2, 1],
                )
                min_weight = min(chahinian_weight, min_weight)
            if min_weight == float("inf"):
                min_weight = 0
            d["chahinian_angle"] = min_weight
        return G


@register_graphfcn
class calculate_weights(
    BaseGraphFunction,
    required_edge_attributes=parameters.TopologyDerivation().weights,
    adds_edge_attributes=["weight"],
):
    """calculate_weights class."""

    # TODO.. I guess if someone defines their own weights, this will need
    # to change, will want an automatic way to do that...

    def __call__(
        self, G: nx.Graph, topology_derivation: parameters.TopologyDerivation, **kwargs
    ) -> nx.Graph:
        """Calculate the weights for each edge.

        This function calculates the weights for each edge. The weights are
        calculated from the edge attributes.

        Args:
            G (nx.Graph): A graph
            topology_derivation (parameters.TopologyDerivation): A TopologyDerivation
                parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        # Calculate bounds to normalise between
        bounds: Dict[Any, List[float]] = defaultdict(lambda: [np.inf, -np.inf])

        for w in topology_derivation.weights:
            bounds[w][0] = min(nx.get_edge_attributes(G, w).values())  # lower bound
            bounds[w][1] = max(nx.get_edge_attributes(G, w).values())  # upper bound

        # Avoid division by zero
        bounds = {w: [b[0], b[1]] for w, b in bounds.items() if b[0] != b[1]}

        G = G.copy()
        eps = np.finfo(float).eps
        for u, v, d in G.edges(data=True):
            total_weight = 0
            for attr, bds in bounds.items():
                # Normalise
                weight = max((d[attr] - bds[0]) / (bds[1] - bds[0]), eps)
                # Exponent
                weight = weight ** getattr(topology_derivation, f"{attr}_exponent")
                # Scaling
                weight = weight * getattr(topology_derivation, f"{attr}_scaling")
                # Sum
                total_weight += weight
            # Set
            d["weight"] = total_weight
        return G


@register_graphfcn
class derive_topology(
    BaseGraphFunction,
    required_edge_attributes=[
        "edge_type",  # 'rivers' and 'streets'
        "weight",
    ],
    adds_node_attributes=["outfall", "shortest_path"],
):
    """derive_topology class."""

    def __call__(
        self, G: nx.Graph, outfall_derivation: parameters.OutfallDerivation, **kwargs
    ) -> nx.Graph:
        """Derive the topology of a graph.

        Derives the network topology based on the weighted graph of potential
        pipe carrying edges in the graph.

        Two methods are available:
        - `separate`: The original and that assumes outfalls have already been
        narrowed down from the original plausible set. Runs a djiikstra-based
        algorithm to identify the shortest path from each node to its nearest
        outfall (weighted by the 'weight' edge value). The
        returned graph is one that only contains the edges that feature  on the
        shortest paths.
        - `withtopo`: The alternative method that assumes no narrowing of plausible
        outfalls has been performed. This method runs a Tarjan's algorithm to
        identify the spanning forest starting from a `waste` node that all
        plausible outfalls are connected to (whether via a river or directly).

        In both methods, street nodes that have no plausible route to any outfall
        are removed.

        Args:
            G (nx.Graph): A graph
            outfall_derivation (parameters.OutfallDerivation): An OutfallDerivation
                parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()

        visited: set = set()

        # Increase recursion limit to allow to iterate over the entire graph
        # Seems to be the quickest way to identify which nodes have a path to
        # the outfall
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(original_limit, len(G.nodes)))

        # Identify outfalls
        if outfall_derivation.method == "withtopo":
            visited = set(nx.ancestors(G, "waste")) | {"waste"}

            # Remove nodes not reachable from waste
            G.remove_nodes_from(set(G.nodes) - visited)

            # Run shorted path
            G = shortest_path_utils.tarjans_pq(G, "waste")

            G = filter_streets(G)
        else:
            outfalls = [
                u for u, v, d in G.edges(data=True) if d["edge_type"] == "outfall"
            ]
            visited = set(outfalls)
            for outfall in outfalls:
                visited = visited | set(nx.ancestors(G, outfall))

            G.remove_nodes_from(set(G.nodes) - visited)
            G = filter_streets(G)

            # Check for negative cycles
            if nx.negative_edge_cycle(G, weight="weight"):
                logger.warning("Graph contains negative cycle")

            G = shortest_path_utils.dijkstra_pq(G, outfalls)

        # Reset recursion limit
        sys.setrecursionlimit(original_limit)

        # Log total weight
        total_weight = sum([d["weight"] for u, v, d in G.edges(data=True)])
        logger.info(f"Total graph weight {total_weight}.")

        return G

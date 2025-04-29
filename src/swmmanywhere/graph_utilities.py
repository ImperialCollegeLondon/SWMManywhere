"""Graph utilities module for SWMManywhere.

A module to contain graphfcns, the graphfcn registry object, and other graph
utilities (such as save/load functions).
"""

from __future__ import annotations

import json
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx
import shapely

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters
from swmmanywhere.filepaths import FilePaths
from swmmanywhere.logging import logger, verbose


def load_graph(fid: Path) -> nx.Graph:
    """Load a graph from a file saved with save_graph.

    Args:
        fid (Path): The path to the file

    Returns:
        G (nx.Graph): A graph
    """
    json_data = json.loads(fid.read_text())

    G = nx.node_link_graph(json_data, directed=True)
    for u, v, data in G.edges(data=True):
        if "geometry" in data:
            geometry_coords = data["geometry"]
            line_string = shapely.LineString(shapely.wkt.loads(geometry_coords))
            data["geometry"] = line_string
    return G


def _serialize_line_string(obj):
    if isinstance(obj, shapely.LineString):
        return obj.wkt
    return obj


def save_graph(G: nx.Graph, fid: Path) -> None:
    """Save a graph to a file.

    Args:
        G (nx.Graph): A graph
        fid (Path): The path to the file
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        json_data = nx.node_link_data(G)

    with fid.open("w") as file:
        json.dump(json_data, file, default=_serialize_line_string)


class BaseGraphFunction(ABC):
    """Base class for graph functions.

    On a SWMManywhere project the intention is to iterate over a number of
    graph functions. Each graph function may require certain attributes to
    be present in the graph. Each graph function may add attributes to the
    graph. This class provides a framework for graph functions to check
    their requirements and additions a-priori when the list is provided.
    """

    required_edge_attributes: List[str] = []
    adds_edge_attributes: List[str] = []
    required_node_attributes: List[str] = []
    adds_node_attributes: List[str] = []

    def __init_subclass__(
        cls,
        required_edge_attributes: Optional[List[str]] = None,
        adds_edge_attributes: Optional[List[str]] = None,
        required_node_attributes: Optional[List[str]] = None,
        adds_node_attributes: Optional[List[str]] = None,
    ):
        """Set the required and added attributes for the subclass."""
        cls.required_edge_attributes = required_edge_attributes or []
        cls.adds_edge_attributes = adds_edge_attributes or []
        cls.required_node_attributes = required_node_attributes or []
        cls.adds_node_attributes = adds_node_attributes or []

    @abstractmethod
    def __call__(self, G: nx.Graph, *args, **kwargs) -> nx.Graph:
        """Run the graph function."""
        return G

    def validate_requirements(self, edge_attributes: set, node_attributes: set) -> None:
        """Validate that the graph has the required attributes."""
        for attribute in self.required_edge_attributes:
            assert attribute in edge_attributes, f"{attribute} not in edge attributes"

        for attribute in self.required_node_attributes:
            assert attribute in node_attributes, f"{attribute} not in node attributes"

    def add_graphfcn(
        self, edge_attributes: set, node_attributes: set
    ) -> tuple[set, set]:
        """Add the attributes that the graph function adds."""
        self.validate_requirements(edge_attributes, node_attributes)
        edge_attributes = edge_attributes.union(self.adds_edge_attributes)
        node_attributes = node_attributes.union(self.adds_node_attributes)
        return edge_attributes, node_attributes


class GraphFunctionRegistry(dict):
    """Registry object."""

    def register(self, cls):
        """Register a graph function."""
        if cls.__name__ in self:
            raise ValueError(f"{cls.__name__} already in the graph functions registry!")

        self[cls.__name__] = cls()
        return cls

    def __getattr__(self, name):
        """Get a graph function from the graphfcn dict."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{name} NOT in the graph functions registry!")


graphfcns = GraphFunctionRegistry()


def register_graphfcn(cls) -> Callable:
    """Register a graph function.

    Args:
        cls (Callable): A class that inherits from BaseGraphFunction

    Returns:
        cls (Callable): The same class
    """
    graphfcns.register(cls)
    return cls


def filter_streets(G):
    """Filter streets.

    This function removes non streets from a graph.

    Args:
        G (nx.Graph): A graph

    Returns:
       (nx.Graph): A graph of only street edges
    """
    G = G.copy()
    # Remove non-street edges/nodes and unconnected nodes
    nodes_to_remove = []
    for u, v, d in G.edges(data=True):
        if d["edge_type"] != "street":
            if d["edge_type"] == "outfall":
                nodes_to_remove.append(v)
            else:
                nodes_to_remove.extend((u, v))
    G.remove_nodes_from(nodes_to_remove)
    return G


def validate_graphfcn_list(
    graphfcn_list: list[str], starting_graph: nx.Graph | None = None
) -> None:
    """Validate that the graph functions are registered.

    Tests that the graph functions are registered.

    Tests that the graph functions have the required attributes in the graph
    and updates the attributes that are added to the graph.
    `required_edge_attributes` and `required_node_attributes` currently only
    specify that one element in the graph must have the attribute (e.g., if a
    graphfcn has `required_node_attributes=['id']`, and only one node in the
    graph has the `id` attribute, then it will be valid).

    Args:
        graphfcn_list (list[str]): A list of graph functions
        starting_graph (nx.Graph, optional): A graph to check the starting
            attributes of. Defaults to None.

    Raises:
        ValueError: If a graph function is not registered
        ValueError: If a graph function requires an attribute that is not in
            the graph
    """
    # Check that the graph functions are registered
    not_exists = [g for g in graphfcn_list if g not in graphfcns]
    if not_exists:
        raise ValueError(f"Graphfcns are not registered:\n{', '.join(not_exists)}")

    if starting_graph is None:
        return

    # Get starting graph attributes
    edge_attributes: set = set()
    for u, v, data in starting_graph.edges(data=True):
        edge_attributes = edge_attributes.union(data.keys())

    node_attributes: set = set()
    for node, data in starting_graph.nodes(data=True):
        node_attributes = node_attributes.union(data.keys())

    # Iterate over graphfcn_list and check that the required attributes are
    # present in the graph, updating the add attributes
    for graphfcn in graphfcn_list:
        if node_attributes.intersection(
            graphfcns[graphfcn].required_node_attributes
        ) != set(graphfcns[graphfcn].required_node_attributes):
            raise ValueError(
                f"""Graphfcn {graphfcn} requires node attributes 
                {graphfcns[graphfcn].required_node_attributes}"""
            )
        if edge_attributes.intersection(
            graphfcns[graphfcn].required_edge_attributes
        ) != set(graphfcns[graphfcn].required_edge_attributes):
            raise ValueError(
                f"""Graphfcn {graphfcn} requires edge attributes 
                {graphfcns[graphfcn].required_edge_attributes}"""
            )

        edge_attributes = edge_attributes.union(
            graphfcns[graphfcn].adds_edge_attributes
        )
        node_attributes = node_attributes.union(
            graphfcns[graphfcn].adds_node_attributes
        )


with tempfile.TemporaryDirectory() as temp_dir:
    temp_addresses = FilePaths(
        base_dir=Path(temp_dir), bbox_bounds=(0, 1, 0, 1), project_name="temp"
    )


def iterate_graphfcns(
    G: nx.Graph,
    graphfcn_list: list[str],
    params: dict = parameters.get_full_parameters(),
    addresses: FilePaths = temp_addresses,
) -> nx.Graph:
    """Iterate a list of graph functions over a graph.

    Args:
        G (nx.Graph): The graph to iterate over.
        graphfcn_list (list[str]): A list of graph functions to iterate.
        params (dict): A dictionary of parameters to pass to the graph
            functions.
        addresses (FilePaths): A FilePaths parameter object

    Returns:
        nx.Graph: The graph after the graph functions have been applied.
    """
    validate_graphfcn_list(graphfcn_list)

    for function in graphfcn_list:
        G = graphfcns[function](G, addresses=addresses, **params)
        if len(filter_streets(G).edges) == 0:
            logger.warning(
                f"""graphfcn: {function} removed all edges, 
                           returning graph."""
            )
            return G
        else:
            logger.info(f"graphfcn: {function} completed.")

        if verbose():
            save_graph(G, addresses.model_paths.model / f"{function}_graph.json")
            go.graph_to_geojson(
                graphfcns.fix_geometries(G),
                addresses.model_paths.model / f"{function}_nodes.geojson",
                addresses.model_paths.model / f"{function}_edges.geojson",
                G.graph["crs"],
            )
    return G

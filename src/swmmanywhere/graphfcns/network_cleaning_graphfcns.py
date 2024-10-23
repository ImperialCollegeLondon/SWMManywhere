"""Module for graphfcns that change the plausible pipe location network."""

from __future__ import annotations

import re
from typing import Any, Hashable, cast

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import shapely

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters
from swmmanywhere.filepaths import FilePaths
from swmmanywhere.graph_utilities import BaseGraphFunction, register_graphfcn


def get_osmid_id(data: dict) -> Hashable:
    """Get the ID of an edge."""
    id_ = data.get("osmid", data.get("id"))
    if isinstance(id_, list):
        id_ = id_[0]
    return id_


@register_graphfcn
class assign_id(BaseGraphFunction, adds_edge_attributes=["id"]):
    """assign_id class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """Assign an ID to each edge.

        This function takes a graph and assigns an ID to each edge. The ID is
        assigned to the 'id' attribute of each edge. Needed because some edges
        have 'osmid', some have a list of 'osmid', others have 'id'.

        Args:
            G (nx.Graph): A graph
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): The same graph with an ID assigned to each edge
        """
        edge_ids: set[str] = set()
        edges_to_remove = []
        for u, v, key, data in G.edges(data=True, keys=True):
            data["id"] = f"{u}-{v}"
            if data["id"] in edge_ids:
                edges_to_remove.append((u, v, key))
            edge_ids.add(data["id"])
        for u, v, key in edges_to_remove:
            G.remove_edge(u, v, key)
        return G


@register_graphfcn
class remove_parallel_edges(BaseGraphFunction):
    """remove_parallel_edges class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """Remove parallel edges from a street network.

        Retain the edge with the smallest weight (i.e., length).

        Args:
            G (nx.MultiDiGraph): A graph.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.DiGraph): The graph with parallel edges removed.

        Author:
            Taher Chegini
        """
        # Set the attribute (weight) used to determine which parallel edge to
        # retain. Could make this a parameter in parameters.py if needed.
        weight = "length"
        graph = ox.convert.to_digraph(G)
        _, _, attr_list = next(iter(graph.edges(data=True)))  # type: ignore
        attr_list = cast("dict[str, Any]", attr_list)
        if weight not in attr_list:
            raise ValueError(f"{weight} not in edge attributes.")
        attr = nx.get_node_attributes(graph, weight)
        parallels = (e for e in attr if e[::-1] in attr)
        graph.remove_edges_from(
            {e if attr[e] > attr[e[::-1]] else e[::-1] for e in parallels}
        )
        return graph


@register_graphfcn
class remove_non_pipe_allowable_links(BaseGraphFunction):
    """remove_non_pipe_allowable_links class."""

    def __call__(
        self, G: nx.Graph, topology_derivation: parameters.TopologyDerivation, **kwargs
    ) -> nx.Graph:
        """Remove non-pipe allowable links.

        This function removes links that are not allowable for pipes. The non-
        allowable links are specified in the `omit_edges` attribute of the
        topology_derivation parameter. There two cases handled:

        1. The `highway` property of the edge. In `osmnx`, `highway` is a category
            that contains the road type, e.g., motorway, trunk, primary. If the
            edge contains a value in the `highway` property that is in `omit_edges`,
            the edge is removed.

        2. Any other properties of the edge that are in `omit_edges`. If the
            property is not null in the edge data, the edge is removed. e.g.,
            if `bridge` is in `omit_edges` and the `bridge` entry of the edge
            is NULL, then the edge is retained, if it is something like 'yes',
            or 'viaduct' then the edge is removed.

        Args:
            G (nx.Graph): A graph
            topology_derivation (parameters.TopologyDerivation): A TopologyDerivation
                parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        edges_to_remove = set()
        for u, v, keys, data in G.edges(data=True, keys=True):
            for omit in topology_derivation.omit_edges:
                if data.get("highway", None) == omit:
                    # Check whether the 'highway' property is 'omit'
                    edges_to_remove.add((u, v, keys))
                elif data.get(omit, None):
                    # Check whether the 'omit' property of edge is not None
                    edges_to_remove.add((u, v, keys))
        for edges in edges_to_remove:
            G.remove_edge(*edges)
        return G


def sum_over_delimiter(s: int | str | float) -> float:
    """Sum over a delimiter.

    This function takes a value, if it is not a string it is casted as a
    float, otherwise it sums over the numbers in the string. The
    numbers are separated by a delimiter. The delimiter is any non-numeric
    character. If the input is not a string, the function returns the input.

    Args:
        s (int | str | float): The input.

    Returns:
        float: The sum of the numbers in the string
    """
    if not isinstance(s, str):
        return float(s)
    return float(sum([int(num) for num in re.split(r"\D+", s) if num]))


@register_graphfcn
class calculate_streetcover(
    BaseGraphFunction, required_edge_attributes=["lanes", "geometry"]
):
    """calculate_streetcover class."""

    # i.e., in osmnx format, i.e., empty for single lane, an int for a
    # number of lanes or a list if the edge has multiple carriageways

    def __call__(
        self,
        G: nx.Graph,
        subcatchment_derivation: parameters.SubcatchmentDerivation,
        addresses: FilePaths,
        **kwargs,
    ) -> nx.Graph:
        """Format the lanes attribute of each edge and calculates width.

        Only the `drive` network is assumed to contribute to impervious area and
        so others `network_types` have lanes set to 0. If no `network_type` is
        present, the edge is assumed to be of type `drive`.

        Args:
            G (nx.Graph): A graph
            subcatchment_derivation (parameters.SubcatchmentDerivation): A
                SubcatchmentDerivation parameter object
            addresses (FilePaths): A FilePaths parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        lines = []
        for u, v, data in G.edges(data=True):
            if data.get("network_type", "drive") == "drive":
                lanes = data.get("lanes", 1)
            else:
                lanes = 0
            if isinstance(lanes, list):
                lanes = sum([sum_over_delimiter(x) for x in lanes])
            else:
                lanes = sum_over_delimiter(lanes)
            lines.append(
                {
                    "geometry": data["geometry"].buffer(
                        lanes * subcatchment_derivation.lane_width,
                        cap_style=2,
                        join_style=2,
                    ),
                    "u": u,
                    "v": v,
                }
            )
        lines_df = pd.DataFrame(lines)
        lines_gdf = gpd.GeoDataFrame(
            lines_df, geometry=lines_df.geometry, crs=G.graph["crs"]
        )
        if addresses.model_paths.streetcover.suffix in (".geoparquet", ".parquet"):
            lines_gdf.to_parquet(addresses.model_paths.streetcover)
        else:
            lines_gdf.to_file(addresses.model_paths.streetcover, driver="GeoJSON")

        return G


@register_graphfcn
class double_directed(BaseGraphFunction, required_edge_attributes=["id"]):
    """double_directed class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """Create a 'double directed graph'.

        This function duplicates a graph and adds reverse edges to the new graph.
        These new edges share the same data as the 'forward' edges but have a new
        'id'. An undirected graph is not suitable because it removes data of one of
        the edges if there are edges in both directions between two nodes
        (necessary to preserve, e.g., consistent 'width'). If 'edge_type' is
        present, then the function will only be performed on 'street' types.

        Args:
            G (nx.Graph): A graph
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        # Convert to directed
        G_new = nx.MultiDiGraph(G.copy())

        # MultiDiGraph adds edges in both directions, but rivers (and geometries)
        # are only in one direction. So we remove the reverse edges and add them
        # back in with the correct geometry.
        # This assumes that 'id' is of format 'start-end' (see assign_id)
        arcs_to_remove = [
            (u, v) for u, v, d in G_new.edges(data=True) if f"{u}-{v}" != d.get("id")
        ]

        # Remove the reverse edges
        for u, v in arcs_to_remove:
            G_new.remove_edge(u, v)

        # Add in reversed edges for streets only and with geometry
        for u, v, data in G.edges(data=True):
            include = data.get("edge_type", True)
            if isinstance(include, str):
                include = include == "street"
            if ((v, u) not in G_new.edges) & include:
                reverse_data = data.copy()
                reverse_data["id"] = f"{data['id']}.reversed"
                new_geometry = shapely.LineString(
                    list(reversed(data["geometry"].coords))
                )
                reverse_data["geometry"] = new_geometry
                G_new.add_edge(v, u, **reverse_data)
        return G_new


@register_graphfcn
class to_undirected(BaseGraphFunction):
    """to_undirected class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """Convert the graph to an undirected graph.

        Args:
            G (nx.Graph): A graph
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): An undirected graph
        """
        # Don't use osmnx.to_undirected! It enables multigraph if the geometries
        # are different, but we have already saved the street cover so don't
        # want this!
        return G.to_undirected()


@register_graphfcn
class split_long_edges(BaseGraphFunction, required_edge_attributes=["id", "geometry"]):
    """split_long_edges class."""

    def __call__(
        self,
        G: nx.Graph,
        subcatchment_derivation: parameters.SubcatchmentDerivation,
        **kwargs,
    ) -> nx.Graph:
        """Split long edges into shorter edges.

        This function splits long edges into shorter edges. The edges are split
        into segments of length 'max_street_length'. The 'geometry' of the
        original edge must be a LineString. Intended to follow up with call of
        `merge_nodes`.

        Args:
            G (nx.Graph): A graph
            subcatchment_derivation (parameters.SubcatchmentDerivation): A
                SubcatchmentDerivation parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            graph (nx.Graph): A graph
        """
        max_length = subcatchment_derivation.max_street_length

        # Split edges
        new_linestrings = shapely.segmentize(
            [d["geometry"] for u, v, d in G.edges(data=True)], max_length
        )
        new_nodes = shapely.get_coordinates(new_linestrings)

        new_edges = {}
        for new_linestring, (u, v, d) in zip(new_linestrings, G.edges(data=True)):
            # Create an arc for each segment
            for start, end in zip(
                new_linestring.coords[:-1], new_linestring.coords[1:]
            ):
                geom = shapely.LineString([start, end])
                new_edges[(start, end, 0)] = {**d, "length": geom.length}

        # Create new graph
        new_graph = nx.MultiGraph()
        new_graph.graph = G.graph.copy()
        new_graph.add_edges_from(new_edges)
        nx.set_edge_attributes(new_graph, new_edges)
        nx.set_node_attributes(
            new_graph, {tuple(node): {"x": node[0], "y": node[1]} for node in new_nodes}
        )
        return nx.relabel_nodes(
            new_graph, {node: ix for ix, node in enumerate(new_graph.nodes)}
        )


@register_graphfcn
class merge_street_nodes(BaseGraphFunction):
    """merge_nodes class."""

    def __call__(
        self,
        G: nx.Graph,
        subcatchment_derivation: parameters.SubcatchmentDerivation,
        **kwargs,
    ) -> nx.Graph:
        """Merge nodes that are close together.

        Merges `street` nodes that are within a certain distance of each
        other. The distance is specified in the `node_merge_distance` attribute
        of the `subcatchment_derivation` parameter. The merged nodes are given
        the same coordinates, and the graph is relabeled with nx.relabel_nodes.
        Suggest to follow with call of `assign_id` to remove duplicate edges.

        Args:
            G (nx.Graph): A graph
            subcatchment_derivation (parameters.SubcatchmentDerivation): A
                SubcatchmentDerivation parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()

        # Separate out streets
        street_edges = [
            (u, v, k)
            for u, v, k, d in G.edges(data=True, keys=True)
            if d.get("edge_type", "street") == "street"
        ]
        streets = G.edge_subgraph(street_edges).copy()

        # Identify nodes that are within threshold of each other
        mapping = go.merge_points(
            [(d["x"], d["y"]) for u, d in streets.nodes(data=True)],
            subcatchment_derivation.node_merge_distance,
        )

        # Get indexes of node names
        node_indices = {ix: node for ix, node in enumerate(streets.nodes)}

        # Create a mapping of old node names to new node names
        node_names = {}
        for ix, node in enumerate(streets.nodes):
            if ix in mapping:
                # If the node is in the mapping, then it is mapped and
                # given the new coordinate (all nodes in a mapping family must
                # be given the same coordinate because of how relabel_nodes
                # works)
                node_names[node] = node_indices[mapping[ix]["maps_to"]]
                G.nodes[node]["x"] = mapping[ix]["coordinate"][0]
                G.nodes[node]["y"] = mapping[ix]["coordinate"][1]
            else:
                node_names[node] = node

        G = nx.relabel_nodes(G, node_names)

        # Relabelling will create selfloops within a mapping family, which
        # are removed
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)

        return G


@register_graphfcn
class fix_geometries(
    BaseGraphFunction,
    adds_edge_attributes=["geometry"],
    required_node_attributes=["x", "y"],
):
    """fix_geometries class."""

    def __call__(self, G: nx.Graph, **kwargs) -> nx.Graph:
        """Fix the geometries of the edges.

        This function fixes the geometries of the edges. The geometries are
        recalculated from the node coordinates.

        Args:
            G (nx.Graph): A graph
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        for u, v, data in G.edges(data=True):
            geom = data.get("geometry", None)

            start_point_node = (G.nodes[u]["x"], G.nodes[u]["y"])
            end_point_node = (G.nodes[v]["x"], G.nodes[v]["y"])
            if not geom:
                start_point_edge = (None, None)
                end_point_edge = (None, None)
            else:
                start_point_edge = data["geometry"].coords[0]
                end_point_edge = data["geometry"].coords[-1]

            if (start_point_edge == end_point_node) & (
                end_point_edge == start_point_node
            ):
                data["geometry"] = data["geometry"].reverse()
            elif (start_point_edge != start_point_node) | (
                end_point_edge != end_point_node
            ):
                data["geometry"] = shapely.LineString(
                    [start_point_node, end_point_node]
                )
        return G

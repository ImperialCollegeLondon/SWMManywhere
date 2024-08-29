"""Module for graphfcns that identify outfalls."""

from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd
import shapely

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters
from swmmanywhere.graph_utilities import BaseGraphFunction, register_graphfcn
from swmmanywhere.logging import logger


def _get_points(
    G: nx.Graph,
) -> tuple[Dict[str, shapely.Point], Dict[str, shapely.Point]]:
    """Get the river and street points from the graph.

    A river point are the start and end nodes of an edge with `edge_type = river`.
    A street point are the start and end nodes of an edge with
    `edge_type = street`.

    Args:
        G (nx.Graph): A graph

    Returns:
        river_points (dict): A dictionary of river points
        street_points (dict): A dictionary of street points
    """
    # Get edge types, convert to nx.Graph to remove keys
    etypes = nx.get_edge_attributes(nx.Graph(G), "edge_type")

    # Get river and street points as a dict
    n_types = (
        pd.DataFrame(etypes.items(), columns=["key", "type"])
        .explode("key")
        .reset_index(drop=True)
        .groupby("type")["key"]
        .apply(list)
        .to_dict()
    )
    river_points = {
        n: shapely.Point(G.nodes[n]["x"], G.nodes[n]["y"])
        for n in n_types.get("river", {})
    }
    street_points = {
        n: shapely.Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in n_types["street"]
    }

    return river_points, street_points


def _pair_rivers(
    G: nx.Graph,
    river_points: Dict[str, shapely.Point],
    street_points: Dict[str, shapely.Point],
    river_buffer_distance: float,
    outfall_length: float,
) -> nx.Graph:
    """Pair river and street nodes.

    Pair river and street nodes within a certain distance of each other. If
    there are no plausible outfalls for an entire subgraph, then a dummy river
    node is created and the lowest elevation node is used as the outfall.

    Args:
        G (nx.Graph): A graph
        river_points (dict): A dictionary of river points
        street_points (dict): A dictionary of street points
        river_buffer_distance (float): The distance within which a river and
            street node can be paired
        outfall_length (float): The length of the outfall

    Returns:
        G (nx.Graph): A graph
    """
    matched_outfalls = {}

    # Insert a dummy river node and use lowest elevation node as outfall
    # for each subgraph with no matched outfalls
    subgraphs = []
    for sg in nx.weakly_connected_components(G):
        sg = G.subgraph(sg).copy()
        subgraphs.append(sg)

        # Ignore the rivers, they are drained later
        if all([d["edge_type"] == "river" for _, _, d in sg.edges(data=True)]):
            continue

        # Pair up the river and street nodes for each subgraph
        street_points_ = {k: v for k, v in street_points.items() if k in sg.nodes}

        subgraph_outfalls = go.nearest_node_buffer(
            street_points_, river_points, river_buffer_distance
        )

        # Check if there are any matched outfalls
        if subgraph_outfalls:
            # Update all matched outfalls
            matched_outfalls.update(subgraph_outfalls)
            continue

        # In cases of e.g., an area with no rivers to discharge into or too
        # small a buffer

        # Identify the lowest elevation node
        lowest_elevation_node = min(
            sg.nodes, key=lambda x: sg.nodes[x]["surface_elevation"]
        )

        # Create a dummy river to discharge into
        name = f"{lowest_elevation_node}-dummy_river"
        dummy_river = {
            "id": name,
            "x": G.nodes[lowest_elevation_node]["x"] + 1,
            "y": G.nodes[lowest_elevation_node]["y"] + 1,
        }
        sg.add_node(name)
        nx.set_node_attributes(sg, {name: dummy_river})

        # Update function's dicts
        matched_outfalls[lowest_elevation_node] = name
        river_points[name] = shapely.Point(dummy_river["x"], dummy_river["y"])

        logger.warning(
            f"""No outfalls found for subgraph containing 
                        {lowest_elevation_node}, using this node as outfall."""
        )

    G = nx.compose_all(subgraphs)

    # Add edges between the paired river and street nodes
    for street_id, river_id in matched_outfalls.items():
        # TODO instead use a weight based on the distance between the two nodes
        G.add_edge(
            street_id,
            river_id,
            **{
                "length": outfall_length,
                "weight": outfall_length,
                "edge_type": "outfall",
                "geometry": shapely.LineString(
                    [street_points[street_id], river_points[river_id]]
                ),
                "id": f"{street_id}-{river_id}-outfall",
            },
        )

    return G


def _root_nodes(G: nx.Graph) -> nx.Graph:
    """Root nodes with a waste node.

    Connect all nodes that have nowhere to flow to to a waste node, i.e., the
    root of the entire graph.

    Args:
        G (nx.Graph): A graph

    Returns:
        G (nx.Graph): A graph
    """
    G_ = G.copy()

    # Add edges from the river nodes to a waste node
    G.add_node("waste")

    for node in G_.nodes:
        if G.out_degree(node) == 0:
            # Location of the waste node doesn't matter - so if there
            # are multiple river nodes with out_degree 0 - that's fine.
            G.nodes["waste"]["x"] = G.nodes[node]["x"] + 1
            G.nodes["waste"]["y"] = G.nodes[node]["y"] + 1
            G.add_edge(
                node,
                "waste",
                **{
                    "length": 0,
                    "weight": 0,
                    "edge_type": "waste-outfall",
                    "id": f"{node}-waste-outfall",
                },
            )
    return G


def _connect_mst_outfalls(paired_G: nx.Graph, raw_G: nx.Graph) -> nx.Graph:
    """Connect outfalls to a waste node.

    Run a minimum spanning tree (MST) on the paired graph to identify the
    'efficient' outfalls. These outfalls are inserted into the original graph.

    Args:
        paired_G (nx.Graph): A graph where streets and rivers are paired with
            outfalls.
        raw_G (nx.Graph): A graph where streets and rivers are separated

    Returns:
        (nx.Graph): A graph
    """
    # Find shortest path to identify only 'efficient' outfalls. The MST
    # makes sense here over shortest path as each node is only allowed to
    # be visited once - thus encouraging fewer outfalls. In shortest path
    # nodes near rivers will always just pick their nearest river node.
    T = nx.minimum_spanning_tree(paired_G.to_undirected(), weight="length")

    # Retain the shortest path outfalls in the original graph
    for u, v, d in T.edges(data=True):
        if (d["edge_type"] == "outfall") & (v != "waste") & (u != "waste"):
            if u not in raw_G.nodes():
                raw_G.add_node(u, **paired_G.nodes[u])
            elif v not in raw_G.nodes():
                raw_G.add_node(v, **paired_G.nodes[v])

            # Need to check both directions since T is undirected
            if (u, v) in paired_G.edges():
                raw_G.add_edge(u, v, **d)
            elif (v, u) in paired_G.edges():
                raw_G.add_edge(v, u, **d)
            else:
                raise ValueError(f"Edge {u}-{v} not found in paired_G")

    return raw_G


@register_graphfcn
class identify_outfalls(
    BaseGraphFunction,
    required_edge_attributes=["length", "edge_type"],
    required_node_attributes=["x", "y", "surface_elevation"],
):
    """identify_outfalls class."""

    def __call__(
        self, G: nx.Graph, outfall_derivation: parameters.OutfallDerivation, **kwargs
    ) -> nx.Graph:
        """Identify outfalls in a combined river-street graph.

        This function identifies outfalls in a combined river-street graph. An
        outfall is a node that is connected to a river and a street. Each street
        node is paired with the nearest river node provided that it is within
        a distance of outfall_derivation.river_buffer_distance - this provides a
        large set of plausible outfalls. If there are no plausible outfalls for an
        entire subgraph, then a dummy river node is created and the lowest
        elevation node is paired with it. Any street->river/outfall link is given
        a `weight` and `length` of outfall_derivation.outfall_length, this is to
        ensure some penalty on the total number of outfalls selected.

        Two methods are available for determining which plausible outfalls to
        retain:

        - `withtopo`: all plausible outfalls are retained, connected to a single
        'waste' node and assumed to be later identified as part of the
        `derive_topology` graphfcn.

        - `separate`: the retained outfalls are those that are selected as part
        of the minimum spanning tree (MST) of the combined street-river graph.
        This method can be temporamental because the MST is undirected, because
        rivers are inherently directed unusual outfall locations may be retained.

        Args:
            G (nx.Graph): A graph
            outfall_derivation (parameters.OutfallDerivation): An OutfallDerivation
                parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()

        river_points, street_points = _get_points(G)

        G_ = _pair_rivers(
            G,
            river_points,
            street_points,
            outfall_derivation.river_buffer_distance,
            outfall_derivation.outfall_length,
        )

        # Set the length of the river edges to 0 - from a design perspective
        # once water is in the river we don't care about the length - since it
        # costs nothing
        for _, _, d in G_.edges(data=True):
            if d["edge_type"] == "river":
                d["length"] = 0
                d["weight"] = 0

        # Add edges from the river nodes to a waste node
        G_ = _root_nodes(G_)

        if outfall_derivation.method == "withtopo":
            # The outfalls can be derived as part of the shortest path calculations
            return G_
        elif outfall_derivation.method == "separate":
            return _connect_mst_outfalls(G_, G)
        else:
            raise ValueError(f"Unknown method {outfall_derivation.method}")

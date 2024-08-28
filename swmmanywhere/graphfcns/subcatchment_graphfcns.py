"""Module for graphfcns that change subcatchments."""

from __future__ import annotations

import tempfile
from itertools import product
from pathlib import Path

import geopandas as gpd
import networkx as nx

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters
from swmmanywhere.filepaths import FilePaths
from swmmanywhere.graph_utilities import BaseGraphFunction, register_graphfcn
from swmmanywhere.logging import logger, verbose


@register_graphfcn
class clip_to_catchments(
    BaseGraphFunction,
    required_node_attributes=["x", "y"],
    required_edge_attributes=["length"],
    adds_node_attributes=["community", "basin"],
):
    """clip_to_catchments class."""

    def __call__(
        self,
        G: nx.Graph,
        addresses: FilePaths,
        subcatchment_derivation: parameters.SubcatchmentDerivation,
        **kwargs,
    ) -> nx.Graph:
        """Clip the graph to the subcatchments.

        Derive the subbasins with `subcatchment_derivation.subbasin_streamorder`.
        If no subbasins exist for that stream order, the value is iterated
        downwards and a warning it flagged.

        If `subcatchment_derivation.subbasin_clip_method` is 'subbasin', then
        links between subbasins are removed. If it is 'community', then links
        between communities in different subbasins may be removed based on the
        following method.

        Run Louvain community detection on the street network to create street
        node communities.

        Communities with less than `subcatchment_derivation.subbasin_membership`
        proportion of nodes in a subbasin have their links to all other nodes
        in that subbasin removed. Nodes not in any subbasin are assigned to a
        subbasin to cover all unassigned nodes.

        Community and basin ids are added to nodes mainly to help with debugging.

        Args:
            G (nx.Graph): A graph
            addresses (FilePaths): A FilePaths parameter object
            subcatchment_derivation (parameters.SubcatchmentDerivation): A
                SubcatchmentDerivation parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()

        # Derive subbasins
        subbasins = go.derive_subbasins_streamorder(
            addresses.bbox_paths.elevation,
            subcatchment_derivation.subbasin_streamorder,
            x=list(nx.get_node_attributes(G, "x").values()),
            y=list(nx.get_node_attributes(G, "y").values()),
        )

        if verbose():
            subbasins.to_file(
                str(addresses.model_paths.nodes).replace("nodes", "subbasins"),
                driver="GeoJSON",
            )

        # Extract street network
        street = G.copy()
        street.remove_edges_from(
            [
                (u, v)
                for u, v, d in street.edges(data=True)
                if d.get("edge_type", "street") != "street"
            ]
        )

        # Create gdf of street points
        street_points = gpd.GeoDataFrame(
            G.nodes,
            columns=["id"],
            geometry=gpd.points_from_xy(
                [G.nodes[u]["x"] for u in G.nodes], [G.nodes[u]["y"] for u in G.nodes]
            ),
            crs=G.graph["crs"],
        ).set_index("id")

        # Classify street points by subbasin
        street_points = gpd.sjoin(
            street_points,
            subbasins.set_index("basin"),
            how="left",
        ).rename(columns={"index_right": "basin"})

        if subcatchment_derivation.subbasin_clip_method == "subbasin":
            edges_to_remove = [
                (u, v)
                for u, v in G.edges()
                if street_points.loc[u, "basin"] != street_points.loc[v, "basin"]
            ]
            G.remove_edges_from(edges_to_remove)
            return G

        # Derive road network clusters
        louv_membership = nx.community.louvain_communities(
            street, weight="length", seed=1
        )

        street_points["community"] = 0
        # Assign louvain membership to street points
        for ix, community in enumerate(louv_membership):
            street_points.loc[list(community), "community"] = ix

        # Introduce a non catchment basin for nan
        street_points["basin"] = street_points["basin"].fillna(-1)
        # TODO possibly it makes sense to just remove these nodes, or at least
        # any communities that are all nan

        nx.set_node_attributes(G, street_points["community"].to_dict(), "community")
        nx.set_node_attributes(G, street_points["basin"].to_dict(), "basin")

        # Calculate most percentage of each subbasin in each community
        community_basin = (
            street_points.groupby("community").basin.value_counts().reset_index()
        )
        community_size = street_points.community.value_counts().reset_index()
        community_basin = community_basin.merge(
            community_size, on="community", how="left", suffixes=("_basin", "_size")
        )

        # Normalize
        community_basin["percentage"] = (
            community_basin["count_basin"] / community_basin["count_size"]
        )

        # Identify community-basin combinations where the percentage is less than
        # the threshold
        community_omit = community_basin.loc[
            community_basin["percentage"] <= subcatchment_derivation.subbasin_membership
        ]

        community_basin = community_basin.set_index("basin")

        # Cut links between communities in community_omit and commuities in those
        # basins
        arcs_to_remove = []
        street_points = street_points.reset_index().set_index("basin")
        for idx, row in community_omit.iterrows():
            community_nodes = louv_membership[int(row["community"])]
            basin_nodes = street_points.loc[[row["basin"]], "id"]
            basin_nodes = set(basin_nodes).difference(community_nodes)

            # Include both directions because operation should work on
            # undirected or directed graph
            arcs_to_remove.extend(
                [(u, v, 0) for u, v in product(community_nodes, basin_nodes)]
                + [(v, u, 0) for u, v in product(community_nodes, basin_nodes)]
            )
        G.remove_edges_from(set(G.edges).intersection(arcs_to_remove))
        if G.is_directed():
            subgraphs = len(list(nx.weakly_connected_components(G)))
        else:
            subgraphs = len(list(nx.connected_components(G)))
        logger.info(f"clip_to_catchments has created {subgraphs} subgraphs.")
        return G


@register_graphfcn
class calculate_contributing_area(
    BaseGraphFunction,
    required_edge_attributes=["id", "geometry"],
    adds_edge_attributes=["contributing_area"],
    adds_node_attributes=["contributing_area"],
):
    """calculate_contributing_area class."""

    def __call__(
        self,
        G: nx.Graph,
        subcatchment_derivation: parameters.SubcatchmentDerivation,
        addresses: FilePaths,
        **kwargs,
    ) -> nx.Graph:
        """Calculate the contributing area for each edge.

        This function calculates the contributing area for each edge. The
        contributing area is the area of the subcatchment that drains to the
        edge. The contributing area is calculated from the elevation data.
        Runoff coefficient (RC) for each contributing area is also calculated,
        the RC is calculated using `addresses.bbox_paths.buildings` and
        `addresses.model_paths.streetcover`.

        Also writes the file 'subcatchments.geojson' to
        addresses.model_paths.subcatchments.

        Args:
            G (nx.Graph): A graph
            subcatchment_derivation (parameters.SubcatchmentDerivation): A
                SubcatchmentDerivation parameter object
            addresses (FilePaths): An FilePaths parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()

        # Carve
        # TODO I guess we don't need to keep this 'carved' file..
        # maybe could add verbose/debug option to keep it
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_fid = Path(temp_dir) / "carved.tif"
            go.burn_shape_in_raster(
                [d["geometry"] for u, v, d in G.edges(data=True)],
                subcatchment_derivation.carve_depth,
                addresses.bbox_paths.elevation,
                temp_fid,
            )

            # Derive
            subs_gdf = go.derive_subcatchments(G, temp_fid)
            if verbose():
                subs_gdf.to_file(addresses.model_paths.subcatchments, driver="GeoJSON")

        # Calculate runoff coefficient (RC)
        if addresses.bbox_paths.building.suffix in (".geoparquet", ".parquet"):
            buildings = gpd.read_parquet(addresses.bbox_paths.building)
        else:
            buildings = gpd.read_file(addresses.bbox_paths.building)
        if addresses.model_paths.streetcover.suffix in (".geoparquet", ".parquet"):
            streetcover = gpd.read_parquet(addresses.model_paths.streetcover)
        else:
            streetcover = gpd.read_file(addresses.model_paths.streetcover)

        subs_rc = go.derive_rc(subs_gdf, buildings, streetcover)

        # Write subs
        # TODO - could just attach subs to nodes where each node has a list of subs
        if addresses.model_paths.subcatchments.suffix in (".geoparquet", ".parquet"):
            subs_rc.to_parquet(addresses.model_paths.subcatchments)
        else:
            subs_rc.to_file(addresses.model_paths.subcatchments, driver="GeoJSON")

        # Assign contributing area
        imperv_lookup = subs_rc.set_index("id").impervious_area.to_dict()

        # Set node attributes
        nx.set_node_attributes(G, 0.0, "contributing_area")
        nx.set_node_attributes(G, imperv_lookup, "contributing_area")

        # Prepare edge attributes
        edge_attributes = {
            edge: G.nodes[edge[0]]["contributing_area"] for edge in G.edges
        }

        # Set edge attributes
        nx.set_edge_attributes(G, edge_attributes, "contributing_area")
        return G

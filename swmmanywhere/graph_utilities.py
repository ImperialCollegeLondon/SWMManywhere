"""Graph utilities module for SWMManywhere.

A module to contain graphfcns, the graphfcn registry object, and other graph
utilities (such as save/load functions).
"""
from __future__ import annotations

import json
import sys
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from tqdm.auto import tqdm

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters, shortest_path_utils
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

    G = nx.node_link_graph(json_data,directed=True)
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            geometry_coords = data['geometry']
            line_string = shapely.LineString(shapely.wkt.loads(geometry_coords))
            data['geometry'] = line_string
    return G
def _serialize_line_string(obj):
    if isinstance(obj, shapely.LineString):
        return obj.wkt
    return obj
def save_graph(G: nx.Graph, 
               fid: Path) -> None:
    """Save a graph to a file.

    Args:
        G (nx.Graph): A graph
        fid (Path): The path to the file
    """
    json_data = nx.node_link_data(G)
    
    with fid.open('w') as file:
        json.dump(json_data, 
                  file,
                  default = _serialize_line_string)


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
    def __init_subclass__(cls, 
                          required_edge_attributes: Optional[List[str]] = None,
                          adds_edge_attributes: Optional[List[str]] = None,
                          required_node_attributes : Optional[List[str]] = None,
                          adds_node_attributes : Optional[List[str]] = None
                          ):
        """Set the required and added attributes for the subclass."""
        cls.required_edge_attributes = required_edge_attributes or []
        cls.adds_edge_attributes = adds_edge_attributes or []
        cls.required_node_attributes = required_node_attributes or []
        cls.adds_node_attributes = adds_node_attributes or []

    @abstractmethod
    def __call__(self, 
                 G: nx.Graph,
                 *args,
                 **kwargs) -> nx.Graph:
        """Run the graph function."""
        return G
    
    def validate_requirements(self,
                              edge_attributes: set,
                              node_attributes: set) -> None:
        """Validate that the graph has the required attributes."""
        for attribute in self.required_edge_attributes:
            assert attribute in edge_attributes, f"{attribute} not in edge attributes"
            
        for attribute in self.required_node_attributes:
            assert attribute in node_attributes, f"{attribute} not in node attributes"
            

    def add_graphfcn(self,
                    edge_attributes: set,
                    node_attributes: set) -> tuple[set, set]:
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

def get_osmid_id(data: dict) -> Hashable:
    """Get the ID of an edge."""
    id_ = data.get('osmid', data.get('id'))
    if isinstance(id_, list):
        id_ = id_[0]
    return id_

def iterate_graphfcns(G: nx.Graph, 
                      graphfcn_list: list[str], 
                      params: dict,
                      addresses: FilePaths) -> nx.Graph:
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
    not_exists = [g for g in graphfcn_list if g not in graphfcns]
    if not_exists:
        raise ValueError(f"Graphfcns are not registered:\n{', '.join(not_exists)}")
    for function in graphfcn_list:
        G = graphfcns[function](G, addresses = addresses, **params)
        if len(_filter_streets(G).edges) == 0:
            logger.warning(f"""graphfcn: {function} removed all edges, 
                           returning graph.""")
            return G
        else:
            logger.info(f"graphfcn: {function} completed.")
        
        if verbose():
            save_graph(G, addresses.model / f"{function}_graph.json")
            go.graph_to_geojson(graphfcns.fix_geometries(G),
                                addresses.model / f"{function}_nodes.geojson",
                                addresses.model / f"{function}_edges.geojson",
                                G.graph['crs']
                                )
    return G

@register_graphfcn
class assign_id(BaseGraphFunction, 
                adds_edge_attributes = ['id']
                ):
    """assign_id class."""

    def __call__(self,
                 G: nx.Graph, 
                 **kwargs) -> nx.Graph:
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
        for u, v, key, data in G.edges(data=True, keys = True):
            data['id'] = f'{u}-{v}'
            if data['id'] in edge_ids:
                edges_to_remove.append((u, v, key))
            edge_ids.add(data['id'])
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
        weight = 'length' 
        graph = ox.get_digraph(G)
        _, _, attr_list = next(iter(graph.edges(data=True)))  # type: ignore
        attr_list = cast("dict[str, Any]", attr_list)
        if weight not in attr_list:
            raise ValueError(f"{weight} not in edge attributes.")
        attr = nx.get_node_attributes(graph, weight)
        parallels = (e for e in attr if e[::-1] in attr)
        graph.remove_edges_from({e if attr[e] > attr[e[::-1]] 
                                else e[::-1] for e in parallels})
        return graph
    
@register_graphfcn
class remove_non_pipe_allowable_links(BaseGraphFunction):
    """remove_non_pipe_allowable_links class."""
    def __call__(self,
                 G: nx.Graph,
                 topology_derivation: parameters.TopologyDerivation,
                 **kwargs) -> nx.Graph:
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
        for u, v, keys, data in G.edges(data=True,keys = True):
            for omit in topology_derivation.omit_edges:
                if data.get('highway', None) == omit:
                    # Check whether the 'highway' property is 'omit'
                    edges_to_remove.add((u, v, keys))
                elif data.get(omit, None):
                    # Check whether the 'omit' property of edge is not None 
                    edges_to_remove.add((u, v, keys))
        for edges in edges_to_remove:
            G.remove_edge(*edges)
        return G

@register_graphfcn
class calculate_streetcover(BaseGraphFunction,
                         required_edge_attributes = ['lanes','geometry']
                         ):
    """calculate_streetcover class."""
    # i.e., in osmnx format, i.e., empty for single lane, an int for a
    # number of lanes or a list if the edge has multiple carriageways

    def __call__(self,
                 G: nx.Graph, 
                subcatchment_derivation: parameters.SubcatchmentDerivation, 
                addresses: FilePaths,
                **kwargs) -> nx.Graph:
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
            if data.get('network_type','drive') == 'drive':
                lanes = data.get('lanes',1)
            else:
                lanes = 0
            if isinstance(lanes, list):
                lanes = sum([float(x) for x in lanes])
            else:
                lanes = float(lanes)
            lines.append({'geometry' : data['geometry'].buffer(lanes * 
                                subcatchment_derivation.lane_width,
                                cap_style=2,
                                join_style=2),
                            'u' : u,
                            'v' : v
                            }
                        )
        lines_df = pd.DataFrame(lines)
        lines_gdf = gpd.GeoDataFrame(lines_df, 
                                    geometry=lines_df.geometry,
                                        crs = G.graph['crs'])
        if addresses.streetcover.suffix in ('.geoparquet','.parquet'):
            lines_gdf.to_parquet(addresses.streetcover)
        else:
            lines_gdf.to_file(addresses.streetcover, driver='GeoJSON')

        return G

@register_graphfcn
class double_directed(BaseGraphFunction,
                      required_edge_attributes = ['id']):
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
        arcs_to_remove = [(u,v) for u,v,d in G_new.edges(data=True)
                          if f'{u}-{v}' != d.get('id')]
        
        # Remove the reverse edges
        for u, v in arcs_to_remove:
            G_new.remove_edge(u, v)

        # Add in reversed edges for streets only and with geometry
        for u, v, data in G.edges(data=True):
            include = data.get('edge_type', True)
            if isinstance(include, str):
                include = include == 'street'
            if ((v, u) not in G_new.edges) & include:
                reverse_data = data.copy()
                reverse_data['id'] = f"{data['id']}.reversed"
                new_geometry = shapely.LineString(
                    list(reversed(data['geometry'].coords)))
                reverse_data['geometry'] = new_geometry
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
class split_long_edges(BaseGraphFunction,
                       required_edge_attributes = ['id', 'geometry']):
    """split_long_edges class."""
    
    def __call__(self, 
                 G: nx.Graph, 
                 subcatchment_derivation: parameters.SubcatchmentDerivation, 
                 **kwargs) -> nx.Graph:
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
        new_linestrings = shapely.segmentize([d['geometry'] 
                                             for u,v,d in G.edges(data=True)], 
                                             max_length)
        new_nodes = shapely.get_coordinates(new_linestrings)

        
        new_edges = {}
        for new_linestring, (u,v,d) in zip(new_linestrings, G.edges(data=True)):
            # Create an arc for each segment
            for start, end in zip(new_linestring.coords[:-1],
                                  new_linestring.coords[1:]):
                geom = shapely.LineString([start, end])
                new_edges[(start, end, 0)] = {**d,
                                           'length' : geom.length
                                           }

        # Create new graph
        new_graph = nx.MultiGraph()
        new_graph.graph = G.graph.copy()
        new_graph.add_edges_from(new_edges)
        nx.set_edge_attributes(new_graph, new_edges)
        nx.set_node_attributes(
            new_graph,
            {tuple(node): {'x': node[0], 'y': node[1]} for node in new_nodes}
            )
        return nx.relabel_nodes(new_graph,
                         {node: ix for ix, node in enumerate(new_graph.nodes)}
                         )
    
@register_graphfcn
class merge_street_nodes(BaseGraphFunction):
    """merge_nodes class."""
    def __call__(self, 
                 G: nx.Graph, 
                 subcatchment_derivation: parameters.SubcatchmentDerivation,
                 **kwargs) -> nx.Graph:
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
        street_edges = [(u, v, k) for u, v, k, d in G.edges(data=True, keys=True)
                        if d.get('edge_type','street') == 'street']
        streets = G.edge_subgraph(street_edges).copy()

        # Identify nodes that are within threshold of each other
        mapping = go.merge_points([(d['x'], d['y']) 
                                   for u,d in streets.nodes(data=True)],
                              subcatchment_derivation.node_merge_distance)

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
                node_names[node] = node_indices[mapping[ix]['maps_to']]
                G.nodes[node]['x'] = mapping[ix]['coordinate'][0]
                G.nodes[node]['y'] = mapping[ix]['coordinate'][1]
            else:
                node_names[node] = node

        G = nx.relabel_nodes(G, node_names)

        # Relabelling will create selfloops within a mapping family, which 
        # are removed
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)

        return G
    
@register_graphfcn
class fix_geometries(BaseGraphFunction,
                     required_edge_attributes = ['geometry'],
                     required_node_attributes = ['x', 'y']):
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
            geom = data.get('geometry', None)
            
            start_point_node = (G.nodes[u]['x'], G.nodes[u]['y'])
            end_point_node = (G.nodes[v]['x'], G.nodes[v]['y'])
            if not geom:
                start_point_edge = (None, None)
                end_point_edge = (None, None)
            else:
                start_point_edge = data['geometry'].coords[0]
                end_point_edge = data['geometry'].coords[-1]

            if (start_point_edge == end_point_node) & \
                    (end_point_edge == start_point_node):
                data['geometry'] = data['geometry'].reverse()
            elif (start_point_edge != start_point_node) | \
                    (end_point_edge != end_point_node):
                data['geometry'] = shapely.LineString([start_point_node,
                                                       end_point_node])
        return G

@register_graphfcn
class clip_to_catchments(BaseGraphFunction,
                         required_node_attributes = ['x','y'],
                         required_edge_attributes = ['length'],
                         adds_node_attributes = ['community','basin']):
    """clip_to_catchments class."""
    def __call__(self, 
                 G: nx.Graph,
                addresses: FilePaths,
                subcatchment_derivation: parameters.SubcatchmentDerivation,
                **kwargs) -> nx.Graph:
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
        subbasins = go.derive_subbasins_streamorder(addresses.elevation,
                        subcatchment_derivation.subbasin_streamorder,
                        x = list(nx.get_node_attributes(G, 'x').values()),
                        y = list(nx.get_node_attributes(G, 'y').values()))
        
        if verbose():
            subbasins.to_file(
                str(addresses.nodes).replace('nodes','subbasins'), 
                driver='GeoJSON')

        # Extract street network
        street = G.copy()
        street.remove_edges_from([(u, v) for u, v, d in street.edges(data=True)
                                  if d.get('edge_type', 'street') != 'street'])

        # Create gdf of street points
        street_points = gpd.GeoDataFrame(G.nodes,
            columns = ['id'],
            geometry = gpd.points_from_xy(
                [G.nodes[u]['x'] for u in G.nodes],
                [G.nodes[u]['y'] for u in G.nodes]
                ),
            crs = G.graph['crs']
            ).set_index('id')
        
        # Classify street points by subbasin
        street_points = gpd.sjoin(street_points,
                                subbasins.set_index('basin'),
                                how='left',
                        ).rename(columns = {'index_right': 'basin'})

        if subcatchment_derivation.subbasin_clip_method == 'subbasin':
            edges_to_remove = [
                (u,v) for u, v in G.edges()
                if street_points.loc[u,'basin'] != street_points.loc[v,'basin']
                ]
            G.remove_edges_from(edges_to_remove)
            return G
        
        # Derive road network clusters
        louv_membership = nx.community.louvain_communities(street,
                                                           weight = 'length',
                                                           seed = 1)
        
        street_points['community'] = 0 
        # Assign louvain membership to street points
        for ix, community in enumerate(louv_membership):
            street_points.loc[list(community), 'community'] = ix
        
        
        
        # Introduce a non catchment basin for nan
        street_points['basin'] = street_points['basin'].fillna(-1)
        # TODO possibly it makes sense to just remove these nodes, or at least
        # any communities that are all nan
        
        nx.set_node_attributes(G,
                               street_points['community'].to_dict(),
                               'community')
        nx.set_node_attributes(G,
                               street_points['basin'].to_dict(),
                               'basin')
        

        # Calculate most percentage of each subbasin in each community
        community_basin = (
            street_points
            .groupby('community')
            .basin
            .value_counts()
            .reset_index()
        )
        community_size = (
            street_points
            .community
            .value_counts()
            .reset_index()
        )
        community_basin = community_basin.merge(community_size, 
                                                on='community',
                                                how = 'left',
                                                suffixes = ('_basin', '_size')
                                                )
        
        # Normalize
        community_basin['percentage'] = (
            community_basin['count_basin'] / community_basin['count_size']
            )
        
        # Identify community-basin combinations where the percentage is less than
        # the threshold
        community_omit = community_basin.loc[
            community_basin['percentage'] <= 
            subcatchment_derivation.subbasin_membership
            ]

        community_basin = community_basin.set_index('basin')
        
        
        # Cut links between communities in community_omit and commuities in those
        # basins
        arcs_to_remove = []
        street_points = street_points.reset_index().set_index('basin')
        for idx, row in community_omit.iterrows():
            community_nodes = louv_membership[int(row['community'])]
            basin_nodes = street_points.loc[[row['basin']],'id']
            basin_nodes = set(basin_nodes).difference(community_nodes)
            
            # Include both directions because operation should work on 
            # undirected or directed graph
            arcs_to_remove.extend(
                [(u, v, 0) for u, v in product(community_nodes, basin_nodes)] +
                [(v, u, 0) for u, v in product(community_nodes, basin_nodes)]
                )
        G.remove_edges_from(set(G.edges).intersection(arcs_to_remove))
        if G.is_directed():
            subgraphs = len(list(nx.weakly_connected_components(G)))
        else:
            subgraphs = len(list(nx.connected_components(G)))
        logger.info(f"clip_to_catchments has created {subgraphs} subgraphs.")
        return G
        

@register_graphfcn
class calculate_contributing_area(BaseGraphFunction,
                                required_edge_attributes = ['id', 'geometry'],
                                adds_edge_attributes = ['contributing_area'],
                                adds_node_attributes = ['contributing_area']):
    """calculate_contributing_area class."""
    
    def __call__(self, G: nx.Graph, 
                         subcatchment_derivation: parameters.SubcatchmentDerivation,
                         addresses: FilePaths,
                         **kwargs) -> nx.Graph:
        """Calculate the contributing area for each edge.
        
        This function calculates the contributing area for each edge. The
        contributing area is the area of the subcatchment that drains to the
        edge. The contributing area is calculated from the elevation data. 
        Runoff coefficient (RC) for each contributing area is also calculated, 
        the RC is calculated using `addresses.buildings` and 
        `addresses.streetcover`.        

        Also writes the file 'subcatchments.geojson' to addresses.subcatchments.

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
            go.burn_shape_in_raster([d['geometry'] for u,v,d in G.edges(data=True)],
                    subcatchment_derivation.carve_depth,
                    addresses.elevation,
                    temp_fid)
            
            # Derive
            subs_gdf = go.derive_subcatchments(G,temp_fid)
            if verbose():
                subs_gdf.to_file(addresses.subcatchments, driver='GeoJSON')

        # Calculate runoff coefficient (RC)
        if addresses.building.suffix in ('.geoparquet','.parquet'):
            buildings = gpd.read_parquet(addresses.building)
        else:
            buildings = gpd.read_file(addresses.building)
        if addresses.streetcover.suffix in ('.geoparquet','.parquet'):
            streetcover = gpd.read_parquet(addresses.streetcover)
        else:
            streetcover = gpd.read_file(addresses.streetcover)

        subs_rc = go.derive_rc(subs_gdf, buildings, streetcover)

        # Write subs
        # TODO - could just attach subs to nodes where each node has a list of subs
        subs_rc.to_file(addresses.subcatchments, driver='GeoJSON')

        # Assign contributing area
        imperv_lookup = subs_rc.set_index('id').impervious_area.to_dict()
        
        # Set node attributes
        nx.set_node_attributes(G, 0.0, 'contributing_area')
        nx.set_node_attributes(G, imperv_lookup, 'contributing_area')

        # Prepare edge attributes
        edge_attributes = {edge: G.nodes[edge[0]]['contributing_area'] 
                           for edge in G.edges}

        # Set edge attributes
        nx.set_edge_attributes(G, edge_attributes, 'contributing_area')
        return G

@register_graphfcn
class set_elevation(BaseGraphFunction,
                    required_node_attributes = ['x', 'y'],
                    adds_node_attributes = ['surface_elevation']):
    """set_elevation class."""
    
    def __call__(self, G: nx.Graph, 
                  addresses: FilePaths,
                  **kwargs) -> nx.Graph:
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
        x = [d['x'] for x, d in G.nodes(data=True)]
        y = [d['y'] for x, d in G.nodes(data=True)]
        elevations = go.interpolate_points_on_raster(x, 
                                                    y, 
                                                    addresses.elevation)
        elevations_dict = {id_: elev for id_, elev in zip(G.nodes, elevations)}
        nx.set_node_attributes(G, elevations_dict, 'surface_elevation')
        return G

@register_graphfcn
class set_surface_slope(BaseGraphFunction,
                        required_node_attributes = ['surface_elevation'],
                        adds_edge_attributes = ['surface_slope']):
    """set_surface_slope class."""

    def __call__(self, G: nx.Graph,
                      **kwargs) -> nx.Graph:
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
        slope_dict = {(u, v, k): (G.nodes[u]['surface_elevation'] - \
                                  G.nodes[v]['surface_elevation']) 
                        / d['length'] for u, v, k, d in G.edges(data=True,
                                                                keys=True)}

        # Set the 'surface_slope' attribute for all edges
        nx.set_edge_attributes(G, slope_dict, 'surface_slope')
        return G
    
@register_graphfcn
class set_chahinian_slope(BaseGraphFunction,
                          required_edge_attributes = ['surface_slope'],
                          adds_edge_attributes = ['chahinian_slope']):
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
        weights = np.interp(np.asarray(list(slope.values())) * 100, 
                            slope_points,
                            weights, 
                            left=1, 
                            right=1)
        nx.set_edge_attributes(G, dict(zip(slope, weights)), "chahinian_slope")
        
        return G
    
@register_graphfcn
class set_chahinian_angle(BaseGraphFunction,
                         required_node_attributes = ['x','y'],
                         adds_edge_attributes = ['chahinian_angle']):
    """set_chahinian_angle class."""

    def __call__(self, G: nx.Graph, 
                       **kwargs) -> nx.Graph:
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
        for u,v,d in G.edges(data=True):
            min_weight = float('inf')
            for node in G.successors(v):
                if node == u:
                    continue
                    
                p1 = (G.nodes[u]['x'], G.nodes[u]['y'])
                p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
                p3 = (G.nodes[node]['x'], G.nodes[node]['y'])
                angle = go.calculate_angle(p1,p2,p3)
                chahinian_weight = np.interp(angle,
                                            [0, 90, 135, 180, 225, 270, 360],
                                            [1, 0.2, 0.7, 0, 0.7, 0.2, 1]
                                            )
                min_weight = min(chahinian_weight, min_weight)
            if min_weight == float('inf'):
                min_weight = 0
            d['chahinian_angle'] = min_weight
        return G

@register_graphfcn
class calculate_weights(BaseGraphFunction,
                        required_edge_attributes = 
                        parameters.TopologyDerivation().weights,
                        adds_edge_attributes = ['weight']):
    """calculate_weights class."""
    # TODO.. I guess if someone defines their own weights, this will need 
    # to change, will want an automatic way to do that...

    def __call__(self, G: nx.Graph, 
                       topology_derivation: parameters.TopologyDerivation,
                       **kwargs) -> nx.Graph:
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
        bounds: Dict[Any, List[float]] = defaultdict(lambda: [np.Inf, -np.Inf])

        for w in topology_derivation.weights:
            bounds[w][0] = min(nx.get_edge_attributes(G, w).values()) # lower bound
            bounds[w][1] = max(nx.get_edge_attributes(G, w).values()) # upper bound

        # Avoid division by zero
        bounds = {w : [b[0], b[1]] for w, b in bounds.items() if b[0] != b[1]}
        
        G = G.copy()
        eps = np.finfo(float).eps
        for u, v, d in G.edges(data=True):
            total_weight = 0
            for attr, bds in bounds.items():
                # Normalise
                weight = max((d[attr] - bds[0]) / (bds[1] - bds[0]), eps)
                # Exponent
                weight = weight ** getattr(topology_derivation,f'{attr}_exponent')
                # Scaling
                weight = weight * getattr(topology_derivation,f'{attr}_scaling')
                # Sum
                total_weight += weight
            # Set
            d['weight'] = total_weight
        return G

def _get_points(G: nx.Graph) -> tuple[Dict[str, shapely.Point],
                                      Dict[str, shapely.Point]]:
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
        pd.DataFrame(etypes.items(), 
                    columns=['key', 'type'])
        .explode('key')
        .reset_index(drop=True)
        .groupby("type")["key"]
        .apply(list)
        .to_dict()
    )
    river_points = {n: shapely.Point(G.nodes[n]['x'], G.nodes[n]['y']) 
                    for n in n_types.get('river',{})}
    street_points = {n: shapely.Point(G.nodes[n]['x'], G.nodes[n]['y']) 
                     for n in n_types['street']}

    return river_points, street_points

def _pair_rivers(G: nx.Graph, 
                 river_points: Dict[str, shapely.Point], 
                 street_points: Dict[str, shapely.Point], 
                 river_buffer_distance: float,
                 outlet_length: float) -> nx.Graph:
    """Pair river and street nodes.

    Pair river and street nodes within a certain distance of each other. If
    there are no plausible outlets for an entire subgraph, then a dummy river
    node is created and the lowest elevation node is used as the outlet.

    Args:
        G (nx.Graph): A graph
        river_points (dict): A dictionary of river points
        street_points (dict): A dictionary of street points
        river_buffer_distance (float): The distance within which a river and
            street node can be paired
        outlet_length (float): The length of the outlet

    Returns:
        G (nx.Graph): A graph
    """
    matched_outlets = {}

    # Insert a dummy river node and use lowest elevation node as outlet
    # for each subgraph with no matched outlets
    subgraphs = []
    for sg in nx.weakly_connected_components(G):      
        sg = G.subgraph(sg).copy()
        subgraphs.append(sg)

        # Ignore the rivers, they are drained later
        if all([d['edge_type'] == "river" for _,_,d in sg.edges(data=True)]):
            continue
        
        # Pair up the river and street nodes for each subgraph
        street_points_ = {k: v for k, v in street_points.items() 
                            if k in sg.nodes}
        
        subgraph_outlets = go.nearest_node_buffer(street_points_,
                                    river_points,
                                    river_buffer_distance)

        # Check if there are any matched outlets
        if subgraph_outlets:
            # Update all matched outlets
            matched_outlets.update(subgraph_outlets)
            continue

        # In cases of e.g., an area with no rivers to discharge into or too
        # small a buffer

        # Identify the lowest elevation node
        lowest_elevation_node = min(sg.nodes, 
                    key = lambda x: sg.nodes[x]['surface_elevation'])
        
        # Create a dummy river to discharge into
        name = f'{lowest_elevation_node}-dummy_river'
        dummy_river = {'id' : name,
                    'x' : G.nodes[lowest_elevation_node]['x'] + 1,
                    'y' : G.nodes[lowest_elevation_node]['y'] + 1}
        sg.add_node(name)
        nx.set_node_attributes(sg, {name : dummy_river})

        # Update function's dicts
        matched_outlets[lowest_elevation_node] = name
        river_points[name] = shapely.Point(dummy_river['x'],
                                            dummy_river['y'])
        
        logger.warning(f"""No outlets found for subgraph containing 
                        {lowest_elevation_node}, using this node as outlet.""")
        
    G = nx.compose_all(subgraphs)

    # Add edges between the paired river and street nodes
    for street_id, river_id in matched_outlets.items():
        # TODO instead use a weight based on the distance between the two nodes
        G.add_edge(street_id, river_id,
                    **{'length' : outlet_length,
                        'weight' : outlet_length,
                    'edge_type' : 'outlet',
                    'geometry' : shapely.LineString([street_points[street_id],
                                                river_points[river_id]]),
                    'id' : f'{street_id}-{river_id}-outlet'})
            
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
    G.add_node('waste')

    for node in G_.nodes:
        if G.out_degree(node) == 0:
            # Location of the waste node doesn't matter - so if there
            # are multiple river nodes with out_degree 0 - that's fine.
            G.nodes['waste']['x'] = G.nodes[node]['x'] + 1
            G.nodes['waste']['y'] = G.nodes[node]['y'] + 1
            G.add_edge(node,
                        'waste',
                        **{'length' : 0,
                            'weight' : 0,
                        'edge_type' : 'waste-outlet',
                        'id' : f'{node}-waste-outlet'})
    return G

def _connect_mst_outlets(paired_G: nx.Graph, raw_G: nx.Graph) -> nx.Graph:
    """Connect outlets to a waste node.

    Run a minimum spanning tree (MST) on the paired graph to identify the 
    'efficient' outlets. These outlets are inserted into the original graph.

    Args:
        paired_G (nx.Graph): A graph where streets and rivers are paired with 
            outlets.
        raw_G (nx.Graph): A graph where streets and rivers are separated

    Returns:
        (nx.Graph): A graph
    """
    # Find shortest path to identify only 'efficient' outlets. The MST
    # makes sense here over shortest path as each node is only allowed to
    # be visited once - thus encouraging fewer outlets. In shortest path
    # nodes near rivers will always just pick their nearest river node.
    T = nx.minimum_spanning_tree(paired_G.to_undirected(),
                                    weight = 'length')
    
    # Retain the shortest path outlets in the original graph
    for u,v,d in T.edges(data=True):
        if (d['edge_type'] == 'outlet') & (v != 'waste') & (u != 'waste'):
            if u not in raw_G.nodes():
                raw_G.add_node(u, **paired_G.nodes[u])
            elif v not in raw_G.nodes():
                raw_G.add_node(v, **paired_G.nodes[v])

            # Need to check both directions since T is undirected
            if (u,v) in paired_G.edges():
                raw_G.add_edge(u,v,**d)
            elif (v,u) in paired_G.edges():
                raw_G.add_edge(v,u,**d)
            else:
                raise ValueError(f"Edge {u}-{v} not found in paired_G")
            
                
    return raw_G

@register_graphfcn
class identify_outlets(BaseGraphFunction,
                       required_edge_attributes = ['length', 'edge_type'],
                       required_node_attributes = ['x', 'y','surface_elevation']):
    """identify_outlets class."""

    def __call__(self, 
                 G: nx.Graph,
                 outlet_derivation: parameters.OutletDerivation,
                 **kwargs) -> nx.Graph:
        """Identify outlets in a combined river-street graph.

        This function identifies outlets in a combined river-street graph. An
        outlet is a node that is connected to a river and a street. Each street
        node is paired with the nearest river node provided that it is within
        a distance of outlet_derivation.river_buffer_distance - this provides a
        large set of plausible outlets. If there are no plausible outlets for an
        entire subgraph, then a dummy river node is created and the lowest
        elevation node is paired with it. Any street->river/outlet link is given
        a `weight` and `length` of outlet_derivation.outlet_length, this is to
        ensure some penalty on the total number of outlets selected.

        Two methods are available for determining which plausible outlets to 
        retain:

        - `withtopo`: all plausible outlets are retained, connected to a single
        'waste' node and assumed to be later identified as part of the 
        `derive_topology` graphfcn.
        
        - `separate`: the retained outlets are those that are selected as part 
        of the minimum spanning tree (MST) of the combined street-river graph. 
        This method can be temporamental because the MST is undirected, because
        rivers are inherently directed unusual outlet locations may be retained.

        Args:
            G (nx.Graph): A graph
            outlet_derivation (parameters.OutletDerivation): An OutletDerivation
                parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        
        river_points, street_points = _get_points(G)
        
        G_ = _pair_rivers(G, 
            river_points, 
            street_points, 
            outlet_derivation.river_buffer_distance,
            outlet_derivation.outlet_length)
        
        # Set the length of the river edges to 0 - from a design perspective 
        # once water is in the river we don't care about the length - since it 
        # costs nothing
        for _,_,d in G_.edges(data=True):
            if d['edge_type'] == 'river':
                d['length'] = 0
                d['weight'] = 0
        
        # Add edges from the river nodes to a waste node
        G_ = _root_nodes(G_)

        if outlet_derivation.method == 'withtopo':
            # The outlets can be derived as part of the shortest path calculations
            return G_
        elif outlet_derivation.method == 'separate':
            return _connect_mst_outlets(G_, G)
        else:
            raise ValueError(f"Unknown method {outlet_derivation.method}")

def _filter_streets(G):
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
        if d['edge_type'] != 'street':
            if d['edge_type'] == 'outlet':
                nodes_to_remove.append(v)
            else:
                nodes_to_remove.extend((u,v))
    G.remove_nodes_from(nodes_to_remove)
    return G

@register_graphfcn
class derive_topology(BaseGraphFunction,
                      required_edge_attributes = ['edge_type', # 'rivers' and 'streets'
                                                  'weight'],
                      adds_node_attributes = ['outlet', 
                                              'shortest_path']):
    """derive_topology class."""
    

    def __call__(self, 
                 G: nx.Graph,
                 outlet_derivation: parameters.OutletDerivation,
                 **kwargs) -> nx.Graph:
        """Derive the topology of a graph.

        Derives the network topology based on the weighted graph of potential
        pipe carrying edges in the graph.

        Two methods are available:
        - `separate`: The original and that assumes outlets have already been
        narrowed down from the original plausible set. Runs a djiikstra-based 
        algorithm to identify the shortest path from each node to its nearest 
        outlet (weighted by the 'weight' edge value). The 
        returned graph is one that only contains the edges that feature  on the 
        shortest paths. 
        - `withtopo`: The alternative method that assumes no narrowing of plausible
        outlets has been performed. This method runs a Tarjan's algorithm to
        identify the spanning forest starting from a `waste` node that all 
        plausible outlets are connected to (whether via a river or directly).
        
        In both methods, street nodes that have no plausible route to any outlet
        are removed.

        Args:
            G (nx.Graph): A graph
            outlet_derivation (parameters.OutletDerivation): An OutletDerivation
                parameter object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        
        visited: set = set()
        
        # Increase recursion limit to allow to iterate over the entire graph
        # Seems to be the quickest way to identify which nodes have a path to
        # the outlet
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(original_limit, len(G.nodes)))

        # Identify outlets
        if outlet_derivation.method == 'withtopo':
            visited = set(nx.ancestors(G,'waste')) | {'waste'}

            # Remove nodes not reachable from waste
            G.remove_nodes_from(set(G.nodes) - visited)

            # Run shorted path
            G = shortest_path_utils.tarjans_pq(G,'waste')
            
            G = _filter_streets(G)
        else:
            outlets = [u for u,v,d in G.edges(data=True) if d['edge_type'] == 'outlet']
            visited = set(outlets)
            for outlet in outlets:
                visited = visited | set(nx.ancestors(G,outlet))

            G.remove_nodes_from(set(G.nodes) - visited)
            G = _filter_streets(G)

            # Check for negative cycles
            if nx.negative_edge_cycle(G, weight = 'weight'):
                logger.warning('Graph contains negative cycle')

            G = shortest_path_utils.dijkstra_pq(G, outlets)

        # Reset recursion limit
        sys.setrecursionlimit(original_limit)

        # Log total weight
        total_weight = sum([d['weight'] for u,v,d in G.edges(data=True)])
        logger.info(f"Total graph weight {total_weight}.")

        return G

def design_pipe(ds_elevation: float,
                       chamber_floor: float, 
                       edge_length: float,
                       hydraulic_design: parameters.HydraulicDesign,
                       Q: float
                       ) -> nx.Graph:
    """Design a pipe.

    This function designs a pipe by iterating over a range of diameters and
    depths. It returns the diameter and depth of the pipe that minimises the
    cost function, while also maintaining or minimising feasibility parameters
    associated with: surcharging, velocity and filling ratio.

    Args:
        ds_elevation (float): The downstream elevationq
        chamber_floor (float): The elevation of the chamber floor
        edge_length (float): The length of the edge
        hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
            object
        Q (float): The flow rate

    Returns:
        diam (float): The diameter of the pipe
        depth (float): The depth of the pipe
    """
    designs = product(hydraulic_design.diameters,
                          np.linspace(hydraulic_design.min_depth, 
                                      hydraulic_design.max_depth, 
                                      10) # TODO should 10 be a param?
                          )
    pipes = []
    for diam, depth in designs:
        A = (np.pi * diam ** 2 / 4)
        n = 0.012 # mannings n
        R = A / (np.pi * diam)  # hydraulic radius
        # TODO... presumably need to check depth > (diam + min_depth)

        elev_diff = chamber_floor - (ds_elevation - depth)
        slope = elev_diff / edge_length
        # Always pick a pipe that is feasible without surcharging 
        # if available
        surcharge_feasibility = 0.0
        # Use surcharged elevation                        
        while slope <= 0:
            surcharge_feasibility += 0.05
            slope = (chamber_floor + surcharge_feasibility - 
                    (ds_elevation - depth)) / edge_length
            # TODO could make the feasibility penalisation increase
            # when you get above surface_elevation[node]... but 
            # then you'd need a feasibility tracker and an offset 
            # tracker                    
        v = (slope ** 0.5) * (R ** (2/3)) / n
        filling_ratio = Q / (v * A)
        # buffers from: https://www.polypipe.com/sites/default/files/Specification_Clauses_Underground_Drainage.pdf 
        average_depth = (depth + chamber_floor) / 2
        V = edge_length * (diam + 0.3) * (average_depth + 0.1)
        cost = 1.32 / 2000 * (9579.31 * diam ** 0.5737 + 1153.77 * V**1.31)
        v_feasibility = max(hydraulic_design.min_v - v, 0) + \
            max(v - hydraulic_design.max_v, 0)
        fr_feasibility = max(filling_ratio - hydraulic_design.max_fr, 0)
        """
        TODO shear stress... got confused here
        density = 1000
        dyn_visc = 0.001
        hydraulic_diameter = 4 * (A * filling_ratio**2) / \
            (np.pi * diam * filling_ratio)
        Re = density * v * 2 * (diam / 4) * (filling_ratio ** 2) / dyn_visc
        fd = 64 / Re
        shear_stress = fd * density * v**2 / fd
        shear_feasibility = max(min_shear - shear_stress, 0)
        """
        slope = (chamber_floor - (ds_elevation - depth)) / edge_length
        pipes.append({'diam' : diam,
                    'depth' : depth,
                    'slope' : slope,
                    'v' : v,
                    'fr' : filling_ratio,
                    # 'tau' : shear_stress,
                    'cost' : cost,
                    'v_feasibility' : v_feasibility,
                    'fr_feasibility' : fr_feasibility,
                    'surcharge_feasibility' : surcharge_feasibility,
                    # 'shear_feasibility' : shear_feasibility
                    })
    
    pipes_df = pd.DataFrame(pipes).dropna()
    if pipes_df.shape[0] > 0:
        ideal_pipe = pipes_df.sort_values(by=['surcharge_feasibility',
                                                    'v_feasibility',
                                                    'fr_feasibility',
                                                    # 'shear_feasibility',
                                                    'depth',
                                                    'cost'], 
                                                ascending = True).iloc[0]
        return ideal_pipe.diam, ideal_pipe.depth
    else:
        raise Exception('something odd - no non nan pipes')
    
def process_successors(G: nx.Graph, 
                       node: Hashable, 
                       surface_elevations: dict[Hashable, float], 
                       chamber_floor: dict[Hashable, float], 
                       edge_diams: dict[tuple[Hashable,Hashable,int], float],
                       hydraulic_design: parameters.HydraulicDesign
                       ) -> None:
    """Process the successors of a node.

    This function processes the successors of a node. It designs a pipe to the
    downstream node and sets the diameter and downstream invert level of the
    pipe. It also sets the downstream invert level of the downstream node. It
    returns None but modifies the edge_diams and chamber_floor dictionaries.

    Args:
        G (nx.Graph): A graph
        node (Hashable): A node
        surface_elevations (dict): A dictionary of surface elevations keyed by
            node
        chamber_floor (dict): A dictionary of chamber floor elevations keyed by
            node
        edge_diams (dict): A dictionary of pipe diameters keyed by edge
        hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
            object
    """
    for ix, ds_node in enumerate(G.successors(node)):
        edge = G.get_edge_data(node,ds_node,0)
        # Find contributing area with ancestors
        # TODO - could do timearea here if i hated myself enough
        anc = nx.ancestors(G,node).union([node])
        tot = sum([G.nodes[anc_node]['contributing_area'] for anc_node in anc])
        
        M3_PER_HR_TO_M3_PER_S = 1 / 60 / 60
        Q = tot * hydraulic_design.precipitation * M3_PER_HR_TO_M3_PER_S
        
        # Design the pipe to find the diameter and invert depth
        diam, depth = design_pipe(surface_elevations[ds_node],
                                    chamber_floor[node],
                                    edge['length'],
                                    hydraulic_design,
                                    Q
                                    )
        edge_diams[(node,ds_node,0)] = diam
        chamber_floor[ds_node] = surface_elevations[ds_node] - depth
        if ix > 0:
            logger.warning('''a node has multiple successors, 
                not sure how that can happen if using shortest path
                to derive topology''')

@register_graphfcn
class pipe_by_pipe(BaseGraphFunction,
                   required_edge_attributes = ['length'],
                   required_node_attributes = ['contributing_area', 
                                               'surface_elevation'],
                   adds_edge_attributes = ['diameter'],
                   adds_node_attributes = ['chamber_floor_elevation']):
    """pipe_by_pipe class."""
    # If doing required_graph_attributes - it would be something like 'dendritic'

    def __call__(self, 
                 G: nx.Graph, 
                 hydraulic_design: parameters.HydraulicDesign,
                 **kwargs
                 )->nx.Graph:
        """Pipe by pipe hydraulic design.

        Starting from the most upstream node, design a pipe to the downstream node
        specifying a diameter and downstream invert level. A range of diameters and
        invert levels are tested (ranging between conditions defined in 
        hydraulic_design). From the tested diameters/inverts, a selection is made based
        on each pipe's satisfying feasibility constraints on: surcharge velocity,
        filling ratio, (and shear stress - not currently implemented). Prioritising 
        feasibility in this order it identifies pipes with the preferred feasibility
        level. If multiple pipes are feasible, it picks the lowest cost pipe. Once
        the feasible pipe is identified, the diameter and downstream invert are set
        and then the next downstream pipe can be designed. 

        This approach is based on the pipe-by-pipe design proposed in:
            https://doi.org/10.1016/j.watres.2021.117903

        Args:
            G (nx.Graph): A graph
            hydraulic_design (parameters.HydraulicDesign): A HydraulicDesign parameter
                object
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            G (nx.Graph): A graph
        """
        G = G.copy()
        surface_elevations = nx.get_node_attributes(G, 'surface_elevation')
        topological_order = list(nx.topological_sort(G))
        chamber_floor = {}
        edge_diams: dict[tuple[Hashable,Hashable,int],float] = {}
        # Iterate over nodes in topological order
        for node in tqdm(topological_order,
                         disable = not verbose()):
            # Check if there's any nodes upstream, if not set the depth to min_depth
            if len(nx.ancestors(G,node)) == 0:
                chamber_floor[node] = surface_elevations[node] - \
                    hydraulic_design.min_depth
            
            process_successors(G, 
                       node, 
                       surface_elevations, 
                       chamber_floor, 
                       edge_diams,
                       hydraulic_design
                       )
            
        nx.function.set_edge_attributes(G, edge_diams, "diameter")
        nx.function.set_node_attributes(G, chamber_floor, "chamber_floor_elevation")
        return G
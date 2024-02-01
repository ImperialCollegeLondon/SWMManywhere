# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
import json
import tempfile
from pathlib import Path
from typing import Callable

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely import geometry as sgeom

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters


def load_graph(fid: Path) -> nx.Graph:
    """Load a graph from a file saved with save_graph.

    Args:
        fid (Path): The path to the file

    Returns:
        G (nx.Graph): A graph
    """
    # Define the custom decoding function    
    with open(fid, 'r') as file:
        json_data = json.load(file)

    G = nx.node_link_graph(json_data,directed=True)
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            geometry_coords = data['geometry']
            line_string = sgeom.LineString(geometry_coords)
            data['geometry'] = line_string
    return G

def save_graph(G: nx.Graph, 
               fid: Path):
    """Save a graph to a file.

    Args:
        G (nx.Graph): A graph
        fid (Path): The path to the file
    """
    json_data = nx.node_link_data(G)
    def serialize_line_string(obj):
        if isinstance(obj, sgeom.LineString):
            return list(obj.coords)
        else:
            return obj
    with open(fid, 'w') as file:
        json.dump(json_data, 
                  file,
                  default = serialize_line_string)
        
graphfcns = {}

def register_graphfcn(func: Callable[..., 
                                     nx.Graph]) -> Callable[..., 
                                                            nx.Graph]:
    """Register a graph function.

    Args:
        func (Callable): A function that takes a graph and other parameters

    Returns:
        func (Callable): Returns the same function
    """
    # Add the function to the registry
    graphfcns[func.__name__] = func
    return func

def get_osmid_id(data):
    """Get the ID of an edge."""
    id_ = data.get('osmid', data.get('id'))
    if isinstance(id_, list):
        id_ = id_[0]
    return id_

@register_graphfcn
def assign_id(G: nx.Graph, 
              **kwargs):
    """Assign an ID to each edge.

    This function takes a graph and assigns an ID to each edge. The ID is
    assigned to the 'id' attribute of each edge. Needed because some edges
    have 'osmid', some have a list of 'osmid', others have 'id'.

    Requires a graph with edges that have:
        - 'osmid' or 'id'
    
    Adds the edge attributes:
        - 'id'

    Args:
        G (nx.Graph): A graph
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): The same graph with an ID assigned to each edge
    """
    for u, v, data in G.edges(data=True):
        data['id'] = get_osmid_id(data)
    return G

@register_graphfcn
def format_osmnx_lanes(G: nx.Graph, 
                       subcatchment_derivation: parameters.SubcatchmentDerivation, 
                       **kwargs):
    """Format the lanes attribute of each edge and calculates width.

    Requires a graph with edges that have:
        - 'lanes' (in osmnx format, i.e., empty for single lane, an int for a
            number of lanes or a list if the edge has multiple carriageways)
    
    Adds the edge attributes:
        - 'lanes' (float)
        - 'width' (float)

    Args:
        G (nx.Graph): A graph
        subcatchment_derivation (parameters.SubcatchmentDerivation): A
            SubcatchmentDerivation parameter object
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): A graph
    """
    G = G.copy()
    for u, v, data in G.edges(data=True):
        lanes = data.get('lanes',1)
        if isinstance(lanes, list):
            lanes = sum([float(x) for x in lanes])
        else:
            lanes = float(lanes)
        data['lanes'] = lanes
        data['width'] = lanes * subcatchment_derivation.lane_width
    return G

@register_graphfcn
def double_directed(G: nx.Graph, **kwargs):
    """Create a 'double directed graph'.

    This function duplicates a graph and adds reverse edges to the new graph. 
    These new edges share the same data as the 'forward' edges but have a new 
    'id'. An undirected graph is not suitable because it removes data of one of 
    the edges if there are edges in both directions between two nodes 
    (necessary to preserve, e.g., consistent 'width').

    Requires a graph with edges that have:
        - 'id'
    
    Args:
        G (nx.Graph): A graph
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): A graph
    """
    G_new = G.copy()
    for u, v, data in G.edges(data=True):
        if (v, u) not in G.edges:
            reverse_data = data.copy()
            reverse_data['id'] = '{0}.reversed'.format(data['id'])
            G_new.add_edge(v, u, **reverse_data)
    return G_new

@register_graphfcn
def split_long_edges(graph: nx.Graph, 
                     subcatchment_derivation: parameters.SubcatchmentDerivation, 
                     **kwargs):
    """Split long edges into shorter edges.

    This function splits long edges into shorter edges. The edges are split
    into segments of length 'max_street_length'. The first and last segment
    are connected to the original nodes. Intermediate segments are connected
    to newly created nodes.

    Requires a graph with edges that have:
        - 'geometry' (shapely LineString)
        - 'length' (float)
        - 'id' (str)
    
    Args:
        graph (nx.Graph): A graph
        subcatchment_derivation (parameters.SubcatchmentDerivation): A
            SubcatchmentDerivation parameter object
        **kwargs: Additional keyword arguments are ignored.
    
    Returns:
        graph (nx.Graph): A graph
    """
    #TODO refactor obviously
    max_length = subcatchment_derivation.max_street_length
    graph = graph.copy()
    edges_to_remove = []
    edges_to_add = []
    nodes_to_add = []
    maxlabel = max(graph.nodes) + 1
    ll = 0

    def create_new_edge_data(line, data, id_):
        new_line = sgeom.LineString(line)
        new_data = data.copy()
        new_data['id'] = id_
        new_data['length'] = new_line.length
        new_data['geometry'] =  sgeom.LineString([(x[0], x[1]) 
                                                  for x in new_line.coords])
        return new_data

    for u, v, data in graph.edges(data=True):
        line = data['geometry']
        length = data['length']
        if ((u, v) not in edges_to_remove) & ((v, u) not in edges_to_remove):
            if length > max_length:
                new_points = [sgeom.Point(x) 
                              for x in ox.utils_geo.interpolate_points(line, 
                                                                       max_length)]
                if len(new_points) > 2:
                    for ix, (start, end) in enumerate(zip(new_points[:-1], 
                                                         new_points[1:])):
                        new_data = create_new_edge_data([start, 
                                                        end], 
                                                        data, 
                                                        '{0}.{1}'.format(
                                                            data['id'],ix))
                        if (v,u) in graph.edges:
                            # Create reversed data
                            data_r = graph.get_edge_data(v, u).copy()[0]
                            id_ = '{0}.{1}'.format(data_r['id'],ix)
                            new_data_r = create_new_edge_data([end, start], 
                                                              data_r.copy(), 
                                                              id_)
                        if ix == 0:
                            # Create start to first intermediate
                            edges_to_add.append((u, maxlabel + ll, new_data.copy()))
                            nodes_to_add.append((maxlabel + ll, 
                                                 {'x': 
                                                  new_data['geometry'].coords[-1][0],
                                                  'y': 
                                                  new_data['geometry'].coords[-1][1]}))
                            
                            if (v, u) in graph.edges:
                                # Create first intermediate to start
                                edges_to_add.append((maxlabel + ll, 
                                                     u, 
                                                     new_data_r.copy()))
                            
                            ll += 1
                        elif ix == len(new_points) - 2:
                            # Create last intermediate to end
                            edges_to_add.append((maxlabel + ll - 1, 
                                                 v, 
                                                 new_data.copy()))
                            if (v, u) in graph.edges:
                                # Create end to last intermediate
                                edges_to_add.append((v, 
                                                     maxlabel + ll - 1, 
                                                     new_data_r.copy()))
                        else:
                            nodes_to_add.append((maxlabel + ll, 
                                                 {'x': 
                                                  new_data['geometry'].coords[-1][0],
                                                  'y': 
                                                  new_data['geometry'].coords[-1][1]}))
                            # Create N-1 intermediate to N intermediate
                            edges_to_add.append((maxlabel + ll - 1, 
                                                 maxlabel + ll, 
                                                 new_data.copy()))
                            if (v, u) in graph.edges:
                                # Create N intermediate to N-1 intermediate
                                edges_to_add.append((maxlabel + ll, 
                                                     maxlabel + ll - 1, 
                                                     new_data_r.copy()))
                            ll += 1
                    edges_to_remove.append((u, v))
                    if (v, u) in graph.edges:
                        edges_to_remove.append((v, u))

    for u, v in edges_to_remove:
        if (u, v) in graph.edges:
            graph.remove_edge(u, v)

    for node in nodes_to_add:
        graph.add_node(node[0], **node[1])

    for edge in edges_to_add:
        graph.add_edge(edge[0], edge[1], **edge[2])

    return graph

@register_graphfcn
def calculate_contributing_area(G: nx.Graph, 
                         subcatchment_derivation: parameters.SubcatchmentDerivation,
                         addresses: parameters.Addresses,
                         **kwargs):
    """Calculate the contributing area for each edge.
    
    This function calculates the contributing area for each edge. The
    contributing area is the area of the subcatchment that drains to the
    edge. The contributing area is calculated from the elevation data.

    Also writes the file 'subcatchments.geojson' to addresses.subcatchments.

    Requires a graph with edges that have:
        - 'geometry' (shapely LineString)
        - 'id' (str)
        - 'width' (float)

    Adds the edge attributes:
        - 'contributing_area' (float)

    Args:
        G (nx.Graph): A graph
        subcatchment_derivation (parameters.SubcatchmentDerivation): A
            SubcatchmentDerivation parameter object
        addresses (parameters.Addresses): An Addresses parameter object
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

    # RC
    buildings = gpd.read_file(addresses.building)
    subs_rc = go.derive_rc(subs_gdf, G, buildings)

    # Write subs
    # TODO - could just attach subs to nodes where each node has a list of subs
    subs_rc.to_file(addresses.subcatchments, driver='GeoJSON')

    # Assign contributing area
    imperv_lookup = subs_rc.set_index('id').impervious_area.to_dict()
    for u,v,d in G.edges(data=True):
        if u in imperv_lookup.keys():
            d['contributing_area'] = imperv_lookup[u]
        else:
            d['contributing_area'] = 0.0
    return G

def set_elevation(G: nx.Graph, 
                  addresses: parameters.Addresses,
                  **kwargs) -> nx.Graph:
    """Set the elevation for each node.

    This function sets the elevation for each node. The elevation is
    calculated from the elevation data.

    Requires a graph with nodes that have:
        - 'x' (float)
        - 'y' (float)

    Adds the node attributes:
        - 'elevation' (float)

    Args:
        G (nx.Graph): A graph
        addresses (parameters.Addresses): An Addresses parameter object
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
    nx.set_node_attributes(G, elevations_dict, 'elevation')
    return G
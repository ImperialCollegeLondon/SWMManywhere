# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
import json
import tempfile
from heapq import heappop, heappush
from pathlib import Path
from typing import Callable, Hashable

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely import geometry as sgeom
from tqdm import tqdm

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
    #TODO the geometry is left as is currently - should be reveresed, however
    # in original osmnx geometry there are some incorrectly directed ones
    # someone with more patience might check start and end Points to check
    # which direction the line should be going in...
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

    Adds the node attributes:
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
    nx.set_node_attributes(G, imperv_lookup, 'contributing_area')
    for u, d in G.nodes(data=True):
        if 'contributing_area' not in d.keys():
            d['contributing_area'] = 0.0
    for u,v,d in G.edges(data=True):
        if u in imperv_lookup.keys():
            d['contributing_area'] = imperv_lookup[u]
        else:
            d['contributing_area'] = 0.0
    return G

@register_graphfcn
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

@register_graphfcn
def set_surface_slope(G: nx.Graph,
                      **kwargs) -> nx.Graph:
    """Set the surface slope for each edge.

    This function sets the surface slope for each edge. The surface slope is
    calculated from the elevation data.

    Requires a graph with nodes that have:
        - 'elevation' (float)
    
    Adds the edge attributes:
        - 'surface_slope' (float)

    Args:
        G (nx.Graph): A graph
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): A graph
    """
    G = G.copy()
    for u,v,d in G.edges(data=True):
        slope = (G.nodes[u]['elevation'] - G.nodes[v]['elevation']) / d['length']
        d['surface_slope'] = slope
    return G

@register_graphfcn
def set_chahinan_angle(G: nx.Graph, 
                       **kwargs) -> nx.Graph:
    """Set the Chahinan angle for each edge.

    This function sets the Chahinan angle for each edge. The Chahinan angle is
    calculated from the geometry of the edge and weighted according to the 
    angle (based on: https://doi.org/10.1016/j.compenvurbsys.2019.101370)

    Requires a graph with nodes that have:
        - 'x' (float)
        - 'y' (float)

    Adds the edge attributes:
        - 'chahinan_angle' (float)

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
            if node != u:
                p1 = (G.nodes[u]['x'], G.nodes[u]['y'])
                p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
                p3 = (G.nodes[node]['x'], G.nodes[node]['y'])
                angle = go.calculate_angle(p1,p2,p3)
                chahinan_weight = np.interp(angle,
                                            [0, 90, 135, 180, 225, 270, 360],
                                            [1, 0.2, 0.7, 0, 0.7, 0.2, 1]
                                            )
                min_weight = min(chahinan_weight, min_weight)
        if min_weight == float('inf'):
            min_weight = 0
        d['chahinan_angle'] = min_weight
    return G

@register_graphfcn
def calculate_weights(G: nx.Graph, 
                       topo_derivation: parameters.TopologyDerivation,
                       **kwargs) -> nx.Graph:
    """Calculate the weights for each edge.

    This function calculates the weights for each edge. The weights are
    calculated from the edge attributes.

    Requires a graph with edges that have:
        - weights as defined in topo_derivation.weights
    
    Adds the edge attributes:
        - 'weight' (float)

    Args:
        G (nx.Graph): A graph
        topo_derivation (parameters.TopologyDerivation): A TopologyDerivation
            parameter object
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): A graph
    """
    # Calculate bounds to normalise between
    bounds = {}
    for weight in topo_derivation.weights:
        bounds[weight] = [np.Inf, -np.Inf]

    for u, v, d in G.edges(data=True):
        for attr, bds in bounds.items():
            bds[0] = min(bds[0], d[attr])
            bds[1] = max(bds[1], d[attr])
    
    G = G.copy()
    for u, v, d in G.edges(data=True):
        total_weight = 0
        for attr, bds in bounds.items():
            # Normalise
            weight = (d[attr] - bds[0]) / (bds[1] - bds[0])
            # Exponent
            weight = weight ** getattr(topo_derivation,f'{attr}_exponent')
            # Scaling
            weight = weight * getattr(topo_derivation,f'{attr}_scaling')
            # Sum
            total_weight += weight
        # Set
        d['weight'] = total_weight
    return G

@register_graphfcn
def identify_outlets(G: nx.Graph,
                     outlet_derivation: parameters.OutletDerivation,
                     **kwargs) -> nx.Graph:
    """Identify outlets in a combined river-street graph.

    This function identifies outlets in a combined river-street graph. An
    outlet is a node that is connected to a river and a street. 

    Requires a graph with edges that have:
        - 'edge_type' ('river' or 'street')
        - 'length' (float)
    
    Requires a graph with nodes that have:
        - 'x' (float)
        - 'y' (float)

    Adds new edges to represent outlets with the attributes:
        - 'edge_type' ('outlet')
        - 'length' (float)
        - 'id' (str)

    Args:
        G (nx.Graph): A graph
        outlet_derivation (parameters.OutletDerivation): An OutletDerivation
            parameter object
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): A graph
    """
    G = G.copy()
    river_points = {}
    street_points = {}

    # Get the points for each river and street node
    for u, v, d in G.edges(data=True):
        upoint = sgeom.Point(G.nodes[u]['x'], G.nodes[u]['y'])
        vpoint = sgeom.Point(G.nodes[v]['x'], G.nodes[v]['y'])
        if d['edge_type'] == 'river':
            river_points[u] = upoint
            river_points[v] = vpoint
        else:
            street_points[u] = upoint
            street_points[v] = vpoint
    
    # Pair up the river and street nodes
    matched_outlets = go.nearest_node_buffer(river_points,
                                            street_points,
                                            outlet_derivation.river_buffer_distance)
    
    # Copy graph to run shortest path on
    G_ = G.copy()

    # Add edges between the paired river and street nodes
    for river_id, street_id in matched_outlets.items():
        # TODO instead use a weight based on the distance between the two nodes
        G_.add_edge(street_id, river_id,
                    **{'length' : outlet_derivation.outlet_length,
                       'edge_type' : 'outlet',
                       'id' : f'{street_id}-{river_id}-outlet'})
    
    # Add edges from the river nodes to a waste node
    for river_node in river_points.keys():
        if G.out_degree(river_node) == 0:
            G_.add_edge(river_node,
                        'waste',
                        **{'length' : 0,
                           'edge_type' : 'outlet',
                           'id' : f'{river_node}-waste-outlet'})
    
    # Set the length of the river edges to 0 - from a design perspective 
    # once water is in the river we don't care about the length - since it 
    # costs nothing
    for u,v,d in G_.edges(data=True):
        if d['edge_type'] == 'river':
            d['length'] = 0
    
    # Find shortest path to identify only 'efficient' outlets
    # TODO convert to shortest path when the speedy mechanism is implemented
    G_ = nx.minimum_spanning_tree(G_.to_undirected(),
                                       weight = 'length')
    
    # Retain the shortest path outlets in the original graph
    for u,v,d in G_.edges(data=True):
        if (d['edge_type'] == 'outlet') & (v != 'waste'):
            G.add_edge(u,v,**d)

    return G

@register_graphfcn
def derive_topology(G: nx.Graph,
                    **kwargs) -> nx.Graph:
    """Derive the topology of a graph.

    Runs a djiikstra-based algorithm to identify the shortest path from each
    node to its nearest outlet (weighted by the 'weight' edge value). The 
    returned graph is one that only contains the edges that feature  on the 
    shortest paths.

    Requires a graph with edges that have:
        - 'edge_type' ('river' or 'street')
        - 'weight' (float)
    
    Adds the node attributes:
        - 'outlet' (str)
        - 'shortest_path' (float)

    Args:
        G (nx.Graph): A graph
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): A graph
    """
    G = G.copy()
    
    # Identify outlets
    outlets = [u for u,v,d in G.edges(data=True) if d['edge_type'] == 'outlet']

    # Remove non-street edges/nodes and unconnected nodes
    nodes_to_remove = []
    for u, v, d in G.edges(data=True):
        if d['edge_type'] != 'street':
            if d['edge_type'] == 'outlet':
                nodes_to_remove.append(v)
            else:
                nodes_to_remove.append(u)
                nodes_to_remove.append(v)

    isolated_nodes = list(nx.isolates(G))

    for u in set(nodes_to_remove).union(isolated_nodes):
        G.remove_node(u)

    # Initialize the dictionary with infinity for all nodes
    shortest_paths = {node: float('inf') for node in G.nodes}

    # Initialize the dictionary to store the paths
    paths: dict[Hashable,list] = {node: [] for node in G.nodes}

    # Set the shortest path length to 0 for outlets
    for outlet in outlets:
        shortest_paths[outlet] = 0
        paths[outlet] = [outlet]

    # Initialize a min-heap with (distance, node) tuples
    heap = [(0, outlet) for outlet in outlets]
    while heap:
        # Pop the node with the smallest distance
        dist, node = heappop(heap)

        # For each neighbor of the current node
        for neighbor, edge_data in G[node].items():
            # Calculate the distance through the current node
            alt_dist = dist + edge_data[0]['weight']
            # If the alternative distance is shorter

            if alt_dist < shortest_paths[neighbor]:
                # Update the shortest path length
                shortest_paths[neighbor] = alt_dist
                # Update the path
                paths[neighbor] = paths[node] + [neighbor]
                # Push the neighbor to the heap
                heappush(heap, (alt_dist, neighbor))

    edges_to_keep = set()
    for path in paths.values():
        # Assign outlet
        outlet = path[0]
        for node in path:
            G.nodes[node]['outlet'] = outlet
            G.nodes[node]['shortest_path'] = shortest_paths[node]

        # Store path
        for i in range(len(path) - 1):
            edges_to_keep.add((path[i+1], path[i]))

    # Remvoe edges not on paths
    new_graph = G.copy()
    for u,v in G.edges():
        if (u,v) not in edges_to_keep:
            new_graph.remove_edge(u,v)
    return new_graph

@register_graphfcn
def pipe_by_pipe(G: nx.Graph, 
                 pipe_design: parameters.HydraulicDesign,
                 **kwargs
                 ):
    """Pipe by pipe hydraulic design.

    Starting from the most upstream node, design a pipe to the downstream node
    specifying a diameter and downstream invert level. A range of diameters and
    invert levels are tested (ranging between conditions defined in 
    pipe_design). From the tested diameters/inverts, a selection is made based
    on each pipe's satisfying feasibility constraints on: surcharge velocity,
    filling ratio, (and shear stress - not currently implemented). Prioritising 
    feasibility in this order it identifies pipes with the preferred feasibility
    level. If multiple pipes are feasible, it picks the lowest cost pipe. Once
    the feasible pipe is identified, the diameter and downstream invert are set
    and then the next downstream pipe can be designed. 

    This approach is based on the pipe-by-pipe design proposed in:
        https://doi.org/10.1016/j.watres.2021.117903
    
    Requires a graph with edges that have:
        - 'length' (float)
        - 'elevation' (float)
    
    Requires a graph with nodes that have:
        - 'contributing_area' (float)
        - 'elevation' (float)
    
    Adds the edge attributes:
        - 'diameter' (float)

    Adds the node attributes:
        - 'chamber_floor_elevation' (float)

    Args:
        G (nx.Graph): A graph
        pipe_design (parameters.HydraulicDesign): A HydraulicDesign parameter
            object
        **kwargs: Additional keyword arguments are ignored.

    Returns:
        G (nx.Graph): A graph
    """
    # TODO obviously needs a refactor and better testing

    G = G.copy()
    surface_elevations = {n : d['elevation'] for n, d in G.nodes(data=True)}
    topological_order = list(nx.topological_sort(G))
    chamber_floor = {}
    edge_diams = {}
    # Iterate over nodes in topological order
    for node in tqdm(topological_order):
        # Check if there's any nodes upstream, if not set the depth to min_depth
        if len(nx.ancestors(G,node)) == 0:
            chamber_floor[node] = surface_elevations[node] - pipe_design.min_depth
        for ix, ds_node in enumerate(G.successors(node)):
            edge = G.get_edge_data(node,ds_node,0)
            # Find contributing area with ancestors
            # TODO - could do timearea here if i hated myself enough
            anc = nx.ancestors(G,node).union([node])
            tot = sum([G.nodes[anc_node]['contributing_area'] for anc_node in anc])
            
            M3_PER_HR_TO_M3_PER_S = 1 / 60 / 60
            Q = tot * pipe_design.precipitation * M3_PER_HR_TO_M3_PER_S
            
            # Within all allowable slopes and diams, calculate the parameters
            # of all feasible pipes
            feasible_pipes = []
            for diam in pipe_design.diameters:
                A = (np.pi * diam ** 2 / 4)
                n = 0.012 # mannings n
                R = A / (np.pi * diam)  # hydraulic radius
                for depth in np.linspace(pipe_design.min_depth, 
                                         pipe_design.max_depth, 
                                         10):
                    # TODO... presumably need to check depth > (diam + min_depth)

                    elev_diff = chamber_floor[node] - \
                        (surface_elevations[ds_node] - depth)
                    slope = elev_diff / edge['length']
                    # Always pick a pipe that is feasible without surcharging 
                    # if available
                    surcharge_feasibility = 0.0
                    # Use surcharged elevation                        
                    while slope <= 0:
                        surcharge_feasibility += 0.05
                        slope = (chamber_floor[node] + surcharge_feasibility - \
                                 (surface_elevations[ds_node] - depth)) / edge['length']
                        # TODO could make the feasibility penalisation increase
                        # when you get above surface_elevation[node]... but 
                        # then you'd need a feasibility tracker and an offset 
                        # tracker                    
                    v = (slope ** 0.5) * (R ** (2/3)) / n
                    filling_ratio = Q / (v * A)
                    # buffers from: https://www.polypipe.com/sites/default/files/Specification_Clauses_Underground_Drainage.pdf 
                    average_depth = (depth + chamber_floor[node]) / 2
                    V = edge['length'] * (diam + 0.3) * (average_depth + 0.1)
                    cost = 1.32 / 2000 * (9579.31 * diam ** 0.5737 + 1153.77 * V**1.31)
                    v_feasibility = max(pipe_design.min_v - v, 0) + \
                        max(v - pipe_design.max_v, 0)
                    fr_feasibility = max(filling_ratio - pipe_design.max_fr, 0)
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
                    slope = (chamber_floor[node] - (surface_elevations[ds_node] -\
                                                     depth)) / edge['length']
                    feasible_pipes.append({'diam' : diam,
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
            feasible_pipes_df = pd.DataFrame(feasible_pipes).dropna()
            if feasible_pipes_df.shape[0] > 0:
                ideal_pipe = feasible_pipes_df.sort_values(by=['surcharge_feasibility',
                                                            'v_feasibility',
                                                            'fr_feasibility',
                                                            # 'shear_feasibility',
                                                            'cost'], 
                                                        ascending = True).iloc[0]
                edge_diams[(node,ds_node,0)] = ideal_pipe.diam
                chamber_floor[ds_node] = surface_elevations[ds_node] - ideal_pipe.depth
            else:
                print('something odd - no non nan pipes')
            if ix > 0:
                print('''a node has multiple successors, 
                      not sure how that can happen if using shortest path
                      to derive topology''')
    nx.function.set_edge_attributes(G, edge_diams, "diameter")
    nx.function.set_node_attributes(G, chamber_floor, "chamber_floor_elevation")
    return G
"""Graph utilities module for SWMManywhere.

A module to contain graphfcns, the graphfcn registry object, and other graph
utilities (such as save/load functions).
"""
import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from heapq import heappop, heappush
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from tqdm import tqdm

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import parameters
from swmmanywhere.logging import logger


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
                      addresses: parameters.FilePaths) -> nx.Graph:
    """Iterate a list of graph functions over a graph.

    Args:
        G (nx.Graph): The graph to iterate over.
        graphfcn_list (list[str]): A list of graph functions to iterate.
        params (dict): A dictionary of parameters to pass to the graph
            functions.
        addresses (parameters.FilePaths): A FilePaths parameter object

    Returns:
        nx.Graph: The graph after the graph functions have been applied.
    """
    not_exists = [g for g in graphfcn_list if g not in graphfcns]
    if not_exists:
        raise ValueError(f"Graphfcns are not registered:\n{', '.join(not_exists)}")
    verbose = os.getenv("SWMMANYWHERE_VERBOSE", "false").lower() == "true"
    for function in graphfcn_list:
        G = graphfcns[function](G, addresses = addresses, **params)
        logger.info(f"graphfcn: {function} completed.")
        if verbose:
            save_graph(G, addresses.model / f"{function}_graph.json")
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
                logger.warning(f"Duplicate edge ID: {data['id']}")
                edges_to_remove.append((u, v, key))
            edge_ids.add(data['id'])
        for u, v, key in edges_to_remove:
            G.remove_edge(u, v, key)
        return G

@register_graphfcn
class format_osmnx_lanes(BaseGraphFunction,
                         required_edge_attributes = ['lanes'],
                         adds_edge_attributes = ['width']):
    """format_osmnx_lanes class."""
    # i.e., in osmnx format, i.e., empty for single lane, an int for a
    # number of lanes or a list if the edge has multiple carriageways

    def __call__(self,
                 G: nx.Graph, 
                       subcatchment_derivation: parameters.SubcatchmentDerivation, 
                       **kwargs) -> nx.Graph:
        """Format the lanes attribute of each edge and calculates width.

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
            data['width'] = lanes * subcatchment_derivation.lane_width
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
        #TODO the geometry is left as is currently - should be reversed, however
        # in original osmnx geometry there are some incorrectly directed ones
        # someone with more patience might check start and end Points to check
        # which direction the line should be going in...
        G_new = G.copy()
        for u, v, data in G.edges(data=True):
            include = data.get('edge_type', True)
            if isinstance(include, str):
                include = include == 'street'
            if ((v, u) not in G.edges) & include:
                reverse_data = data.copy()
                reverse_data['id'] = f"{data['id']}.reversed"
                G_new.add_edge(v, u, **reverse_data)
        return G_new

@register_graphfcn
class split_long_edges(BaseGraphFunction,
                       required_edge_attributes = ['id', 'geometry', 'length']):
    """split_long_edges class."""
    
    def __call__(self, 
                 G: nx.Graph, 
                 subcatchment_derivation: parameters.SubcatchmentDerivation, 
                 **kwargs) -> nx.Graph:
        """Split long edges into shorter edges.

        This function splits long edges into shorter edges. The edges are split
        into segments of length 'max_street_length'. The first and last segment
        are connected to the original nodes. Intermediate segments are connected
        to newly created nodes. The 'geometry' of the original edge must be
        a LineString.
        
        Args:
            G (nx.Graph): A graph
            subcatchment_derivation (parameters.SubcatchmentDerivation): A
                SubcatchmentDerivation parameter object
            **kwargs: Additional keyword arguments are ignored.
        
        Returns:
            graph (nx.Graph): A graph
        """
        #TODO refactor obviously
        max_length = subcatchment_derivation.max_street_length
        graph = G.copy()
        edges_to_remove = []
        edges_to_add = []
        nodes_to_add = []
        maxlabel = max(graph.nodes) + 1
        ll = 0

        def create_new_edge_data(line, data, id_):
            new_line = shapely.LineString(line)
            new_data = data.copy()
            new_data['id'] = id_
            new_data['length'] = new_line.length
            new_data['geometry'] =  shapely.LineString([(x[0], x[1]) 
                                                    for x in new_line.coords])
            return new_data

        for u, v, data in graph.edges(data=True):
            line = data['geometry']
            length = data['length']
            if ((u, v) not in edges_to_remove) & ((v, u) not in edges_to_remove):
                if length > max_length:
                    new_points = [shapely.Point(x) 
                                for x in ox.utils_geo.interpolate_points(line, 
                                                                        max_length)]
                    if len(new_points) > 2:
                        for ix, (start, end) in enumerate(zip(new_points[:-1], 
                                                            new_points[1:])):
                            new_data = create_new_edge_data([start, 
                                                            end], 
                                                            data, 
                                                            f"{data['id']}.{ix}")
                            if (v,u) in graph.edges:
                                # Create reversed data
                                data_r = graph.get_edge_data(v, u).copy()[0]
                                id_ = f"{data_r['id']}.{ix}"
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
            start_point_node = (G.nodes[u]['x'], G.nodes[u]['y'])
            start_point_edge = data['geometry'].coords[0]
            end_point_node = (G.nodes[v]['x'], G.nodes[v]['y'])
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
class calculate_contributing_area(BaseGraphFunction,
                                required_edge_attributes = ['id', 'geometry', 'width'],
                                adds_edge_attributes = ['contributing_area'],
                                adds_node_attributes = ['contributing_area']):
    """calculate_contributing_area class."""
    
    def __call__(self, G: nx.Graph, 
                         subcatchment_derivation: parameters.SubcatchmentDerivation,
                         addresses: parameters.FilePaths,
                         **kwargs) -> nx.Graph:
        """Calculate the contributing area for each edge.
        
        This function calculates the contributing area for each edge. The
        contributing area is the area of the subcatchment that drains to the
        edge. The contributing area is calculated from the elevation data.

        Also writes the file 'subcatchments.geojson' to addresses.subcatchments.

        Args:
            G (nx.Graph): A graph
            subcatchment_derivation (parameters.SubcatchmentDerivation): A
                SubcatchmentDerivation parameter object
            addresses (parameters.FilePaths): An FilePaths parameter object
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

        # Calculate runoff coefficient (RC)
        if addresses.building.suffix in ('.geoparquet','.parquet'):
            buildings = gpd.read_parquet(addresses.building)
        else:
            buildings = gpd.read_file(addresses.building)
        subs_rc = go.derive_rc(subs_gdf, G, buildings)

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
                  addresses: parameters.FilePaths,
                  **kwargs) -> nx.Graph:
        """Set the elevation for each node.

        This function sets the elevation for each node. The elevation is
        calculated from the elevation data.

        Args:
            G (nx.Graph): A graph
            addresses (parameters.FilePaths): An FilePaths parameter object
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

        # Values where the weight of the angle can be matched to the values 
        # in weights
        angle_points = [-1, 0.3, 0.7, 10] 
        weights = [1, 0, 0, 1]

        # Calculate weights
        slope = nx.get_edge_attributes(G, "surface_slope")
        weights = np.interp(np.asarray(list(slope.values())) * 100, 
                            angle_points,
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

        for (u, v, d), w in product(G.edges(data=True), 
                                    topology_derivation.weights):
            bounds[w][0] = min(bounds[w][0], d.get(w, np.Inf))
            bounds[w][1] = max(bounds[w][1], d.get(w, -np.Inf))

        G = G.copy()
        for u, v, d in G.edges(data=True):
            total_weight = 0
            for attr, bds in bounds.items():
                # Normalise
                weight = (d[attr] - bds[0]) / (bds[1] - bds[0])
                # Exponent
                weight = weight ** getattr(topology_derivation,f'{attr}_exponent')
                # Scaling
                weight = weight * getattr(topology_derivation,f'{attr}_scaling')
                # Sum
                total_weight += weight
            # Set
            d['weight'] = total_weight
        return G

@register_graphfcn
class identify_outlets(BaseGraphFunction,
                       required_edge_attributes = ['length', 'edge_type'],
                       required_node_attributes = ['x', 'y']):
    """identify_outlets class."""

    def __call__(self, G: nx.Graph,
                     outlet_derivation: parameters.OutletDerivation,
                     **kwargs) -> nx.Graph:
        """Identify outlets in a combined river-street graph.

        This function identifies outlets in a combined river-street graph. An
        outlet is a node that is connected to a river and a street. 

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
            upoint = shapely.Point(G.nodes[u]['x'], G.nodes[u]['y'])
            vpoint = shapely.Point(G.nodes[v]['x'], G.nodes[v]['y'])
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
                        'geometry' : shapely.LineString([street_points[street_id],
                                                    river_points[river_id]]),
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
        
        # Find shortest path to identify only 'efficient' outlets. The MST
        # makes sense here over shortest path as each node is only allowed to
        # be visited once - thus encouraging fewer outlets. In shortest path
        # nodes near rivers will always just pick their nearest river node.
        T = nx.minimum_spanning_tree(G_.to_undirected(),
                                        weight = 'length')
        
        # Retain the shortest path outlets in the original graph
        for u,v,d in T.edges(data=True):
            if (d['edge_type'] == 'outlet') & (v != 'waste'):
                # Need to check both directions since T is undirected
                if (u,v) in G_.edges():
                    G.add_edge(u,v,**d)
                elif (v,u) in G_.edges():
                    G.add_edge(v,u,**d)

        return G

@register_graphfcn
class derive_topology(BaseGraphFunction,
                      required_edge_attributes = ['edge_type', # 'rivers' and 'streets'
                                                  'weight'],
                      adds_node_attributes = ['outlet', 'shortest_path']):
    """derive_topology class."""
    

    def __call__(self, G: nx.Graph,
                    **kwargs) -> nx.Graph:
        """Derive the topology of a graph.

        Runs a djiikstra-based algorithm to identify the shortest path from each
        node to its nearest outlet (weighted by the 'weight' edge value). The 
        returned graph is one that only contains the edges that feature  on the 
        shortest paths.

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
                    nodes_to_remove.extend((u,v))

        isolated_nodes = list(nx.isolates(G))

        for u in set(nodes_to_remove).union(isolated_nodes):
            G.remove_node(u)

        # Check for negative cycles
        if nx.negative_edge_cycle(G, weight = 'weight'):
            logger.warning('Graph contains negative cycle')

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
            for neighbor, _, edge_data in G.in_edges(node, data=True):
                # Calculate the distance through the current node
                alt_dist = dist + edge_data['weight']
                # If the alternative distance is shorter

                if alt_dist >= shortest_paths[neighbor]:
                    continue
                
                # Update the shortest path length
                shortest_paths[neighbor] = alt_dist
                # Update the path
                paths[neighbor] = paths[node] + [neighbor]
                # Push the neighbor to the heap
                heappush(heap, (alt_dist, neighbor))

        edges_to_keep: set = set()
        for path in paths.values():
            # Assign outlet
            outlet = path[0]
            for node in path:
                G.nodes[node]['outlet'] = outlet
                G.nodes[node]['shortest_path'] = shortest_paths[node]

            # Store path
            edges_to_keep.update(zip(path[1:], path[:-1]))
            
        # Remove edges not on paths
        new_graph = G.copy()
        for u,v in G.edges():
            if (u,v) not in edges_to_keep:
                new_graph.remove_edge(u,v)
        return new_graph

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
        for node in tqdm(topological_order):
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
# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

from typing import Callable

import networkx as nx

from swmmanywhere import parameters

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
    
    Adds the attributes:
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
    
    Adds the attributes:
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

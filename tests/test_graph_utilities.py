# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""


from swmmanywhere import geospatial_utilities as go
from swmmanywhere import graph_utilities as gu
from swmmanywhere import parameters
from swmmanywhere.prepare_data import download_street


def generate_street_graph():
    """Generate a street graph."""
    bbox = (-0.11643,51.50309,-0.11169,51.50549)
    G = download_street(bbox)
    return G

def test_assign_id():
    """Test the assign_id function."""
    G = generate_street_graph()
    G = gu.assign_id(G)
    for u, v, data in G.edges(data=True):
        assert 'id' in data.keys()
        assert isinstance(data['id'], int)

def test_double_directed():
    """Test the double_directed function."""
    G = generate_street_graph()
    G = gu.assign_id(G)
    G = gu.double_directed(G)
    for u, v in G.edges():
        assert (v,u) in G.edges

def test_format_osmnx_lanes():
    """Test the format_osmnx_lanes function."""
    G = generate_street_graph()
    params = parameters.SubcatchmentDerivation()
    G = gu.format_osmnx_lanes(G, params)
    for u, v, data in G.edges(data=True):
        assert 'lanes' in data.keys()
        assert isinstance(data['lanes'], float)
        assert 'width' in data.keys()
        assert isinstance(data['width'], float)

def test_split_long_edges():
    """Test the split_long_edges function."""
    G = generate_street_graph()
    G = gu.assign_id(G)
    id_ = list(G.nodes)[0]
    G = go.reproject_graph(G, 
                           'EPSG:4326', 
                           go.get_utm_epsg(G.nodes[id_]['x'], 
                                           G.nodes[id_]['y'])
                            )
    max_length = 20
    params = parameters.SubcatchmentDerivation(max_street_length = max_length)
    G = gu.split_long_edges(G, params)
    for u, v, data in G.edges(data=True):
        assert data['length'] <= (max_length * 2)
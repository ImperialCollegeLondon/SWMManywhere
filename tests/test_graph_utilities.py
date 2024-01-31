# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
import tempfile
from pathlib import Path

import geopandas as gpd
from shapely import geometry as sgeom

from swmmanywhere import graph_utilities as gu
from swmmanywhere import parameters


def load_street_network():
    """Load a street network."""
    bbox = (-0.11643,51.50309,-0.11169,51.50549)
    G = gu.load_graph(Path(__file__).parent / 'test_data' / 'street_graph.json')
    return G, bbox

def test_assign_id():
    """Test the assign_id function."""
    G, _ = load_street_network()
    G = gu.assign_id(G)
    for u, v, data in G.edges(data=True):
        assert 'id' in data.keys()
        assert isinstance(data['id'], int)

def test_double_directed():
    """Test the double_directed function."""
    G, _ = load_street_network()
    G = gu.assign_id(G)
    G = gu.double_directed(G)
    for u, v in G.edges():
        assert (v,u) in G.edges

def test_format_osmnx_lanes():
    """Test the format_osmnx_lanes function."""
    G, _ = load_street_network()
    params = parameters.SubcatchmentDerivation()
    G = gu.format_osmnx_lanes(G, params)
    for u, v, data in G.edges(data=True):
        assert 'lanes' in data.keys()
        assert isinstance(data['lanes'], float)
        assert 'width' in data.keys()
        assert isinstance(data['width'], float)

def test_split_long_edges():
    """Test the split_long_edges function."""
    G, _ = load_street_network()
    G = gu.assign_id(G)
    max_length = 20
    params = parameters.SubcatchmentDerivation(max_street_length = max_length)
    G = gu.split_long_edges(G, params)
    for u, v, data in G.edges(data=True):
        assert data['length'] <= (max_length * 2)

def test_derive_subcatchments():
    """Test the derive_subcatchments function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        addresses = parameters.Addresses(base_dir = temp_path, 
                            project_name = 'test', 
                            bbox_number = 1,
                            extension = 'json',
                            model_number = 1)
        addresses.elevation = Path(__file__).parent / 'test_data' / 'elevation.tif'
        addresses.building = temp_path / 'building.geojson'
        addresses.subcatchments = temp_path / 'subcatchments.geojson'
        params = parameters.SubcatchmentDerivation()
        G, bbox = load_street_network()
        
        # mock up buildings
        eg_bldg = sgeom.Polygon([(700291.346,5709928.922), 
                       (700331.206,5709927.815),
                       (700321.610,5709896.444), 
                       (700293.192,5709900.503),
                       (700291.346,5709928.922)])
        gdf = gpd.GeoDataFrame(geometry = [eg_bldg],
                               crs = G.graph['crs'])
        gdf.to_file(addresses.building, driver='GeoJSON')

        G = gu.calculate_contributing_area(G, params, addresses)
        for u, v, data in G.edges(data=True):
            assert 'contributing_area' in data.keys()
            assert isinstance(data['contributing_area'], float)
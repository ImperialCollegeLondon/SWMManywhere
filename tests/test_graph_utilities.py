# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
import math
import tempfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely import geometry as sgeom

from swmmanywhere import parameters
from swmmanywhere.graph_utilities import graphfcns as gu
from swmmanywhere.graph_utilities import iterate_graphfcns, load_graph, save_graph


def load_street_network():
    """Load a street network."""
    bbox = (-0.11643,51.50309,-0.11169,51.50549)
    G = load_graph(Path(__file__).parent / 'test_data' / 'street_graph.json')
    return G, bbox

def test_save_load():
    """Test the save_graph and load_graph functions."""
    # Load a street network
    G,_ = load_street_network()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the graph
        save_graph(G, Path(temp_dir) / 'test_graph.json')
        # Load the graph
        G_new = load_graph(Path(temp_dir) / 'test_graph.json')
        # Check if the loaded graph is the same as the original graph
        assert nx.is_isomorphic(G, G_new)

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
        addresses = parameters.FilePaths(base_dir = temp_path, 
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

        for u, data in G.nodes(data=True):
            assert 'contributing_area' in data.keys()
            assert isinstance(data['contributing_area'], float)

def test_set_elevation_and_slope():
    """Test the set_elevation, set_surface_slope, chahinian_slope function."""
    G, _ = load_street_network()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        addresses = parameters.FilePaths(base_dir = temp_path, 
                                project_name = 'test', 
                                bbox_number = 1,
                                extension = 'json',
                                model_number = 1)
        addresses.elevation = Path(__file__).parent / 'test_data' / 'elevation.tif'
        G = gu.set_elevation(G, addresses)
        for id_, data in G.nodes(data=True):
            assert 'surface_elevation' in data.keys()
            assert math.isfinite(data['surface_elevation'])
            assert data['surface_elevation'] > 0
        
        G = gu.set_surface_slope(G)
        for u, v, data in G.edges(data=True):
            assert 'surface_slope' in data.keys()
            assert math.isfinite(data['surface_slope'])

        
        G = gu.set_chahinian_slope(G)
        for u, v, data in G.edges(data=True):
            assert 'chahinian_slope' in data.keys()
            assert math.isfinite(data['chahinian_slope'])
        
        slope_vals = {-2 : 1,
                      0.3 : 0,
                      0.4 : 0,
                      12 : 1}
        for slope, expected in slope_vals.items():
            first_edge = list(G.edges)[0]
            G.edges[first_edge]['surface_slope'] = slope / 100
            G = gu.set_chahinian_slope(G)
            assert G.edges[first_edge]['chahinian_slope'] == expected



def test_chahinian_angle():
    """Test the chahinian_angle function."""
    G, _ = load_street_network()
    G = gu.set_chahinian_angle(G)
    for u, v, data in G.edges(data=True):
        assert 'chahinian_angle' in data.keys()
        assert math.isfinite(data['chahinian_angle'])



def test_calculate_weights():
    """Test the calculate_weights function."""
    G, _ = load_street_network()
    params = parameters.TopologyDerivation()
    for weight in params.weights:
        for ix, (u,v,data) in enumerate(G.edges(data=True)):
            data[weight] = ix

    G = gu.calculate_weights(G, params)
    for u, v, data in G.edges(data=True):
        assert 'weight' in data.keys()
        assert math.isfinite(data['weight'])

def test_identify_outlets_and_derive_topology():
    """Test the identify_outlets and derive_topology functions."""
    G, _ = load_street_network()
    G = gu.assign_id(G)
    G = gu.double_directed(G)
    for ix, (u,v,d) in enumerate(G.edges(data=True)):
        d['edge_type'] = 'street'
        d['weight'] = ix

    params = parameters.OutletDerivation(river_buffer_distance = 300)
    dummy_river1 = sgeom.LineString([(699913.878,5709769.851), 
                                    (699932.546,5709882.575)])
    dummy_river2 = sgeom.LineString([(699932.546,5709882.575),    
                                    (700011.524,5710060.636)])
    dummy_river3 = sgeom.LineString([(700011.524,5710060.636),
                                    (700103.427,5710169.052)])
    
    G.add_edge('river1', 'river2', **{'length' :  10,
                                    'edge_type' : 'river',
                                    'id' : 'river1-to-river2',
                                    'geometry' :  dummy_river1})
    G.add_edge('river2', 'river3', **{'length' :  10,
                                    'edge_type' : 'river',
                                    'id' : 'river2-to-river3',
                                    'geometry' :  dummy_river2})
    
    G.add_edge('river3', 'river4', **{'length' :  10,
                                    'edge_type' : 'river',
                                    'id' : 'river3-to-river4',
                                    'geometry' :  dummy_river3})
    
    G.nodes['river1']['x'] = 699913.878
    G.nodes['river1']['y'] = 5709769.851
    G.nodes['river2']['x'] = 699932.546
    G.nodes['river2']['y'] = 5709882.575
    G.nodes['river3']['x'] = 700011.524
    G.nodes['river3']['y'] = 5710060.636
    G.nodes['river4']['x'] = 700103.427
    G.nodes['river4']['y'] = 5710169.052

    # Test outlet derivation
    G_ = G.copy()
    G_ = gu.identify_outlets(G_, params)

    outlets = [(u,v,d) for u,v,d in G_.edges(data=True) if d['edge_type'] == 'outlet']
    assert len(outlets) == 2
    
    # Test topo derivation
    G_ = gu.derive_topology(G_)
    assert len(G_.edges) == 22

    # Test outlet derivation parameters
    G_ = G.copy()
    params.outlet_length = 600
    G_ = gu.identify_outlets(G_, params)
    outlets = [(u,v,d) for u,v,d in G_.edges(data=True) if d['edge_type'] == 'outlet']
    assert len(outlets) == 1

def test_pipe_by_pipe():
    """Test the pipe_by_pipe function."""
    G = load_graph(Path(__file__).parent / 'test_data' / 'graph_topo_derived.json')
    for ix, (u,d) in enumerate(G.nodes(data=True)):
        d['surface_elevation'] = ix
        d['contributing_area'] = ix
    
    params = parameters.HydraulicDesign()

    G = gu.pipe_by_pipe(G, params)
    for u, v, d in G.edges(data=True):
        assert 'diameter' in d.keys()
        assert d['diameter'] in params.diameters
    
    for u, d in G.nodes(data=True):
        assert 'chamber_floor_elevation' in d.keys()
        assert math.isfinite(d['chamber_floor_elevation'])
        
def test_iterate_graphfcns():
    """Test the iterate_graphfcns function."""
    G = load_graph(Path(__file__).parent / 'test_data' / 'graph_topo_derived.json')
    params = parameters.get_full_parameters()
    addresses = parameters.FilePaths(base_dir = None,
                                    project_name = None,
                                    bbox_number = None,
                                    model_number = None)
    G = iterate_graphfcns(G, 
                             ['assign_id',
                              'format_osmnx_lanes'],
                              params, 
                              addresses)
    for u, v, d in G.edges(data=True):
        assert 'id' in d.keys()
        assert 'width' in d.keys()

# -*- coding: utf-8 -*-
"""Created on Tue Oct 18 10:35:51 2022.

@author: Barney
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio as rst
from scipy.interpolate import RegularGridInterpolator
from shapely import geometry as sgeom

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import graph_utilities as ge


def load_street_network():
    """Load a street network."""
    G = ge.load_graph(Path(__file__).parent / 'test_data' / 'street_graph.json')
    return G

def test_interp_with_nans():
    """Test the interp_interp_with_nans function."""
    # Define a simple grid and values
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    values = np.linspace(0, 1, 25)
    values_grid = values.reshape(5, 5)

    # Define an interpolator
    interp = RegularGridInterpolator((x,y), 
                                        values_grid)

    # Test the function at a point inside the grid
    yx = (0.875, 0.875)
    result = go.interp_with_nans(yx, interp, grid, values)
    assert result == 0.875

    # Test the function on a nan point
    values_grid[1][1] = np.nan
    yx = (0.251, 0.25) 
    result = go.interp_with_nans(yx, interp, grid, values)
    assert result == values_grid[1][2]

@patch('rasterio.open')
def test_interpolate_points_on_raster(mock_rst_open):
    """Test the interpolate_points_on_raster function."""
    # Mock the raster file
    mock_src = MagicMock()
    mock_src.read.return_value = np.array([[1, 2], [3, 4]])
    mock_src.bounds = MagicMock()
    mock_src.bounds.left = 0
    mock_src.bounds.right = 1
    mock_src.bounds.bottom = 0
    mock_src.bounds.top = 1
    mock_src.width = 2
    mock_src.height = 2
    mock_src.nodata = None
    mock_rst_open.return_value.__enter__.return_value = mock_src
    
    # Define the x and y coordinates
    x = [0.25, 0.75]
    y = [0.25, 0.75]

    # Call the function
    result = go.interpolate_points_on_raster(x, 
                                             y, 
                                             Path('fake_path'))

    # [2.75, 2.25] feels unintuitive but it's because rasters measure from the top
    assert result == [2.75, 2.25]

def test_get_utm():
    """Test the get_utm_epsg function."""
    # Test a northern hemisphere point
    crs = go.get_utm_epsg(-1.0, 51.0)
    assert crs == 'EPSG:32630'

    # Test a southern hemisphere point
    crs = go.get_utm_epsg(-1.0, -51.0)
    assert crs == 'EPSG:32730'

def create_raster(fid):
    """Define a function to create a mock raster file."""
    data = np.ones((100, 100))
    transform = rst.transform.from_origin(0, 0, 0.1, 0.1)
    with rst.open(fid, 
                  'w', 
                  driver='GTiff', 
                  height=100, 
                  width=100, 
                  count=1, 
                  dtype='uint8', 
                  crs='EPSG:4326', 
                  transform=transform) as src:
        src.write(data, 1)
def test_reproject_raster():
    """Test the reproject_raster function."""
    # Create a mock raster file
    fid = Path('test.tif')
    try:
        create_raster(fid)

        # Define the input parameters
        target_crs = 'EPSG:32630'
        new_fid = Path('test_reprojected.tif')

        # Call the function
        go.reproject_raster(target_crs, fid)

        # Check if the reprojected file exists
        assert new_fid.exists()

        # Check if the reprojected file has the correct CRS
        with rst.open(new_fid) as src:
            assert src.crs.to_string() == target_crs
    finally:
        # Regardless of test outcome, delete the temp file
        if fid.exists():
            fid.unlink()
        if new_fid.exists():
            new_fid.unlink()


def almost_equal(a, b, tol=1e-6):
    """Check if two numbers are almost equal."""
    return abs(a-b) < tol

def test_get_transformer():
    """Test the get_transformer function."""
    # Test a northern hemisphere point
    transformer = go.get_transformer('EPSG:4326', 'EPSG:32630')

    initial_point = (-0.1276, 51.5074)
    expected_point = (699330.1106898375, 5710164.30300683)
    new_point = transformer.transform(*initial_point)
    assert almost_equal(new_point[0],
                        expected_point[0])
    assert almost_equal(new_point[1],
                        expected_point[1])

def test_reproject_graph():
    """Test the reproject_graph function."""
    # Create a mock graph
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2)
    G.add_node(3, x=1, y=2)
    G.add_edge(2, 3, geometry=sgeom.LineString([(1, 1), (1, 2)]))

    # Define the input parameters
    source_crs = 'EPSG:4326'
    target_crs = 'EPSG:32630'

    # Call the function
    G_new = go.reproject_graph(G, source_crs, target_crs)

    # Test node coordinates
    assert almost_equal(G_new.nodes[1]['x'], 833978.5569194595)
    assert almost_equal(G_new.nodes[1]['y'], 0)
    assert almost_equal(G_new.nodes[2]['x'], 945396.6839773951)
    assert almost_equal(G_new.nodes[2]['y'], 110801.83254625657)
    assert almost_equal(G_new.nodes[3]['x'], 945193.8596723974)
    assert almost_equal(G_new.nodes[3]['y'], 221604.0105092727)

    # Test edge geometry
    assert almost_equal(list(G_new[1][2]['geometry'].coords)[0][0],
                        833978.5569194595)
    assert almost_equal(list(G_new[2][3]['geometry'].coords)[0][0],
                        945396.6839773951)

def test_nearest_node_buffer():
    """Test the nearest_node_buffer function."""
    # Create mock dictionaries of points
    points1 = {'a': sgeom.Point(0, 0), 'b': sgeom.Point(1, 1)}
    points2 = {'c': sgeom.Point(0.5, 0.5), 'd': sgeom.Point(2, 2)}

    # Define the input threshold
    threshold = 1.0

    # Call the function
    matching = go.nearest_node_buffer(points1, points2, threshold)

    # Check if the function returns the correct matching nodes
    assert matching == {'a': 'c', 'b': 'c'}

def test_burn_shape_in_raster():
    """Test the burn_shape_in_raster function."""
    # Create a mock geometry
    geoms = [sgeom.LineString([(0, 0), (1, 1)]),
             sgeom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]

    # Define the input parameters
    depth = 1.0
    raster_fid = Path('input.tif')
    new_raster_fid = Path('output.tif')
    try:
        create_raster(raster_fid)
        
        # Call the function
        go.burn_shape_in_raster(geoms, depth, raster_fid, new_raster_fid)

        with rst.open(raster_fid) as src:
            data_ = src.read(1)

        # Open the new GeoTIFF file and check if it has been correctly modified
        with rst.open(new_raster_fid) as src:
            data = src.read(1)
            assert (data != data_).any()
    finally:
        # Regardless of test outcome, delete the temp file
        if raster_fid.exists():
            raster_fid.unlink()
        if new_raster_fid.exists():
            new_raster_fid.unlink()
        
def test_derive_subcatchments():
    """Test the derive_subcatchments function."""
    G = load_street_network()
    elev_fid = Path(__file__).parent / 'test_data' / 'elevation.tif'
    
    polys = go.derive_subcatchments(G, elev_fid)
    assert 'slope' in polys.columns
    assert 'area' in polys.columns
    assert 'geometry' in polys.columns
    assert 'id' in polys.columns
    assert polys.shape[0] > 0
    assert polys.dropna().shape == polys.shape
    assert polys.crs == G.graph['crs']


def test_derive_rc():
    """Test the derive_rc function."""
    G = load_street_network()
    crs = G.graph['crs']
    eg_bldg = sgeom.Polygon([(700291,5709928), 
                       (700331,5709927),
                       (700321,5709896), 
                       (700293,5709900),
                       (700291,5709928)])
    buildings = gpd.GeoDataFrame(geometry = [eg_bldg],
                                 crs = crs)
    subs = [sgeom.Polygon([(700262, 5709928),
                            (700262, 5709883),
                            (700351, 5709883),
                            (700351, 5709906),
                            (700306, 5709906),
                            (700306, 5709928),
                            (700262, 5709928)]),
            sgeom.Polygon([(700306, 5709928),
                            (700284, 5709928),
                            (700284, 5709950),
                            (700374, 5709950),
                            (700374, 5709906),
                            (700351, 5709906),
                            (700306, 5709906),
                            (700306, 5709928)]),
            sgeom.Polygon([(700351, 5709883),
                            (700351, 5709906),
                            (700374, 5709906),
                            (700374, 5709883),
                            (700396, 5709883),
                            (700396, 5709816),
                            (700329, 5709816),
                            (700329, 5709838),
                            (700329, 5709883),
                            (700351, 5709883)])]

    subs = gpd.GeoDataFrame(data = {'id' : [107733,
                                            1696030874,
                                            6277683849]
                                            },
                                    geometry = subs,
                                    crs = crs)
    subs['area'] = subs.geometry.area

    subs_rc = go.derive_rc(subs, G, buildings).set_index('id')
    assert subs_rc.loc[6277683849,'impervious_area'] == 0
    assert subs_rc.loc[107733,'impervious_area'] > 0
    for u,v,d in G.edges(data=True):
        d['width'] = 10

    subs_rc = go.derive_rc(subs, G, buildings).set_index('id')
    assert subs_rc.loc[6277683849,'impervious_area'] > 0
    assert subs_rc.loc[6277683849,'rc'] > 0
    assert subs_rc.rc.max() <= 100

    for u,v,d in G.edges(data=True):
        d['width'] = 0

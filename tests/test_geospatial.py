# -*- coding: utf-8 -*-
"""Created on Tue Oct 18 10:35:51 2022.

@author: Barney
"""

# import pytest
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import rasterio as rst
from scipy.interpolate import RegularGridInterpolator

from swmmanywhere import geospatial_operations as go


def test_interp_wrap():
    """Test the interp_wrap function."""
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
    result = go.interp_wrap(yx, interp, grid, values)
    assert result == 0.875

    # Test the function on a nan point
    values_grid[1][1] = np.nan
    yx = (0.251, 0.25) 
    result = go.interp_wrap(yx, interp, grid, values)
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
    result = go.interpolate_points_on_raster(x, y, 'fake_path')

    # [3,2] feels unintuitive but it's because rasters measure from the top
    assert result == [3.0, 2.0]

def test_get_utm():
    """Test the get_utm_epsg function."""
    # Test a northern hemisphere point
    crs = go.get_utm_epsg(-1, 51)
    assert crs == 'EPSG:32630'

    # Test a southern hemisphere point
    crs = go.get_utm_epsg(-1, -51)
    assert crs == 'EPSG:32730'


def test_reproject_raster():
    """Test the reproject_raster function."""
    # Create a mock raster file
    fid = 'test.tif'
    data = np.random.randint(0, 255, (100, 100)).astype('uint8')
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

    # Define the input parameters
    target_crs = 'EPSG:32630'
    new_fid = 'test_reprojected.tif'

    # Call the function
    go.reproject_raster(target_crs, fid)

    # Check if the reprojected file exists
    assert os.path.exists(new_fid)

    # Check if the reprojected file has the correct CRS
    with rst.open(new_fid) as src:
        assert src.crs.to_string() == target_crs

    # Clean up the created files
    os.remove(fid)
    os.remove(new_fid)

def test_get_transformer():
    """Test the get_transformer function."""
    # Test a northern hemisphere point
    transformer = go.get_transformer('EPSG:4326', 'EPSG:32630')

    initial_point = (-0.1276, 51.5074)
    expected_point = (699330.1106898375, 5710164.30300683)
    assert transformer.transform(*initial_point) == expected_point

def test_reproject_df():
    """Test the reproject_df function."""
    # Create a mock DataFrame
    df = pd.DataFrame({
        'longitude': [-0.1276],
        'latitude': [51.5074]
    })

    # Define the input parameters
    source_crs = 'EPSG:4326'
    target_crs = 'EPSG:32630'

    # Call the function
    transformed_df = go.transform_df(df, source_crs, target_crs)

    # Check the output
    assert transformed_df['x'].values[0] == 699330.1106898375
    assert transformed_df['y'].values[0] == 5710164.30300683

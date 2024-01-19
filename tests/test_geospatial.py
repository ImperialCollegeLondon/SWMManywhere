# -*- coding: utf-8 -*-
"""Created on Tue Oct 18 10:35:51 2022.

@author: Barney
"""

# import pytest
from unittest.mock import MagicMock, patch

import numpy as np
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
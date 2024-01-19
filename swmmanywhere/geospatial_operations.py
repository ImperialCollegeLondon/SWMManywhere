# -*- coding: utf-8 -*-
"""Created 2024-01-20.

@author: Barnaby Dobson
"""
import numpy as np
import rasterio as rst
from scipy.interpolate import RegularGridInterpolator


def interp_wrap(xy: tuple[float,float], 
                interp: RegularGridInterpolator, 
                grid: np.ndarray, 
                values: list[float]) -> float:
    """Wrap the interpolation function to handle NaNs.

    Picks the nearest non NaN grid point if the interpolated value is NaN,
    otherwise returns the interpolated value.
    
    Args:
        xy (tuple): Coordinate of interest
        interp (RegularGridInterpolator): The interpolator object.
        grid (np.ndarray): List of xy coordinates of the grid points.
        values (list): The list of values at each point in the grid.

    Returns:
        float: The interpolated value.
    """
    # Call the interpolator
    val = float(interp(xy))
    # If the value is NaN, we need to pick nearest non nan grid point
    if np.isnan(val):
        # Get the distances to all grid points
        distances = np.linalg.norm(grid - xy, axis=1)
        # Get the indices of the grid points sorted by distance
        indices = np.argsort(distances)
        # Iterate over the grid points in order of increasing distance
        for index in indices:
            # If the value at this grid point is not NaN, return it
            if not np.isnan(values[index]):
                return values[index]
    else:
        return val
    
    raise ValueError("No non NaN values found in grid.")

def interpolate_points_on_raster(x: list[float], 
                                 y: list[float], 
                                 elevation_fid: str) -> list[float ]:
    """Interpolate points on a raster.

    Args:
        x (list): X coordinates.
        y (list): Y coordinates.
        elevation_fid (str): Filepath to elevation raster.
        
    Returns:
        elevation (float): Elevation at point.
    """
    with rst.open(elevation_fid) as src:
        # Read the raster data
        data = src.read(1).astype(float)  # Assuming it's a single-band raster
        data[data == src.nodata] = None

        # Get the raster's coordinates
        x = np.linspace(src.bounds.left, src.bounds.right, src.width)
        y = np.linspace(src.bounds.bottom, src.bounds.top, src.height)

        # Define grid
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack([xx.ravel(), yy.ravel()]).T
        values = data.ravel()

        # Define interpolator
        interp = RegularGridInterpolator((y,x), 
                                        np.flipud(data), 
                                        method='linear', 
                                        bounds_error=False, 
                                        fill_value=None)
        # Interpolate for x,y
        return [interp_wrap((y_, x_), interp, grid, values) for x_, y_ in zip(x,y)]
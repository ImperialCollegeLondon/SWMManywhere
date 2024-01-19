# -*- coding: utf-8 -*-
"""Created 2024-01-20.

@author: Barnaby Dobson
"""
from typing import Optional

import numpy as np
import pandas as pd
import pyproj
import rasterio as rst
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.interpolate import RegularGridInterpolator


def get_utm_epsg(lon: float, lat: float) -> str:
    """Get the formatted UTM EPSG code for a given longitude and latitude.

    Args:
        lon (float): Longitude in EPSG:4326 (x)
        lat (float): Latitude in EPSG:4326 (y)

    Returns:
        str: Formatted EPSG code for the UTM zone.
    
    Example:
        >>> get_utm_epsg(-0.1276, 51.5074)
        'EPSG:32630'
    """
    # Determine the UTM zone number
    zone_number = int((lon + 180) / 6) + 1
    # Determine the UTM EPSG code based on the hemisphere
    utm_epsg = 32600 + zone_number if lat >= 0 else 32700 + zone_number
    return 'EPSG:{0}'.format(utm_epsg)

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
    
def reproject_raster(target_crs: str, 
                     fid: str, 
                     new_fid: Optional[str] = None):
    """Reproject a raster to a new CRS.

    Args:
        target_crs (str): Target CRS in EPSG format (e.g., EPSG:32630).
        fid (str): Filepath to the raster to reproject.
        new_fid (str, optional): Filepath to save the reprojected raster. 
            Defaults to None, which will just use fid with '_reprojected'.
    """
    with rst.open(fid) as src:
        # Define the transformation parameters for reprojection
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)

        # Create the output raster file
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        if new_fid is None:
            new_fid = fid.replace('.tif','_reprojected.tif')

        with rst.open(new_fid, 'w', **kwargs) as dst:
            # Reproject the data
            reproject(
                source=rst.band(src, 1),
                destination=rst.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
                )

def get_transformer(source_crs: str, 
                    target_crs: str) -> pyproj.Transformer:
    """Get a transformer object for reprojection.

    Args:
        source_crs (str): Source CRS in EPSG format (e.g., EPSG:32630).
        target_crs (str): Target CRS in EPSG format (e.g., EPSG:32630).

    Returns:
        pyproj.Transformer: Transformer object for reprojection.
    
    Example:
        >>> transformer = get_transformer('EPSG:4326', 'EPSG:32630')
        >>> transformer.transform(-0.1276, 51.5074)
        (699330.1106898375, 5710164.30300683)
    """
    return pyproj.Transformer.from_crs(source_crs, 
                                       target_crs, 
                                       always_xy=True)

def reproject_df(df: pd.DataFrame, 
                 source_crs: str, 
                 target_crs: str) -> pd.DataFrame:
    """Reproject the coordinates in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 'longitude' and 'latitude'.
        source_crs (str): Source CRS in EPSG format (e.g., EPSG:4326).
        target_crs (str): Target CRS in EPSG format (e.g., EPSG:32630).
    """
    # Function to transform coordinates
    df = df.copy()
    transformer = get_transformer(source_crs, target_crs)

    # Reproject the coordinates in the DataFrame
    def f(row):
        return transformer.transform(row['longitude'], 
                                     row['latitude'])

    df['x'], df['y'] = zip(*df.apply(f,axis=1))

    return df
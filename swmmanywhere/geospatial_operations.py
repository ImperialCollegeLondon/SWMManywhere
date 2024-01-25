# -*- coding: utf-8 -*-
"""Created 2024-01-20.

@author: Barnaby Dobson
"""
from functools import lru_cache
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import rasterio as rst
import rioxarray
from rasterio import features
from scipy.interpolate import RegularGridInterpolator
from shapely import geometry as sgeom
from shapely.strtree import STRtree

TransformerFromCRS = lru_cache(pyproj.transformer.Transformer.from_crs)


def get_utm_crs(x: float, 
                y: float, 
                crs: str | int | pyproj.CRS = 'EPSG:4326', 
                datum_name: str = "WGS 84"):
    """Get the UTM CRS code for a given coordinate.

    Note, this function is taken from GeoPandas and modified to use
    for getting the UTM CRS code for a given coordinate.

    Args:
        x (float): Longitude in crs
        y (float): Latitude in crs
        crs (str | int | pyproj.CRS, optional): The CRS of the input 
            coordinates. Defaults to 'EPSG:4326'.
        datum_name (str, optional): The datum name to use for the UTM CRS

    Returns:
        str: Formatted EPSG code for the UTM zone.
    
    Example:
        >>> get_utm_epsg(-0.1276, 51.5074)
        'EPSG:32630'
    """
    if not isinstance(x, float) or not isinstance(y, float):
        raise TypeError("x and y must be floats")

    try:
        crs = pyproj.CRS(crs)
    except pyproj.exceptions.CRSError:
        raise ValueError("Invalid CRS")

    # ensure using geographic coordinates
    if pyproj.CRS(crs).is_geographic:
        lon = x
        lat = y
    else:
        transformer = TransformerFromCRS(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name=datum_name,
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    return f"{utm_crs_list[0].auth_name}:{utm_crs_list[0].code}"


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
     # Open the raster
    with rioxarray.open_rasterio(fid) as raster:

        # Reproject the raster
        reprojected = raster.rio.reproject(target_crs)

        # Define the output filepath
        if new_fid is None:
            new_fid = fid.replace('.tif','_reprojected.tif')

        # Save the reprojected raster
        reprojected.rio.to_raster(new_fid)

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
    pts = gpd.points_from_xy(df["longitude"], 
                             df["latitude"], 
                             crs=source_crs).to_crs(target_crs)
    df = df.copy()
    df['x'] = pts.x
    df['y'] = pts.y
    return df

def reproject_graph(G: nx.Graph, 
                    source_crs: str, 
                    target_crs: str) -> nx.Graph:
    """Reproject the coordinates in a graph.

    osmnx.projection.project_graph might be suitable if some other behaviour
    needs to be captured, but it currently fails the tests so I will ignore for
    now.

    Args:
        G (nx.Graph): Graph with nodes containing 'x' and 'y' properties.
        source_crs (str): Source CRS in EPSG format (e.g., EPSG:4326).
        target_crs (str): Target CRS in EPSG format (e.g., EPSG:32630).
    
    Returns:
        nx.Graph: Graph with nodes containing 'x' and 'y' properties.
    """
    # Create a PyProj transformer for CRS conversion
    transformer = get_transformer(source_crs, target_crs)

    # Create a new graph with the converted nodes and edges
    G_new = G.copy()

    # Convert and add nodes with 'x', 'y' properties
    for node, data in G_new.nodes(data=True):
        x, y = transformer.transform(data['x'], data['y'])
        data['x'] = x
        data['y'] = y

    # Convert and add edges with 'geometry' property
    for u, v, data in G_new.edges(data=True):
        if 'geometry' in data.keys():
            data['geometry'] = sgeom.LineString(transformer.transform(x, y) 
                                      for x, y in data['geometry'].coords)
        else:
            data['geometry'] = sgeom.LineString([[G_new.nodes[u]['x'],
                                        G_new.nodes[u]['y']],
                                       [G_new.nodes[v]['x'],
                                        G_new.nodes[v]['y']]])
    
    return G_new


def nearest_node_buffer(points1: dict[str, sgeom.Point], 
                        points2: dict[str, sgeom.Point], 
                        threshold: float) -> dict:
    """Find the nearest node within a given buffer threshold.

    Args:
        points1 (dict): A dictionary where keys are labels and values are 
            Shapely points geometries.
        points2 (dict): A dictionary where keys are labels and values are 
            Shapely points geometries.
        threshold (float): The maximum distance for a node to be considered 
            'nearest'. If no nodes are within this distance, the node is not 
            included in the output.

    Returns:
        dict: A dictionary where keys are labels from points1 and values are 
            labels from points2 of the nearest nodes within the threshold.
    """
    # Convert the keys of points2 to a list
    labels2 = list(points2.keys())
    
    # Create a spatial index
    tree = STRtree(list(points2.values()))
    
    # Initialize an empty dictionary to store the matching nodes
    matching = {}
    
    # Iterate over points1
    for key, geom in points1.items():
        # Find the nearest node in the spatial index to the current geometry
        nearest = tree.nearest(geom)
        nearest_geom = points2[labels2[nearest]]
        
        # If the nearest node is within the threshold, add it to the 
        # matching dictionary
        if geom.buffer(threshold).intersects(nearest_geom):
            matching[key] = labels2[nearest]
    
    # Return the matching dictionary
    return matching

def burn_shape_in_raster(geoms: list[sgeom.LineString], 
          depth: float,
          raster_fid: str, 
          new_raster_fid: str):
    """Burn a depth into a raster along a list of shapely geometries.

    Args:
        geoms (list): List of Shapely geometries.
        depth (float): Depth to carve.
        raster_fid (str): Filepath to input raster.
        new_raster_fid (str): Filepath to save the carved raster.
    """
    with rst.open(raster_fid) as src:
        # read data
        data = src.read(1)
        data = data.astype(float)
        data_mask = data != src.nodata
        bool_mask = np.zeros(data.shape, dtype=bool)
        for geom in geoms:
            # Create a mask for the line
            mask = features.geometry_mask([sgeom.mapping(geom)], 
                                        out_shape=src.shape, 
                                        transform=src.transform, 
                                        invert=True)
            # modify masked data
            bool_mask[mask] = True  # Adjust this multiplier as needed
        #modify data
        data[bool_mask & data_mask] -= depth
        # Create a new GeoTIFF with modified values
        with rst.open(new_raster_fid, 
                      'w', 
                      driver='GTiff', 
                      height=src.height, 
                      width=src.width, 
                      count=1,
                      dtype=data.dtype, 
                      crs=src.crs, 
                      transform=src.transform, 
                      nodata = src.nodata) as dest:
            dest.write(data, 1)
# -*- coding: utf-8 -*-
"""Created 2024-01-20.

A module containing functions to perform a variety of geospatial operations,
such as reprojecting coordinates and handling raster data.

@author: Barnaby Dobson
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import pysheds
import rasterio as rst
import rioxarray
from pysheds import grid as pgrid
from rasterio import features
from scipy.interpolate import RegularGridInterpolator
from shapely import geometry as sgeom
from shapely import ops as sops
from shapely.strtree import STRtree
from tqdm import tqdm

TransformerFromCRS = lru_cache(pyproj.transformer.Transformer.from_crs)

def get_utm_epsg(x: float, 
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


def interp_with_nans(xy: tuple[float,float], 
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
                                 elevation_fid: Path) -> list[float ]:
    """Interpolate points on a raster.

    Args:
        x (list): X coordinates.
        y (list): Y coordinates.
        elevation_fid (Path): Filepath to elevation raster.
        
    Returns:
        elevation (float): Elevation at point.
    """
    with rst.open(elevation_fid) as src:
        # Read the raster data
        data = src.read(1).astype(float)  # Assuming it's a single-band raster
        data[data == src.nodata] = None

        # Get the raster's coordinates
        x_grid = np.linspace(src.bounds.left, src.bounds.right, src.width)
        y_grid = np.linspace(src.bounds.bottom, src.bounds.top, src.height)

        # Define grid
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid = np.vstack([xx.ravel(), yy.ravel()]).T
        values = data.ravel()

        # Define interpolator
        interp = RegularGridInterpolator((y_grid,x_grid), 
                                        np.flipud(data), 
                                        method='linear', 
                                        bounds_error=False, 
                                        fill_value=None)
        # Interpolate for x,y
        return [interp_with_nans((y_, x_), interp, grid, values) for x_, y_ in zip(x,y)]
    
def reproject_raster(target_crs: str, 
                     fid: Path, 
                     new_fid: Optional[Path] = None):
    """Reproject a raster to a new CRS.

    Args:
        target_crs (str): Target CRS in EPSG format (e.g., EPSG:32630).
        fid (Path): Filepath to the raster to reproject.
        new_fid (Path, optional): Filepath to save the reprojected raster. 
            Defaults to None, which will just use fid with '_reprojected'.
    """
     # Open the raster
    with rioxarray.open_rasterio(fid) as raster:

        # Reproject the raster
        reprojected = raster.rio.reproject(target_crs)

        # Define the output filepath
        if new_fid is None:
            new_fid = Path(str(fid.with_suffix('')) + '_reprojected.tif')

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
          raster_fid: Path, 
          new_raster_fid: Path):
    """Burn a depth into a raster along a list of shapely geometries.

    Args:
        geoms (list): List of Shapely geometries.
        depth (float): Depth to carve.
        raster_fid (Path): Filepath to input raster.
        new_raster_fid (Path): Filepath to save the carved raster.
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

def condition_dem(grid: "pysheds.sgrid.Grid", 
                  dem: "pysheds.sview.Raster") -> "pysheds.sview.Raster":
    """Condition a DEM with pysheds.

    Args:
        grid (pysheds.sgrid.Grid): The grid object.
        dem (pysheds.sview.Raster): The input DEM.

    Returns:
        pysheds.sview.Raster: The conditioned DEM.
    """
    # Fill pits, depressions, and resolve flats in the DEM
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    return inflated_dem

def compute_flow_directions(grid: "pysheds.sgrid.Grid", 
                            inflated_dem: "pysheds.sview.Raster") \
                            -> tuple["pysheds.sview.Raster", tuple]:
    """Compute flow directions.

    Args:
        grid (pysheds.sgrid.Grid): The grid object.
        inflated_dem (pysheds.sview.Raster): The input DEM.

    Returns:
        pysheds.sview.Raster: Flow directions.
        tuple: Direction mapping.
    """
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    return fdir, dirmap

def calculate_flow_accumulation(grid: "pysheds.sgrid.Grid", 
                                fdir: "pysheds.sview.Raster", 
                                dirmap: tuple) -> "pysheds.sview.Raster":
    """Calculate flow accumulation.

    Args:
        grid (pysheds.sgrid.Grid): The grid object.
        fdir (pysheds.sview.Raster): Flow directions.
        dirmap (tuple): Direction mapping.

    Returns:
        pysheds.sview.Raster: Flow accumulations.
    """
    acc = grid.accumulation(fdir, dirmap=dirmap)
    return acc

def delineate_catchment(grid: "pysheds.sgrid.Grid",
                        acc: "pysheds.sview.Raster",
                        fdir: "pysheds.sview.Raster",
                        dirmap: tuple,
                        G: nx.Graph) -> gpd.GeoDataFrame:
    """Delineate catchments.

    Args:
        grid (pysheds.sgrid.Grid): The grid object.
        acc (pysheds.sview.Raster): Flow accumulations.
        fdir (pysheds.sview.Raster): Flow directions.
        dirmap (tuple): Direction mapping.
        G (nx.Graph): The input graph with nodes containing 'x' and 'y'.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', and 'id'. Sorted by area in descending order.
    """
    #TODO - rather than using this mad list of dicts better to just use a gdf
    polys = []
    # Iterate over the nodes in the graph
    for id, data in tqdm(G.nodes(data=True), total=len(G.nodes)):
        # Snap the node to the nearest grid cell
        x, y = data['x'], data['y']
        grid_ = deepcopy(grid)
        x_snap, y_snap = grid_.snap_to_mask(acc > 5, (x, y))
        
        # Delineate the catchment
        catch = grid_.catchment(x=x_snap, 
                                y=y_snap, 
                                fdir=fdir, 
                                dirmap=dirmap, 
                                xytype='coordinate')
        grid_.clip_to(catch)

        # Polygonize the catchment
        shapes = grid_.polygonize()
        catchment_polygon = sops.unary_union([sgeom.shape(shape) for shape, 
                                              value in shapes])
        
        # Add the catchment to the list
        polys.append({'id': id, 
                      'geometry': catchment_polygon,
                      'area': catchment_polygon.area})
    polys = sorted(polys, key=lambda d: d['area'], reverse=True)
    return gpd.GeoDataFrame(polys, crs = grid.crs)

def remove_intersections(polys: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove intersections from polygons.

    Args:
        polys (gpd.GeoDataFrame): A geodataframe with columns id and geometry.
    
    Returns:
        gpd.GeoDataFrame: A geodataframe with columns id and geometry.
    """
    result_polygons = polys.copy()
    for i in tqdm(range(len(polys))):
        for j in range(i + 1, len(polys)):
            if polys.iloc[i]['geometry'].intersects(polys.iloc[j]['geometry']):
                polyi = result_polygons.iloc[i]['geometry']
                polyj = polys.iloc[j]['geometry']
                result_polygons.at[polys.index[i],
                                   'geometry'] = polyi.difference(polyj)
    return result_polygons

def remove_zero_area_subareas(mp: sgeom.MultiPolygon,
                              removed_subareas: List[sgeom.Polygon]) \
                                -> sgeom.MultiPolygon:
    """Remove subareas with zero area from a multipolygon.

    Args:
        mp (sgeom.MultiPolygon): A multipolygon.
        removed_subareas (List[sgeom.Polygon]): A list of removed subareas.
    
    Returns:
        sgeom.MultiPolygon: A multipolygon with zero area subareas removed.
    """
    if hasattr(mp, 'geoms'):
        largest = max(mp.geoms, key=lambda x: x.area)
        removed = [subarea for subarea in mp.geoms if subarea != largest]
        removed_subareas.extend(removed)
        return largest
    else:
        return mp

def attach_unconnected_subareas(polys_gdf: gpd.GeoDataFrame, 
                                unconnected_subareas: List[sgeom.Polygon]) \
                                    -> gpd.GeoDataFrame:
    """Attach unconnected subareas to the nearest polygon.

    Args:
        polys_gdf (gpd.GeoDataFrame): A GeoDataFrame containing polygons with
            columns: 'geometry', 'area', and 'id'. 
        unconnected_subareas (List[sgeom.Polygon]): A list of subareas that are 
            not attached to others.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons.
    """
    tree = STRtree(polys_gdf.geometry)
    for subarea in unconnected_subareas:
        nearest_poly = tree.nearest(subarea)
        ind = polys_gdf.index[nearest_poly]
        new_poly = polys_gdf.loc[ind, 'geometry'].union(subarea)
        if hasattr(new_poly, 'geoms'):
            new_poly = max(new_poly.geoms, key=lambda x: x.area)
        polys_gdf.at[ind, 'geometry'] = new_poly
    return polys_gdf

def calculate_slope(polys_gdf: gpd.GeoDataFrame, 
                    grid: "pysheds.sgrid.Grid", 
                    cell_slopes: np.ndarray) -> gpd.GeoDataFrame:
    """Calculate the average slope of each polygon.

    Args:
        polys_gdf (gpd.GeoDataFrame): A GeoDataFrame containing polygons with
            columns: 'geometry', 'area', and 'id'. 
        grid (pysheds.sgrid.Grid): The grid object.
        cell_slopes (np.ndarray): The slopes of each cell in the grid.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with an added 
            'slope' column.
    """
    polys_gdf['slope'] = None
    for idx, row in polys_gdf.iterrows():
        mask = features.geometry_mask([row.geometry], 
                                      grid.shape, 
                                      grid.affine, 
                                      invert=True)
        average_slope = cell_slopes[mask].mean().mean()
        polys_gdf.loc[idx, 'slope'] = max(float(average_slope), 0)
    return polys_gdf

def derive_subcatchments(G: nx.Graph, fid: Path) -> gpd.GeoDataFrame:
    """Derive subcatchments from the nodes on a graph and a DEM.

    Args:
        G (nx.Graph): The input graph with nodes containing 'x' and 'y'.
        fid (Path): Filepath to the DEM.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'width', and 'slope'.
    """
    # Initialise pysheds grids
    grid = pgrid.Grid.from_raster(str(fid))
    dem = grid.read_raster(str(fid))

    # Condition the DEM
    inflated_dem = condition_dem(grid, dem)

    # Compute flow directions
    fdir, dirmap = compute_flow_directions(grid, inflated_dem)

    # Calculate slopes
    cell_slopes = grid.cell_slopes(dem, fdir)

    # Calculate flow accumulations
    acc = calculate_flow_accumulation(grid, fdir, dirmap)

    # Delineate catchments
    polys = delineate_catchment(grid, acc, fdir, dirmap, G)

    # Remove intersections
    result_polygons = remove_intersections(polys)

    # Convert to GeoDataFrame
    polys_gdf = result_polygons.dropna(subset=['geometry'])
    polys_gdf = polys_gdf[~polys_gdf['geometry'].is_empty]

    # Remove zero area subareas and attach to nearest polygon
    removed_subareas: List[sgeom.Polygon] = list() # needed for mypy
    def remove_(mp): return remove_zero_area_subareas(mp, removed_subareas)
    polys_gdf['geometry'] = polys_gdf['geometry'].apply(remove_)
    polys_gdf = attach_unconnected_subareas(polys_gdf, removed_subareas)

    # Calculate area and slope
    polys_gdf['area'] = polys_gdf.geometry.area
    polys_gdf = calculate_slope(polys_gdf, grid, cell_slopes)

    # Calculate width
    polys_gdf['width'] = polys_gdf['area'].div(np.pi).pow(0.5)
    return polys_gdf

def derive_rc(polys_gdf: gpd.GeoDataFrame,
              G: nx.Graph,
              building_footprints: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Derive the RC of each subcatchment.

    Args:
        polys_gdf (gpd.GeoDataFrame): A GeoDataFrame containing polygons that
            represent subcatchments with columns: 'geometry', 'area', and 'id'. 
        G (nx.Graph): The input graph, with node 'ids' that match polys_gdf and
            edges with the 'id', 'width' and 'geometry' property.
        building_footprints (gpd.GeoDataFrame): A GeoDataFrame containing 
            building footprints with a 'geometry' column.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'impervious_area', and 'rc'.
    """
    polys_gdf = polys_gdf.copy()

    ## Format as swmm type catchments 

    # TODO think harder about lane widths (am I double counting here?)
    lines = []
    for u, v, x in G.edges(data=True):
        lines.append({'geometry' : x['geometry'].buffer(x['width'], 
                                                                cap_style = 2, 
                                                                join_style=2),
                            'id' : x['id']})
    lines_df = pd.DataFrame(lines)
    lines_gdf = gpd.GeoDataFrame(lines_df, 
                                geometry=lines_df.geometry,
                                    crs = polys_gdf.crs)

    result = gpd.overlay(lines_gdf[['geometry']], 
                         building_footprints[['geometry']], 
                         how='union')
    result = gpd.overlay(polys_gdf, result)

    dissolved_result = result.dissolve(by='id').reset_index()
    dissolved_result['impervious_area'] = dissolved_result.geometry.area
    polys_gdf = pd.merge(polys_gdf, 
                            dissolved_result[['id','impervious_area']], 
                            on = 'id',
                            how='left').fillna(0)
    polys_gdf['rc'] = polys_gdf['impervious_area'] / polys_gdf['area'] * 100
    return polys_gdf


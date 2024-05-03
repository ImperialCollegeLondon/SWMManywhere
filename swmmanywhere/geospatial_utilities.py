"""Geospatial utilities module for SWMManywhere.

A module containing functions to perform a variety of geospatial operations,
such as reprojecting coordinates and handling raster data.
"""
from __future__ import annotations

import itertools
import json
import math
import operator
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import rasterio as rst
import rioxarray
import shapely
from rasterio import features
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree
from shapely import geometry as sgeom
from shapely import ops as sops
from shapely.strtree import STRtree
from tqdm import tqdm

from swmmanywhere.logging import logger

os.environ['NUMBA_NUM_THREADS'] = '1'
import pyflwdir  # noqa: E402
import pysheds  # noqa: E402
from pysheds import grid as pgrid  # noqa: E402

TransformerFromCRS = lru_cache(pyproj.transformer.Transformer.from_crs)

def get_utm_epsg(x: float, 
                y: float, 
                crs: str | int | pyproj.CRS = 'EPSG:4326', 
                datum_name: str = "WGS 84") -> str:
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
                     new_fid: Optional[Path] = None) -> None:
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
        >>> x, y = transformer.transform(-0.1276, 51.5074)
        >>> print(f"{x:.6f}, {y:.6f}")
        699330.110690, 5710164.303007
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
            data['geometry'] = sgeom.LineString(
                itertools.starmap(transformer.transform,
                                  data['geometry'].coords))
        else:
            data['geometry'] = sgeom.LineString([[G_new.nodes[u]['x'],
                                        G_new.nodes[u]['y']],
                                       [G_new.nodes[v]['x'],
                                        G_new.nodes[v]['y']]])
    G_new.graph['crs'] = target_crs
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
          new_raster_fid: Path) -> None:
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

def condition_dem(grid: pysheds.sgrid.sGrid, 
                  dem: pysheds.sview.Raster) -> pysheds.sview.Raster:
    """Condition a DEM with pysheds.

    Args:
        grid (pysheds.sgrid.sGrid): The grid object.
        dem (pysheds.sview.Raster): The input DEM.

    Returns:
        pysheds.sview.Raster: The conditioned DEM.
    """
    # Fill pits, depressions, and resolve flats in the DEM
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    return inflated_dem

def compute_flow_directions(grid: pysheds.sgrid.sGrid, 
                            inflated_dem: pysheds.sview.Raster) \
                            -> tuple[pysheds.sview.Raster, tuple]:
    """Compute flow directions.

    Args:
        grid (pysheds.sgrid.sGrid): The grid object.
        inflated_dem (pysheds.sview.Raster): The input DEM.

    Returns:
        pysheds.sview.Raster: Flow directions.
        tuple: Direction mapping.
    """
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_dir = grid.flowdir(inflated_dem, dirmap=dirmap)
    return flow_dir, dirmap

def calculate_flow_accumulation(grid: pysheds.sgrid.sGrid, 
                                flow_dir: pysheds.sview.Raster, 
                                dirmap: tuple) -> pysheds.sview.Raster:
    """Calculate flow accumulation.

    Args:
        grid (pysheds.sgrid.sGrid): The grid object.
        flow_dir (pysheds.sview.Raster): Flow directions.
        dirmap (tuple): Direction mapping.

    Returns:
        pysheds.sview.Raster: Flow accumulations.
    """
    flow_acc = grid.accumulation(flow_dir, dirmap=dirmap)
    return flow_acc

def delineate_catchment(grid: pysheds.sgrid.sGrid,
                        flow_acc: pysheds.sview.Raster,
                        flow_dir: pysheds.sview.Raster,
                        dirmap: tuple,
                        G: nx.Graph) -> gpd.GeoDataFrame:
    """Delineate catchments.

    Args:
        grid (pysheds.sgrid.Grid): The grid object.
        flow_acc (pysheds.sview.Raster): Flow accumulations.
        flow_dir (pysheds.sview.Raster): Flow directions.
        dirmap (tuple): Direction mapping.
        G (nx.Graph): The input graph with nodes containing 'x' and 'y'.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', and 'id'. Sorted by area in descending order.
    """
    polys = []
    # Iterate over the nodes in the graph
    for id, data in tqdm(G.nodes(data=True), total=len(G.nodes)):
        # Snap the node to the nearest grid cell
        x, y = data['x'], data['y']
        grid_ = deepcopy(grid)
        x_snap, y_snap = grid_.snap_to_mask(flow_acc >= 0, (x, y))
        
        # Delineate the catchment
        catch = grid_.catchment(x=x_snap, 
                                y=y_snap, 
                                fdir=flow_dir, 
                                dirmap=dirmap, 
                                xytype='coordinate',
                                algorithm = 'recursive'
                                )
        # n.b. recursive algorithm is not recommended, but crashes with a seg 
        # fault occasionally otherwise.

        grid_.clip_to(catch)

        # Polygonize the catchment
        shapes = grid_.polygonize()
        catchment_polygon = sops.unary_union([sgeom.shape(shape) for shape, 
                                              value in shapes])
        
        # Add the catchment to the list
        polys.append({'id': id, 
                      'geometry': catchment_polygon,
                      'area': catchment_polygon.area})
    polys.sort(key=operator.itemgetter("area"), reverse=True)
    return gpd.GeoDataFrame(polys, crs = grid.crs)

def remove_intersections(polys: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove intersections from a GeoDataFrame of polygons.

    Subcatchments are derived for a given point, and so larger subcatchments
    will contain smaller ones. This function removes the smaller subcatchments
    from the larger ones.

    Args:
        polys (gpd.GeoDataFrame): A GeoDataFrame containing polygons with 
            columns: 'geometry', 'area', and 'id'.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with no 
            intersections.
    """
    result_polygons = polys.copy()
    
    # Sort the polygons by area (smallest first)
    result_polygons = (
        result_polygons.assign(area=result_polygons.geometry.area)
        .sort_values('area', ascending=True)
        .reset_index(drop=True)
    )

    # Store the area of 'trimmed' polygons into a combined geometry, starting
    # with the smallest area polygon
    minimal_geom = result_polygons.iloc[0]['geometry']
    for idx, row in tqdm(result_polygons.iloc[1:].iterrows(), 
                         total=result_polygons.shape[0] - 1):
        
        # Trim the polygon by the combined geometry
        result_polygons.at[idx, 'geometry'] = row['geometry'].difference(
            minimal_geom)

        # Update the combined geometry with the new polygon
        minimal_geom = minimal_geom.union(row['geometry'])
    
    # Sort the polygons by area (largest first) - this is just to conform to
    # the unit test and is not strictly essential
    result_polygons = (
        result_polygons.sort_values('area', ascending=False)
        .drop('area', axis=1)
    )

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
    if not hasattr(mp, 'geoms'):
        return mp

    largest = max(mp.geoms, key=lambda x: x.area)
    removed = [subarea for subarea in mp.geoms if subarea != largest]
    removed_subareas.extend(removed)
    return largest

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
                    grid: pysheds.sgrid.sGrid, 
                    cell_slopes: np.ndarray) -> gpd.GeoDataFrame:
    """Calculate the average slope of each polygon.

    Args:
        polys_gdf (gpd.GeoDataFrame): A GeoDataFrame containing polygons with
            columns: 'geometry', 'area', and 'id'. 
        grid (pysheds.sgrid.sGrid): The grid object.
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

def vectorize(data: np.ndarray, 
              nodata: float, 
              transform: rst.Affine, 
              crs: int, 
              name: str = "value")->gpd.GeoDataFrame:
    """Vectorize raster data into a geodataframe.
    
    Args:
        data (np.ndarray): The raster data.
        nodata (float): The nodata value.
        transform (rst.Affine): The affine transformation.
        crs (int): The CRS of the data.
        name (str, optional): The name of the data. Defaults to "value".

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the vectorized data.
    """
    feats_gen = features.shapes(
        data,
        mask=data != nodata,
        transform=transform,
        connectivity=8,
    )
    feats = [
        {"geometry": geom, "properties": {name: val}} for geom, val in list(feats_gen)
    ]

    # parse to geopandas for plotting / writing to file
    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf[name] = gdf[name].astype(data.dtype)
    return gdf

def delineate_catchment_pyflwdir(grid: pysheds.sgrid.sGrid,
                        flow_dir: pysheds.sview.Raster,
                        G: nx.Graph) -> gpd.GeoDataFrame:
    """Derive subcatchments from the nodes on a graph and a DEM.

    Uses the pyflwdir catchment delineation functionality. About a magnitude
    faster than delineate_catchment.

    Args:
        grid (pysheds.sgrid.Grid): The grid object.
        flow_dir (pysheds.sview.Raster): Flow directions.
        G (nx.Graph): The input graph with nodes containing 'x' and 'y'.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'width', and 'slope'.
    """
    flw = pyflwdir.from_array(
            flow_dir,
            ftype = 'd8',
            check_ftype = False,
            transform = grid.affine,
        )
    bbox = sgeom.box(*grid.bbox)
    u, x, y = zip(*[(u, float(p['x']), float(p['y'])) for u, p in G.nodes(data=True)
                 if sgeom.Point(p['x'], p['y']).within(bbox)])

    subbasins = flw.basins(xy=(x, y))
    gdf_bas = vectorize(subbasins.astype(np.int32), 
                        0, 
                        flw.transform, 
                        G.graph['crs'], 
                        name="basin")
    gdf_bas['id'] = [u[x-1] for x in gdf_bas['basin']]
    return gdf_bas

def derive_subbasins_streamorder(fid: Path,
                                 streamorder: int):
    """Derive subbasins of a given stream order.

    Args:
        fid (Path): Filepath to the DEM.
        streamorder (int): The stream order to delineate subbasins for.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons.
    """
    # Load and process the DEM
    grid, flow_dir, _, _ = load_and_process_dem(fid)

    flw = pyflwdir.from_array(
            flow_dir,
            ftype = 'd8',
            check_ftype = False,
            transform = grid.affine,
        )
    # TODO - use highest valid stream order if streamorder is too high
    streamorder_ = streamorder
    subbasins = np.zeros_like(flow_dir)
    while np.unique(subbasins.reshape(-1)).shape[0] == 1:
        if (
            (streamorder == 0) &
            (os.getenv("SWMMANYWHERE_VERBOSE", "false").lower() == "true")
            ):
            raise ValueError("""No subbasins found in derive_subbasins_streamorder. 
                             Fix your DEM""")


        subbasins, _ = flw.subbasins_streamorder(min_sto=streamorder)
        streamorder -= 1

    if (
        (streamorder != streamorder_ - 1) & 
        (os.getenv("SWMMANYWHERE_VERBOSE", "false").lower() == "true")
        ):
        logger.warning(f"""Stream order {streamorder_} resulted in no subbasins. 
                       Using {streamorder + 1} instead.""")

    gdf_bas = vectorize(subbasins.astype(np.int32),
                            0,
                            flw.transform,
                            grid.crs,
                            name="basin")
    return gdf_bas
    

def load_and_process_dem(fid: Path) -> tuple[pysheds.sgrid.sGrid,
                                                  pysheds.sview.Raster,
                                                    tuple,
                                                  pysheds.sview.Raster]:
    """Load and condition a DEM.

    Args:
        fid (Path): Filepath to the DEM.

    Returns:
        tuple: A tuple containing the grid, flow directions, direction mapping, 
            and cell slopes.
    """
    # Initialise pysheds grids
    grid = pgrid.Grid.from_raster(str(fid))
    dem = grid.read_raster(str(fid))

    # Condition the DEM
    inflated_dem = condition_dem(grid, dem)

     # Compute flow directions
    flow_dir, dirmap = compute_flow_directions(grid, inflated_dem)

    # Calculate slopes
    cell_slopes = grid.cell_slopes(dem, flow_dir)
    
    return grid, flow_dir, dirmap, cell_slopes

def derive_subcatchments(G: nx.Graph, 
                         fid: Path, 
                         method = 'pyflwdir') -> gpd.GeoDataFrame:
    """Derive subcatchments from the nodes on a graph and a DEM.

    Args:
        G (nx.Graph): The input graph with nodes containing 'x' and 'y'.
        fid (Path): Filepath to the DEM.
        method (str, optional): The method to use for delineating catchments. 
            Defaults to 'pyflwdir'. Can also be `pysheds` to use the old 
            method.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'width', and 'slope'.
    """
    if method not in ['pyflwdir', 'pysheds']:
        raise ValueError("Invalid method. Must be 'pyflwdir' or 'pysheds'.")
    
    # Load and process the DEM
    grid, flow_dir, dirmap, cell_slopes = load_and_process_dem(fid)

    if method == 'pysheds':
        # Calculate flow accumulations
        flow_acc = calculate_flow_accumulation(grid, flow_dir, dirmap)

        # Delineate catchments
        polys = delineate_catchment(grid, flow_acc, flow_dir, dirmap, G)

        # Remove intersections
        result_polygons = remove_intersections(polys)

    elif method == 'pyflwdir':
        # Delineate catchments
        result_polygons = delineate_catchment_pyflwdir(grid, flow_dir, G)   

    # Convert to GeoDataFrame
    polys_gdf = result_polygons.dropna(subset=['geometry'])
    polys_gdf = polys_gdf[~polys_gdf['geometry'].is_empty]

    # Remove zero area subareas and attach to nearest polygon
    removed_subareas: List[sgeom.Polygon] = [] # needed for mypy
    def remove_(mp): return remove_zero_area_subareas(mp, removed_subareas)
    polys_gdf['geometry'] = polys_gdf['geometry'].apply(remove_)
    polys_gdf = attach_unconnected_subareas(polys_gdf, removed_subareas)

    # Calculate area and slope
    polys_gdf['area'] = polys_gdf.geometry.area
    polys_gdf = calculate_slope(polys_gdf, grid, cell_slopes)

    # Calculate width
    polys_gdf['width'] = polys_gdf['area'].div(np.pi).pow(0.5)
    return polys_gdf

def derive_rc(subcatchments: gpd.GeoDataFrame,
              building_footprints: gpd.GeoDataFrame,
              streetcover: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Derive the Runoff Coefficient (RC) of each subcatchment.

    The runoff coefficient is the ratio of impervious area to total area. The
    impervious area is calculated by overlaying the subcatchments with building
    footprints and all edges in G buffered by their width parameter (i.e., to
    calculate road area).

    Args:
        subcatchments (gpd.GeoDataFrame): A GeoDataFrame containing polygons that
            represent subcatchments with columns: 'geometry', 'area', and 'id'. 
        building_footprints (gpd.GeoDataFrame): A GeoDataFrame containing 
            building footprints with a 'geometry' column.
        streetcover (gpd.GeoDataFrame): A GeoDataFrame containing street cover
            with a 'geometry' column.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'impervious_area', and 'rc'.

    Author:
        @cheginit, @barneydobson
    """
    # Map buffered streets and buildings to subcatchments
    subcat_tree = subcatchments.sindex
    impervious = gpd.GeoDataFrame(
        pd.concat([building_footprints[['geometry']], 
                   streetcover[['geometry']]]),
        crs = building_footprints.crs
    )
    bf_pidx, sb_pidx = subcat_tree.query(impervious.geometry,
                                         predicate='intersects')
    sb_idx = subcatchments.iloc[sb_pidx].index

    # Calculate impervious area and runoff coefficient (rc)
    subcatchments["impervious_area"] = 0.0

    # Calculate all intersection-impervious geometries
    intersection_area = shapely.intersection(
                subcatchments.iloc[sb_pidx].geometry.to_numpy(), 
                impervious.iloc[bf_pidx].geometry.to_numpy())
    
    # Indicate which catchment each intersection is part of 
    intersections = pd.DataFrame([{'sb_idx': ix,
                                  'impervious_geometry': ia}
                                  for ix, ia in zip(sb_idx, intersection_area)]
                                  )
    
    # Aggregate by catchment
    areas = (
        intersections
        .groupby('sb_idx')
        .apply(shapely.ops.unary_union)
        .apply(shapely.area)
    )

    # Store as impervious area in subcatchments
    subcatchments["impervious_area"] = 0
    subcatchments.loc[areas.index, "impervious_area"] = areas
    subcatchments["rc"] = subcatchments["impervious_area"] / \
        subcatchments.geometry.area * 100
    return subcatchments

def calculate_angle(point1: tuple[float,float], 
                    point2: tuple[float,float],
                    point3: tuple[float,float]) -> float:
    """Calculate the angle between three points.

    Calculate the angle between the vectors formed by (point1, 
    point2) and (point2, point3)

    Args:
        point1 (tuple): The first point (x,y).
        point2 (tuple): The second point (x,y).
        point3 (tuple): The third point (x,y).

    Returns:
        float: The angle between the three points in degrees.
    """
    vector1 = (point1[0] - point2[0], point1[1] - point2[1])
    vector2 = (point3[0] - point2[0], point3[1] - point2[1])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    if magnitude1 * magnitude2 == 0:
        # Avoid division by zero
        return float('inf')

    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Ensure the cosine value is within the valid range [-1, 1]
    cosine_angle = min(max(cosine_angle, -1), 1)

    # Calculate the angle in radians
    angle_radians = math.acos(cosine_angle)

    # Convert angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def nodes_to_features(G: nx.Graph):
    """Convert a graph to a GeoJSON node feature collection.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        dict: A GeoJSON feature collection.
    """
    features = []
    for node, data in G.nodes(data=True):
        feature = {
            'type': 'Feature',
            'geometry': sgeom.mapping(sgeom.Point(data['x'], data['y'])),
            'properties': {'id': node, **data}
        }
        features.append(feature)
    return features

def edges_to_features(G: nx.Graph):
    """Convert a graph to a GeoJSON edge feature collection.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        dict: A GeoJSON feature collection.
    """
    features = []
    for u, v, data in G.edges(data=True):
        if 'geometry' not in data:
            geom = None
        else: 
            geom = sgeom.mapping(data['geometry'])
            del data['geometry']
        feature = {
            'type': 'Feature',
            'geometry': geom,
            'properties': {'u': u, 'v': v, **data}
        }
        features.append(feature)
    return features

def graph_to_geojson(graph: nx.Graph, 
                     fid_nodes: Path, 
                     fid_edges: Path,
                     crs: str):
    """Write a graph to a GeoJSON file.

    Args:
        graph (nx.Graph): The input graph.
        fid_nodes (Path): The filepath to save the nodes GeoJSON file.
        fid_edges (Path): The filepath to save the edges GeoJSON file.
        crs (str): The CRS of the graph.
    """
    graph = graph.copy()
    nodes = nodes_to_features(graph)
    edges = edges_to_features(graph)
    
    for iterable, fid in zip([nodes, edges], 
                             [fid_nodes, fid_edges]):
        geojson = {
            'type': 'FeatureCollection',
            'features' : iterable,
            'crs' : {
                'type': 'name',
                'properties': {
                    'name': "urn:ogc:def:crs:{0}".format(crs.replace(':', '::'))
                }
            }
            }

        with fid.open('w') as output_file:
            json.dump(geojson, output_file, indent=2)
def trim_touching_polygons(polygons: gpd.GeoDataFrame,
                           fid: Path,
                           trim: bool = False) -> gpd.GeoDataFrame:
    """Trim touching polygons in a GeoDataFrame.

    Args:
        polygons (gpd.GeoDataFrame): A GeoDataFrame containing polygons with 
            columns: 'geometry', 'area', and 'id'.
        fid (Path): Filepath to the elevation DEM.

        trim (bool, optional): Whether to trim polygons that touch the edge of
            the DEM or just to warn a user that they do touch the edge. 
            Defaults to False.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with no touching 
            polygons.
    """
    # Get elevation boundary
    with rst.open(fid) as src:
        image = src.read(1)  # Read the first band
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        resolution = abs(transform.a)

    # Mask elevation with data
    data_mask = (image != nodata)
    image[data_mask] = 1
    mask_shapes = features.shapes(image, mask=data_mask, transform=transform)
    
    # Convert shapes to GeoDataFrame
    geoms = [sgeom.Polygon(geom['coordinates'][0]) for geom, value in mask_shapes]
    
    # Create GeoDataFrame from the list of geometries
    dem_outline = gpd.GeoDataFrame({'geometry': geoms}, crs=crs).exterior
    
    ind = polygons.geometry.exterior.buffer(resolution + 1).apply(
        lambda x: x.intersects(dem_outline)).values
    if ind.sum() != 0:
        logger.warning("""Some catchments touch the edge of the elevation DEM, 
                       inspect the outputs and check whether the area of 
                       interest has been included, otherwise widen your bbox""")
    if trim:
        trimmed_gdf = polygons.loc[~ind]
        return trimmed_gdf
    return polygons

def merge_points(coordinates: list[tuple[float, float]], 
                 threshold: float)-> dict:
    """Merge points that are within a threshold distance.

    Args:
        coordinates (list): List of coordinates as tuples.
        threshold(float): The threshold distance for merging points.
    
    Returns:
        dict: A dictionary mapping the original point index to the merged point
            and new coordinate.
    """
    # Create a KDTtree to pair together points within thresholds
    tree = KDTree(coordinates)
    pairs = tree.query_pairs(threshold)

    # Merge pairs into families of points that are all nearby
    families: list = []

    for pair in pairs:
        matched_families = [family for family in families 
                            if pair[0] in family or pair[1] in family]
        
        if matched_families:
            # Merge all matched families and add the current pair
            new_family = set(pair)
            for family in matched_families:
                new_family.update(family)
            
            # Remove the old families and add the newly formed one
            for family in matched_families:
                families.remove(family)
            families.append(new_family)
        else:
            # No matching family found, so create a new one
            families.append(set(pair))

    # Create a mapping of the original point to the merged point
    mapping = {}
    for family in families:
        average_point = np.mean([coordinates[i] for i in family], axis=0)
        family_head = min(list(family))
        for i in family:
            mapping[i] = {'maps_to' : family_head, 
                          'coordinate' : tuple(average_point)}
    return mapping
"""Geospatial utilities module for SWMManywhere.

A module containing functions to perform a variety of geospatial operations,
such as reprojecting coordinates and handling raster data.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import tempfile
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
from pywbt import whitebox_tools
from rasterio import features
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree
from shapely import geometry as sgeom
from shapely.strtree import STRtree
from tqdm.auto import tqdm

from swmmanywhere.logging import logger, verbose

os.environ["NUMBA_NUM_THREADS"] = "1"
import pyflwdir  # noqa: E402

TransformerFromCRS = lru_cache(pyproj.transformer.Transformer.from_crs)


def get_utm_epsg(
    x: float,
    y: float,
    crs: str | int | pyproj.CRS = "EPSG:4326",
    datum_name: str = "WGS 84",
) -> str:
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


def interp_with_nans(
    xy: tuple[float, float],
    interp: RegularGridInterpolator,
    grid: np.ndarray,
    values: list[float],
) -> float:
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


def interpolate_points_on_raster(
    x: list[float], y: list[float], elevation_fid: Path
) -> list[float]:
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
        interp = RegularGridInterpolator(
            (y_grid, x_grid),
            np.flipud(data),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        # Interpolate for x,y
        return [
            interp_with_nans((y_, x_), interp, grid, values) for x_, y_ in zip(x, y)
        ]


def reproject_raster(
    target_crs: str, fid: Path, new_fid: Optional[Path] = None
) -> None:
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
            new_fid = Path(str(fid.with_suffix("")) + "_reprojected.tif")

        # Save the reprojected raster
        reprojected.rio.to_raster(new_fid)


def get_transformer(source_crs: str, target_crs: str) -> pyproj.Transformer:
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
    return pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)


def reproject_df(df: pd.DataFrame, source_crs: str, target_crs: str) -> pd.DataFrame:
    """Reproject the coordinates in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns 'longitude' and 'latitude'.
        source_crs (str): Source CRS in EPSG format (e.g., EPSG:4326).
        target_crs (str): Target CRS in EPSG format (e.g., EPSG:32630).
    """
    # Function to transform coordinates
    pts = gpd.points_from_xy(df["longitude"], df["latitude"], crs=source_crs).to_crs(
        target_crs
    )
    df = df.copy()
    df["x"] = pts.x
    df["y"] = pts.y
    return df


def reproject_graph(G: nx.Graph, source_crs: str, target_crs: str) -> nx.Graph:
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
        x, y = transformer.transform(data["x"], data["y"])
        data["x"] = x
        data["y"] = y

    # Convert and add edges with 'geometry' property
    for u, v, data in G_new.edges(data=True):
        if "geometry" in data.keys():
            data["geometry"] = sgeom.LineString(
                itertools.starmap(transformer.transform, data["geometry"].coords)
            )
        else:
            data["geometry"] = sgeom.LineString(
                [
                    [G_new.nodes[u]["x"], G_new.nodes[u]["y"]],
                    [G_new.nodes[v]["x"], G_new.nodes[v]["y"]],
                ]
            )
    G_new.graph["crs"] = target_crs
    return G_new


def nearest_node_buffer(
    points1: dict[str, sgeom.Point], points2: dict[str, sgeom.Point], threshold: float
) -> dict:
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
    if not points1 or not points2:
        return {}

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


def burn_shape_in_raster(
    geoms: list[sgeom.LineString], depth: float, raster_fid: Path, new_raster_fid: Path
) -> None:
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
            mask = features.geometry_mask(
                [sgeom.mapping(geom)],
                out_shape=src.shape,
                transform=src.transform,
                invert=True,
            )
            # modify masked data
            bool_mask[mask] = True  # Adjust this multiplier as needed
        # modify data
        data[bool_mask & data_mask] -= depth
        # Create a new GeoTIFF with modified values
        with rst.open(
            new_raster_fid,
            "w",
            driver="GTiff",
            height=src.height,
            width=src.width,
            count=1,
            dtype=data.dtype,
            crs=src.crs,
            transform=src.transform,
            nodata=src.nodata,
        ) as dest:
            dest.write(data, 1)


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
        .sort_values("area", ascending=True)
        .reset_index(drop=True)
    )

    # Store the area of 'trimmed' polygons into a combined geometry, starting
    # with the smallest area polygon
    minimal_geom = result_polygons.iloc[0]["geometry"]
    for idx, row in tqdm(
        result_polygons.iloc[1:].iterrows(),
        total=result_polygons.shape[0] - 1,
        disable=not verbose(),
    ):
        # Trim the polygon by the combined geometry
        result_polygons.at[idx, "geometry"] = row["geometry"].difference(minimal_geom)

        # Update the combined geometry with the new polygon
        minimal_geom = minimal_geom.union(row["geometry"])

    # Sort the polygons by area (largest first) - this is just to conform to
    # the unit test and is not strictly essential
    result_polygons = result_polygons.sort_values("area", ascending=False).drop(
        "area", axis=1
    )

    return result_polygons


def remove_zero_area_subareas(
    mp: sgeom.MultiPolygon, removed_subareas: List[sgeom.Polygon]
) -> sgeom.MultiPolygon:
    """Remove subareas with zero area from a multipolygon.

    Args:
        mp (sgeom.MultiPolygon): A multipolygon.
        removed_subareas (List[sgeom.Polygon]): A list of removed subareas.

    Returns:
        sgeom.MultiPolygon: A multipolygon with zero area subareas removed.
    """
    if not hasattr(mp, "geoms"):
        return mp

    largest = max(mp.geoms, key=lambda x: x.area)
    removed = [subarea for subarea in mp.geoms if subarea != largest]
    removed_subareas.extend(removed)
    return largest


def attach_unconnected_subareas(
    polys_gdf: gpd.GeoDataFrame, unconnected_subareas: List[sgeom.Polygon]
) -> gpd.GeoDataFrame:
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
        new_poly = polys_gdf.loc[ind, "geometry"].union(subarea)
        if hasattr(new_poly, "geoms"):
            new_poly = max(new_poly.geoms, key=lambda x: x.area)
        polys_gdf.at[ind, "geometry"] = new_poly
    return polys_gdf


def calculate_slope(
    polys_gdf: gpd.GeoDataFrame, grid: Grid, cell_slopes: np.ndarray
) -> gpd.GeoDataFrame:
    """Calculate the average slope of each polygon.

    Args:
        polys_gdf (gpd.GeoDataFrame): A GeoDataFrame containing polygons with
            columns: 'geometry', 'area', and 'id'.
        grid (Grid): Information of the raster (affine, shape, crs, bbox)
        cell_slopes (np.ndarray): The slopes of each cell in the grid.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with an added
            'slope' column.
    """
    polys_gdf["slope"] = None
    for idx, row in polys_gdf.iterrows():
        mask = features.geometry_mask(
            [row.geometry], grid.shape, grid.affine, invert=True
        )
        average_slope = cell_slopes[mask].mean().mean()
        polys_gdf.loc[idx, "slope"] = max(float(average_slope), 0)
    return polys_gdf


def vectorize(
    data: np.ndarray,
    nodata: float,
    transform: rst.Affine,
    crs: int,
    name: str = "value",
) -> gpd.GeoDataFrame:
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


def delineate_catchment_pyflwdir(
    grid: Grid, flow_dir: np.array, G: nx.Graph
) -> gpd.GeoDataFrame:
    """Derive subcatchments from the nodes on a graph and a DEM.

    Uses the pyflwdir catchment delineation functionality. About a magnitude
    faster than delineate_catchment.

    Args:
        grid (Grid): Information of the raster (affine, shape, crs, bbox).
        flow_dir (np.array): Flow directions.
        G (nx.Graph): The input graph with nodes containing 'x' and 'y'.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'width', and 'slope'.
    """
    flw = pyflwdir.from_array(
        flow_dir,
        ftype="d8",
        check_ftype=False,
        transform=grid.affine,
    )
    bbox = sgeom.box(*grid.bbox)
    u, x, y = zip(
        *[
            (u, float(p["x"]), float(p["y"]))
            for u, p in G.nodes(data=True)
            if sgeom.Point(p["x"], p["y"]).within(bbox)
        ]
    )

    subbasins = flw.basins(xy=(x, y))
    gdf_bas = vectorize(
        subbasins.astype(np.int32), 0, flw.transform, G.graph["crs"], name="basin"
    )
    gdf_bas["id"] = [u[x - 1] for x in gdf_bas["basin"]]
    return gdf_bas


def derive_subbasins_streamorder(
    fid: Path, streamorder: int | None = None, x: list[float] = [], y: list[float] = []
) -> gpd.GeoDataFrame:
    """Derive subbasins.

    Use the pyflwdir snap function to find the most downstream points in each
    subbasin. If streamorder is provided it will use that instead, although
    defaulting to snap if there are no cells of the correct streamorder.

    Args:
        fid (Path): Filepath to the DEM.
        streamorder (int): The stream order to delineate subbasins for.
        x (list): X coordinates.
        y (list): Y coordinates.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons.
    """
    # Load and process the DEM
    grid, flow_dir, _ = load_and_process_dem(fid)

    flw = pyflwdir.from_array(
        flow_dir,
        ftype="d8",
        check_ftype=False,
        transform=grid.affine,
    )
    xy = [
        (x_, y_)
        for x_, y_ in zip(x, y)
        if (x_ > grid.bbox[0])
        and (x_ < grid.bbox[2])
        and (y_ > grid.bbox[1])
        and (y_ < grid.bbox[3])
    ]

    idxs, _ = flw.snap(xy=list(zip(*xy)))
    subbasins = flw.basins(idxs=np.unique(idxs))

    if streamorder is not None:
        # Identify stream order
        subbasins_, _ = flw.subbasins_streamorder(min_sto=streamorder)

        if np.unique(subbasins_).shape[0] == 1:
            logger.warning(
                """No subbasins found in `derive_subbasins_streamorder`. 
                    Instead subbasins have been selected based on the most downstream 
                    points. But you should inspect `subbasins` and probably check your 
                    DEM."""
            )
        else:
            subbasins = subbasins_

    gdf_bas = vectorize(
        subbasins.astype(np.int32), 0, flw.transform, grid.crs, name="basin"
    )

    return gdf_bas


def flwdir_whitebox(fid: Path) -> np.array:
    """Calculate flow direction using WhiteboxTools.

    Args:
        fid (Path): Filepath to the DEM.

    Returns:
        np.array: Flow directions.
    """
    # Initialize WhiteboxTools
    with tempfile.TemporaryDirectory(dir=str(fid.parent)) as temp_dir:
        temp_path = Path(temp_dir)

        # Copy raster to working directory
        dem = temp_path / "dem.tif"
        shutil.copy(fid, dem)

        # Condition
        wbt_args = {
            "BreachDepressions": ["-i=dem.tif", "--fillpits", "-o=dem_corr.tif"],
            "D8Pointer": ["-i=dem_corr.tif", "-o=fdir.tif"],
        }
        whitebox_tools(
            temp_path,
            wbt_args,
            save_dir=temp_path,
            verbose=verbose(),
            wbt_root=temp_path / "WBT",
            zip_path=fid.parent / "whiteboxtools_binaries.zip",
            max_procs=1,
        )

        fdir = temp_path / "fdir.tif"
        if not Path(fdir).exists():
            raise ValueError("Flow direction raster not created.")

        with rst.open(fdir) as src:
            flow_dir = src.read(1)

    # Adjust mapping from WhiteboxTools to pyflwdir
    mapping = {1: 128, 2: 1, 4: 2, 8: 4, 16: 8, 32: 16, 64: 32, 128: 64}
    get_flow_dir = np.vectorize(mapping.get, excluded=["default"])
    flow_dir = get_flow_dir(flow_dir, 0)
    return flow_dir


class Grid:
    """A class to represent a grid."""

    def __init__(self, affine: rst.Affine, shape: tuple, crs: int, bbox: tuple):
        """Initialize the Grid class.

        Args:
            affine (rst.Affine): The affine transformation.
            shape (tuple): The shape of the grid.
            crs (int): The CRS of the grid.
            bbox (tuple): The bounding box of the grid.
        """
        self.affine = affine
        self.shape = shape
        self.crs = crs
        self.bbox = bbox


def load_and_process_dem(
    fid: Path,
    method: str = "whitebox",
) -> tuple[Grid, np.array, np.array]:
    """Load and condition a DEM.

    Args:
        fid (Path): Filepath to the DEM.
        method (str, optional): The method to use for conditioning. Defaults to
            "whitebox".

    Returns:
        tuple: A tuple containing the grid, flow directions, and cell slopes.
    """
    with rst.open(fid, "r") as src:
        elevtn = src.read(1).astype(float)
        nodata = float(src.nodata)
        transform = src.transform
        crs = src.crs

    if method not in ("whitebox", "pyflwdir"):
        raise ValueError("Method must be 'whitebox' or 'pyflwdir'.")

    if method == "whitebox":
        flow_dir = flwdir_whitebox(fid)
    elif method == "pyflwdir":
        flw = pyflwdir.from_dem(
            data=elevtn,
            nodata=nodata,
            transform=transform,
            latlon=crs.is_geographic,
        )
        flow_dir = flw.to_array(ftype="d8").astype(int)

    cell_slopes = pyflwdir.dem.slope(
        elevtn,
        nodata=nodata,
        transform=transform,
        latlon=crs.is_geographic,
    )

    grid = Grid(transform, elevtn.shape, crs, src.bounds)

    return grid, flow_dir, cell_slopes


def derive_subcatchments(
    G: nx.Graph, fid: Path, method: str = "whitebox"
) -> gpd.GeoDataFrame:
    """Derive subcatchments from the nodes on a graph and a DEM.

    Args:
        G (nx.Graph): The input graph with nodes containing 'x' and 'y'.
        fid (Path): Filepath to the DEM.
        method (str, optional): The method to use for conditioning.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons with columns:
            'geometry', 'area', 'id', 'width', and 'slope'.
    """
    # Load and process the DEM
    grid, flow_dir, cell_slopes = load_and_process_dem(fid, method)

    # Delineate catchments
    result_polygons = delineate_catchment_pyflwdir(grid, flow_dir, G)

    # Convert to GeoDataFrame
    polys_gdf = result_polygons.dropna(subset=["geometry"])
    polys_gdf = polys_gdf[~polys_gdf["geometry"].is_empty]

    # Remove zero area subareas and attach to nearest polygon
    removed_subareas: List[sgeom.Polygon] = []  # needed for mypy

    def remove_(mp):
        return remove_zero_area_subareas(mp, removed_subareas)

    polys_gdf["geometry"] = polys_gdf["geometry"].apply(remove_)
    polys_gdf = attach_unconnected_subareas(polys_gdf, removed_subareas)

    # Calculate area and slope
    polys_gdf["area"] = polys_gdf.geometry.area
    polys_gdf = calculate_slope(polys_gdf, grid, cell_slopes)

    # Calculate width
    polys_gdf["width"] = polys_gdf["area"].div(np.pi).pow(0.5)
    return polys_gdf


def derive_rc(
    subcatchments: gpd.GeoDataFrame,
    building_footprints: gpd.GeoDataFrame,
    streetcover: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
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
        pd.concat([building_footprints[["geometry"]], streetcover[["geometry"]]]),
        crs=building_footprints.crs,
    )
    bf_pidx, sb_pidx = subcat_tree.query(impervious.geometry, predicate="intersects")
    sb_idx = subcatchments.iloc[sb_pidx].index

    # Calculate impervious area and runoff coefficient (rc)
    subcatchments["impervious_area"] = 0.0

    # Calculate all intersection-impervious geometries
    intersection_area = shapely.intersection(
        subcatchments.iloc[sb_pidx].geometry.to_numpy(),
        impervious.iloc[bf_pidx].geometry.to_numpy(),
    )

    # Indicate which catchment each intersection is part of
    intersections = pd.DataFrame(
        [
            {"sb_idx": ix, "impervious_geometry": ia}
            for ix, ia in zip(sb_idx, intersection_area)
        ]
    )

    # Aggregate by catchment
    areas = (
        intersections.groupby("sb_idx")
        .apply(shapely.ops.unary_union)
        .apply(shapely.area)
    )

    # Store as impervious area in subcatchments
    subcatchments["impervious_area"] = 0.0
    subcatchments.loc[areas.index, "impervious_area"] = areas
    subcatchments["rc"] = (
        subcatchments["impervious_area"] / subcatchments.geometry.area * 100
    )
    return subcatchments


def calculate_angle(
    point1: tuple[float, float],
    point2: tuple[float, float],
    point3: tuple[float, float],
) -> float:
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
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    if magnitude1 * magnitude2 == 0:
        # Avoid division by zero
        return float("inf")

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
            "type": "Feature",
            "geometry": sgeom.mapping(sgeom.Point(data["x"], data["y"])),
            "properties": {"id": node, **data},
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
        if "geometry" not in data:
            geom = None
        else:
            geom = sgeom.mapping(data["geometry"])
            del data["geometry"]
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": {"u": u, "v": v, **data},
        }
        features.append(feature)
    return features


def graph_to_geojson(graph: nx.Graph, fid_nodes: Path, fid_edges: Path, crs: str):
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

    for iterable, fid in zip([nodes, edges], [fid_nodes, fid_edges]):
        geojson = {
            "type": "FeatureCollection",
            "features": iterable,
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:{0}".format(crs.replace(":", "::"))
                },
            },
        }

        with fid.open("w") as output_file:
            json.dump(geojson, output_file, indent=2)


def merge_points(coordinates: list[tuple[float, float]], threshold: float) -> dict:
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
        matched_families = [
            family for family in families if pair[0] in family or pair[1] in family
        ]

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
            mapping[i] = {"maps_to": family_head, "coordinate": tuple(average_point)}
    return mapping

"""Preprocessing module for SWMManywhere.

A module to call downloads, preprocess these downloads into formats suitable
for graphfcns, and some other utilities (such as creating a project folder
structure or create the starting graph from rivers/streets).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import graph_utilities as gu
from swmmanywhere import prepare_data
from swmmanywhere.filepaths import FilePaths
from swmmanywhere.logging import logger


def write_df(df: pd.DataFrame | gpd.GeoDataFrame, fid: Path):
    """Write a DataFrame to a file.

    Write a DataFrame to a file. The file type is determined by the file
    extension.

    Args:
        df (DataFrame): DataFrame to write to a file.
        fid (Path): Path to the file.
    """
    if fid.suffix in (".geoparquet", ".parquet"):
        df.to_parquet(fid)
    elif fid.suffix in (".geojson", ".json"):
        if isinstance(df, gpd.GeoDataFrame):
            df.to_file(fid, driver="GeoJSON")
        else:
            df.to_json(fid)


def prepare_precipitation(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    api_keys: dict[str, str],
    target_crs: str,
    source_crs: str = "EPSG:4326",
):
    """Download and reproject precipitation data."""
    if addresses.bbox_paths.precipitation.exists():
        return
    logger.info(f"downloading precipitation to {addresses.bbox_paths.precipitation}")
    precip = prepare_data.download_precipitation(
        bbox, api_keys["cds_username"], api_keys["cds_api_key"]
    )
    precip = precip.reset_index()
    precip = go.reproject_df(precip, source_crs, target_crs)
    write_df(precip, addresses.bbox_paths.precipitation)


def prepare_elevation(
    bbox: tuple[float, float, float, float], addresses: FilePaths, target_crs: str
):
    """Download and reproject elevation data."""
    if addresses.bbox_paths.elevation.exists():
        return
    logger.info(f"downloading elevation to {addresses.bbox_paths.elevation}")
    with tempfile.TemporaryDirectory() as temp_dir:
        fid = Path(temp_dir) / "elevation.tif"
        prepare_data.download_elevation(
            fid,
            bbox,
        )
        go.reproject_raster(target_crs, fid, addresses.bbox_paths.elevation)


def prepare_building(
    bbox: tuple[float, float, float, float], addresses: FilePaths, target_crs: str
):
    """Download and reproject building data."""
    if addresses.bbox_paths.building.exists():
        return

    logger.info(f"downloading buildings to {addresses.bbox_paths.building}")
    prepare_data.download_buildings_bbox(addresses.bbox_paths.building, bbox)

    buildings = gpd.read_parquet(addresses.bbox_paths.building)
    buildings = buildings.to_crs(target_crs)
    write_df(buildings, addresses.bbox_paths.building)


def prepare_street(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    target_crs: str,
    source_crs: str = "EPSG:4326",
    network_types=["drive"],
):
    """Download and reproject street graph.

    Download the street graph within the bbox and reproject it to the UTM zone.
    The street graph is downloaded for all network types in network_types. The
    street graph is saved to the addresses.bbox_paths.street directory.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in
            the format (minx, miny, maxx, maxy) in EPSG:4326.
        addresses (FilePaths): Class containing the addresses of the directories.
        target_crs (str): Target CRS to reproject the graph to.
        source_crs (str): Source CRS of the graph.
        network_types (list): List of network types to download. For duplicate
            edges, nx.compose_all selects the attributes in priority of last to
            first. In likelihood, you want to ensure that the last network in
            the list is `drive`, so as to retain information about `lanes`,
            which is needed to calculate impervious area.
    """
    if addresses.bbox_paths.street.exists():
        return
    logger.info(f"downloading network to {addresses.bbox_paths.street}")
    if "drive" in network_types and network_types[-1] != "drive":
        logger.warning(
            """The last network type should be `drive` to retain 
                        `lanes` attribute, needed to calculate impervious area.
                        Moving it to the last position."""
        )
        network_types.pop("drive")
        network_types.append("drive")
    networks = []
    for network_type in network_types:
        network = prepare_data.download_street(bbox, network_type=network_type)
        nx.set_edge_attributes(network, network_type, "network_type")
        networks.append(network)
    street_network = nx.compose_all(networks)

    # Reproject graph
    street_network = go.reproject_graph(street_network, source_crs, target_crs)

    gu.save_graph(street_network, addresses.bbox_paths.street)


def prepare_river(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    target_crs: str,
    source_crs: str = "EPSG:4326",
):
    """Download and reproject river graph."""
    if addresses.bbox_paths.river.exists():
        return
    logger.info(f"downloading river network to {addresses.bbox_paths.river}")
    river_network = prepare_data.download_river(bbox)
    river_network = go.reproject_graph(river_network, source_crs, target_crs)
    gu.save_graph(river_network, addresses.bbox_paths.river)


def run_downloads(
    bbox: tuple[float, float, float, float],
    addresses: FilePaths,
    network_types=["drive"],
):
    """Run the data downloads.

    Run the precipitation, elevation, building, street and river network
    downloads. If the data already exists, do not download it again. Reprojects
    data to the UTM zone.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in
            the format (minx, miny, maxx, maxy) in EPSG:4326.
        addresses (FilePaths): Class containing the addresses of the directories.
        network_types (list): List of network types to download.
    """
    target_crs = go.get_utm_epsg(bbox[0], bbox[1])

    # Download precipitation data
    # Currently commented because it doesn't work
    # prepare_precipitation(bbox, addresses, api_keys, target_crs)

    # Download elevation data
    prepare_elevation(bbox, addresses, target_crs)

    # Download building data
    prepare_building(bbox, addresses, target_crs)

    # Download street network data
    prepare_street(bbox, addresses, target_crs, network_types=network_types)

    # Download river network data
    prepare_river(bbox, addresses, target_crs)


def create_starting_graph(addresses: FilePaths):
    """Create the starting graph.

    Create the starting graph by combining the street and river networks.

    Args:
        addresses (FilePaths): Class containing the addresses of the directories.

    Returns:
        nx.Graph: Combined street and river network.
    """
    river = gu.load_graph(addresses.bbox_paths.river)
    nx.set_edge_attributes(river, "river", "edge_type")
    street = gu.load_graph(addresses.bbox_paths.street)
    nx.set_edge_attributes(street, "street", "edge_type")
    return nx.compose(river, street)

"""Preprocessing module for SWMManywhere.

A module to call downloads, preprocess these downloads into formats suitable
for graphfcns, and some other utilities (such as creating a project folder 
structure or create the starting graph from rivers/streets).
"""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import graph_utilities as gu
from swmmanywhere import parameters, prepare_data
from swmmanywhere.custom_logging import logger


def next_directory(keyword: str, directory: Path) -> int:
    """Find the next directory number.

    Find the next directory number within a directory with a <keyword>_ in its 
    name.

    Args:
        keyword (str): Keyword to search for in the directory name.
        directory (Path): Path to the directory to search within.

    Returns:
        int: Next directory number.
    """
    existing_dirs = [int(d.name.split("_")[-1]) 
                     for d in directory.glob(f"{keyword}_*")]
    return 1 if not existing_dirs else max(existing_dirs) + 1

def check_bboxes(bbox: tuple[float, float, float, float],
                 data_dir: Path) -> int | bool:
    """Find the bounding box number.
    
    Check if the bounding box coordinates match any existing bounding box
    directories within data_dir.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in 
            the format (minx, miny, maxx, maxy).
        data_dir (Path): Path to the data directory.

    Returns:
        int: Bounding box number if the coordinates match, else False.
    """
    # Find all bounding_box_info.json files
    info_fids = list(data_dir.glob("*/*bounding_box_info.json"))

    # Iterate over info files
    for info_fid in info_fids:
        # Read bounding_box_info.json
        with info_fid.open('r') as info_file:
            bounding_info = json.load(info_file)
        # Check if the bounding box coordinates match
        if Counter(bounding_info.get("bbox")) == Counter(bbox):
            bbox_full_dir = info_fid.parent
            bbox_dir = bbox_full_dir.name
            bbox_number = int(bbox_dir.replace('bbox_',''))
            return bbox_number

    return False

def get_next_bbox_number(bbox: tuple[float, float, float, float], 
                         data_dir: Path) -> int:
    """Get the next bounding box number.

    If there are existing bounding box directories, check within them to see if
    any have the same bounding box, otherwise find the next number. If
    there are no existing bounding box directories, return 1.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in 
            the format (minx, miny, maxx, maxy).
        data_dir (Path): Path to the data directory.

    Returns:
        int: Next bounding box number.
    """
    # Search for existing bounding box directories
    bbox_number = check_bboxes(bbox, data_dir)
    if not bbox_number:
        return next_directory('bbox', data_dir)
    return bbox_number

def create_project_structure(bbox: tuple[float, float, float, float],
                             project: str,
                             base_dir: Path,
                             model_number: int | None = None):
    """Create the project directory structure.

    Create the project, bbox, national, model and download directories within 
    the base directory.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in 
            the format (minx, miny, maxx, maxy).
        project (str): Name of the project.
        base_dir (Path): Path to the base directory.
        model_number (int | None): Model number, if not provided it will use a
            number that is one higher than the highest number that exists for
            that bbox.

    Returns:
        Addresses: Class containing the addresses of the directories.
    """
    addresses = parameters.FilePaths(base_dir = base_dir,
                                        project_name = project,
                                        bbox_number = 0,
                                        model_number = 0,
                                        extension = 'parquet')
    
    # Create project and national directories
    addresses.national.mkdir(parents=True, exist_ok=True)

    # Create bounding box directory
    bbox_number = get_next_bbox_number(bbox, addresses.project)
    addresses.bbox_number = bbox_number
    addresses.bbox.mkdir(parents=True, exist_ok=True)
    bounding_box_info = {"bbox": bbox, "project": project}
    if not (addresses.bbox / 'bounding_box_info.json').exists():
        with open(addresses.bbox / 'bounding_box_info.json', 'w') as info_file:
            json.dump(bounding_box_info, info_file, indent=2)

    # Create downloads directory
    addresses.download.mkdir(parents=True, exist_ok=True)

    # Create model directory
    if not model_number:
        addresses.model_number = next_directory('model', addresses.bbox)
    else:
        addresses.model_number = model_number

    addresses.model.mkdir(parents=True, exist_ok=True)

    return addresses


def write_df(df: pd.DataFrame | gpd.GeoDataFrame, 
             fid: Path):
    """Write a DataFrame to a file.

    Write a DataFrame to a file. The file type is determined by the file
    extension.

    Args:
        df (DataFrame): DataFrame to write to a file.
        fid (Path): Path to the file.
    """
    if fid.suffix in ('.geoparquet','.parquet'):
        df.to_parquet(fid)
    elif fid.suffix in ('.json','.geojson'):
        if isinstance(df, gpd.GeoDataFrame):
            df.to_file(fid, driver='GeoJSON')
        else:
            df.to_json(fid)

def prepare_precipitation(bbox: tuple[float, float, float, float],
                          addresses: parameters.FilePaths,
                          api_keys: dict[str, str],
                          target_crs: str,
                          source_crs: str = 'EPSG:4326'):
    """Download and reproject precipitation data."""
    if addresses.precipitation.exists():
        return
    logger.info(f'downloading precipitation to {addresses.precipitation}')
    precip = prepare_data.download_precipitation(bbox,
                                                    api_keys['cds_username'],
                                                    api_keys['cds_api_key'])
    precip = precip.reset_index()
    precip = go.reproject_df(precip, source_crs, target_crs)
    write_df(precip, addresses.precipitation)
    
def prepare_elevation(bbox: tuple[float, float, float, float],
                    addresses: parameters.FilePaths,
                    api_keys: dict[str, str],
                    target_crs: str):
    """Download and reproject elevation data."""
    if addresses.elevation.exists():
        return
    logger.info(f'downloading elevation to {addresses.elevation}')
    with tempfile.TemporaryDirectory() as temp_dir:
        fid = Path(temp_dir) / 'elevation.tif'
        prepare_data.download_elevation(fid,
                                        bbox, 
                                        api_keys['nasadem_key']
                                        )
        go.reproject_raster(target_crs,
                            fid,
                            addresses.elevation)
        
def prepare_building(bbox: tuple[float, float, float, float],
                    addresses: parameters.FilePaths,
                    target_crs: str):
    """Download, trim and reproject building data."""
    if addresses.building.exists():
        return
    
    if not addresses.national_building.exists():  
        logger.info(f'downloading buildings to {addresses.national_building}')
        prepare_data.download_buildings(addresses.national_building, 
                                        bbox[0],
                                        bbox[1])
        
    logger.info(f'trimming buildings to {addresses.building}')
    national_buildings = gpd.read_parquet(addresses.national_building)
    buildings = national_buildings.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]] # type: ignore 
    
    buildings = buildings.to_crs(target_crs)
    write_df(buildings,addresses.building)

def prepare_street(bbox: tuple[float, float, float, float],
                     addresses: parameters.FilePaths,
                     target_crs: str,
                     source_crs: str = 'EPSG:4326',
                     network_types = ['drive']):
    """Download and reproject street graph.
    
    Download the street graph within the bbox and reproject it to the UTM zone.
    The street graph is downloaded for all network types in network_types. The
    street graph is saved to the addresses.street directory.

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
    if addresses.street.exists():
        return
    logger.info(f'downloading network to {addresses.street}')
    if 'drive' in network_types and network_types[-1] != 'drive':
        logger.warning("""The last network type should be `drive` to retain 
                        `lanes` attribute, needed to calculate impervious area.
                        Moving it to the last position.""")
        network_types.remove("drive")
        network_types.append("drive")
    networks = []
    for network_type in network_types:
        network = prepare_data.download_street(bbox, network_type=network_type)
        nx.set_edge_attributes(network, network_type, 'network_type')
        networks.append(network)
    street_network = nx.compose_all(networks)

    # Reproject graph
    street_network = go.reproject_graph(street_network, 
                                        source_crs, 
                                        target_crs)
    
    gu.save_graph(street_network, addresses.street)

def prepare_river(bbox: tuple[float, float, float, float],
                    addresses: parameters.FilePaths,
                    target_crs: str,
                    source_crs: str = 'EPSG:4326'):
    """Download and reproject river graph."""
    if addresses.river.exists():
        return
    logger.info(f'downloading river network to {addresses.river}')
    river_network = prepare_data.download_river(bbox)
    river_network = go.reproject_graph(river_network, 
                                        source_crs,
                                        target_crs)
    gu.save_graph(river_network, addresses.river)

def run_downloads(bbox: tuple[float, float, float, float],
                  addresses: parameters.FilePaths,
                  api_keys: dict[str, str],
                  network_types = ['drive']):
    """Run the data downloads.

    Run the precipitation, elevation, building, street and river network
    downloads. If the data already exists, do not download it again. Reprojects
    data to the UTM zone.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in 
            the format (minx, miny, maxx, maxy) in EPSG:4326.
        addresses (FilePaths): Class containing the addresses of the directories.
        api_keys (dict): Dictionary containing the API keys.
        network_types (list): List of network types to download.
    """
    target_crs = go.get_utm_epsg(bbox[0], bbox[1])

    # Download precipitation data
    prepare_precipitation(bbox, addresses, api_keys, target_crs)
    
    # Download elevation data
    prepare_elevation(bbox, addresses, api_keys, target_crs)
    
    # Download building data
    prepare_building(bbox, addresses, target_crs)
    
    # Download street network data
    prepare_street(bbox, addresses, target_crs, network_types=network_types)

    # Download river network data
    prepare_river(bbox, addresses, target_crs)

def create_starting_graph(addresses: parameters.FilePaths):
    """Create the starting graph.

    Create the starting graph by combining the street and river networks.

    Args:
        addresses (FilePaths): Class containing the addresses of the directories.

    Returns:
        nx.Graph: Combined street and river network.
    """
    river = gu.load_graph(addresses.river)
    nx.set_edge_attributes(river, 'river', 'edge_type')
    street = gu.load_graph(addresses.street)
    nx.set_edge_attributes(street, 'street', 'edge_type')
    return nx.compose(river, street)
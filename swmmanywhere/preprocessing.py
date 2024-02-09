# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

import json
import tempfile
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import graph_utilities as gu
from swmmanywhere import parameters, prepare_data


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
    existing_dirs = [int(d.name.split("_")[-1]) for d in directory.iterdir() 
                     if d.name.startswith(f"{keyword}_")]
    next_dir_number = 1 if not existing_dirs else max(existing_dirs) + 1
    return next_dir_number

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
        with open(info_fid, 'r') as info_file:
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
    else:
        return bbox_number

def create_project_structure(bbox: tuple[float, float, float, float],
                             project: str,
                             base_dir: Path):
    """Create the project directory structure.

    Create the project, bbox, national, model and download directories within 
    the base directory.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in 
            the format (minx, miny, maxx, maxy).
        project (str): Name of the project.
        base_dir (Path): Path to the base directory.

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
    with open(addresses.bbox / 'bounding_box_info.json', 'w') as info_file:
        json.dump(bounding_box_info, info_file, indent=2)

    # Create downloads directory
    addresses.download.mkdir(parents=True, exist_ok=True)

    # Create model directory
    addresses.model_number = next_directory('model', addresses.bbox)
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
    if fid.suffix == '.parquet':
        df.to_parquet(fid)
    elif fid.suffix == '.json':
        if isinstance(df, gpd.GeoDataFrame):
            df.to_file(fid, driver='GeoJSON')
        else:
            df.to_json(fid)

def run_downloads(bbox: tuple[float, float, float, float],
                  addresses: parameters.FilePaths,
                  api_keys: dict[str, str]):
    """Run the data downloads.

    Run the precipitation, elevation, building, street and river network
    downloads. If the data already exists, do not download it again. Assumes
    that data downloads are in EPSG:4326 and reprojects them to the UTM zone.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box coordinates in 
            the format (minx, miny, maxx, maxy) in EPSG:4326.
        addresses (FilePaths): Class containing the addresses of the directories.
        api_keys (dict): Dictionary containing the API keys.
    """
    source_crs = 'EPSG:4326'
    target_crs = go.get_utm_epsg(bbox[0], bbox[1])

    # Download precipitation data
    #TODO precipitation dates..?
    if not addresses.precipitation.exists():
        print(f'downloading precipitation to {addresses.precipitation}')
        precip = prepare_data.download_precipitation(bbox,
                                                     api_keys['cds_username'],
                                                     api_keys['cds_api_key'])
        precip = precip.reset_index()
        precip = go.reproject_df(precip,
                                 source_crs, 
                                 target_crs)
        write_df(precip, 
                 addresses.precipitation)
    
    # Download elevation data
    if not addresses.elevation.exists():
        print(f'downloading elevation to {addresses.elevation}')
        with tempfile.TemporaryDirectory() as temp_dir:
            fid = Path(temp_dir) / 'elevation.tif'
            prepare_data.download_elevation(fid,
                                            bbox, 
                                            api_keys['nasadem_key']
                                            )
            go.reproject_raster(target_crs,
                                fid,
                                addresses.elevation)
        
    else:
        print('elevation already exists')
    
    # Download building data
    if not addresses.national_building.exists():
        print(f'downloading buildings to {addresses.national_building}')
        prepare_data.download_buildings(addresses.national_building, 
                                        bbox[0],
                                        bbox[1])
    else:
        print('buildings already exist')
    
    # Trim and reproject buildings to bbox
    if not addresses.building.exists():
        print(f'trimming buildings to {addresses.building}')
        national_buildings = gpd.read_parquet(addresses.national_building)
        buildings = national_buildings.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]] # type: ignore 
        
        buildings = buildings.to_crs(target_crs)
        write_df(buildings,
                  addresses.building)
    else:
        print('buildings already trimmed')
    
    # Download street network data
    if not addresses.street.exists():
        print(f'downloading street network to {addresses.street}')
        street_network = prepare_data.download_street(bbox)
        street_network = go.reproject_graph(street_network, 
                                            source_crs, 
                                            target_crs)
        gu.save_graph(street_network, addresses.street)
    else:
        print('street network already exists')

    # Download river network data
    if not addresses.river.exists():
        print(f'downloading river network to {addresses.river}')
        river_network = prepare_data.download_river(bbox)
        river_network = go.reproject_graph(river_network, 
                                           source_crs,
                                           target_crs)
        gu.save_graph(river_network, addresses.river)
    else:
        print('river network already exists')
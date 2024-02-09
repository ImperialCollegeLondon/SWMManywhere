# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

import json
from collections import Counter
from pathlib import Path

from swmmanywhere import parameters


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

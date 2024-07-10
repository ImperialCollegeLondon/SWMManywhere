"""File paths module for SWMMAnywhere."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from swmmanywhere.utilities import yaml_dump, yaml_load


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
    addresses = FilePaths(base_dir = base_dir,
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

class ProjectPaths:
    """Paths for the project folder (within base_dir)."""
    def __init__(self, 
                 base_dir: Path, 
                 project_name: str, 
                 extension: str = 'parquet'):
        """Initialise the project paths.
        
        Args:
            base_dir (Path): The base directory.
            project_name (str): The name of the project.
            extension (str): The extension for the files.
        """
        self.project_name = project_name
        self.extension = extension
        self.base_dir = base_dir

        self.project.mkdir(exist_ok=True)
        self.national.mkdir(exist_ok=True)

    @property
    def project(self):
        """The project folder (sits in the base_dir)."""
        return self.base_dir / self.project_name

    @property
    def national(self):
        """The national folder (for national scale downloads)."""
        return self.project / "national"

    @property
    def national_building(self):
        """The national scale building file."""
        return self.national / f"building.{self.extension}"


class BBoxPaths:
    """Paths for the bounding box folder (within project folder)."""

    def __init__(self, 
                 project_paths: ProjectPaths, 
                 bbox_number: int, 
                 extension: str = 'parquet'):
        """Initialise the bounding box paths.

        Args:
            project_paths (ProjectPaths): The project paths.
            bbox_number (int): The bounding box number.
            extension (str): The extension for the files.
        """
        self.base_dir = project_paths.project
        self.bbox_number = bbox_number
        self.extension = extension

        self.bbox.mkdir(exist_ok=True)
        self.download.mkdir(exist_ok=True)

    @property
    def bbox(self):
        """The bounding box folder (specific to a bounding box)."""
        return self.base_dir / f"bbox_{self.bbox_number}"

    @property
    def download(self):
        """The download folder (for bbox specific downloaded data)."""
        return self.bbox / "download"

    @property
    def river(self):
        """The river graph for the bounding box."""
        return self.download / f"river.{self.extension}"

    @property
    def street(self):
        """The street graph for the bounding box."""
        return self.download / f"street.{self.extension}"

    @property
    def elevation(self):
        """The elevation file for the bounding box."""
        return self.download / "elevation.tif"

    @property
    def building(self):
        """The building file for the bounding box (clipped from national scale)."""
        return self.download / f"building.geo{self.extension}"

    @property
    def precipitation(self):
        """The precipitation data."""
        return self.download / f"precipitation.{self.extension}"

class ModelPaths:
    """Paths for the model folder (within bbox folder)."""

    def __init__(self, 
                 bbox_paths: BBoxPaths, 
                 model_number: int,
                 extension: str = 'parquet'):
        """Initialise the model paths.

        Args:
            bbox_paths (BBoxPaths): The bounding box paths.
            model_number (int): The model number.
            extension (str): The extension for the files.
        """
        self.base_dir = bbox_paths.bbox
        self.model_number = model_number
        self.extension = extension

        self.model.mkdir(exist_ok=True)

    @property
    def model(self):
        """The model folder (one specific synthesised model)."""
        return self.base_dir / f"model_{self.model_number}"

    @property
    def inp(self):
        """The synthesised SWMM input file for the model."""
        return self.model / f"model_{self.model_number}.inp"

    @property
    def subcatchments(self):
        """The subcatchments file for the model."""
        return self.model / f"subcatchments.geo{self.extension}"

    @property
    def graph(self):
        """The graph file for the model."""
        return self.model / f"graph.{self.extension}"

    @property
    def nodes(self):
        """The nodes file for the model."""
        return self.model / f"nodes.geo{self.extension}"

    @property
    def edges(self):
        """The edges file for the model."""
        return self.model / f"edges.geo{self.extension}"

    @property
    def streetcover(self):
        """The street cover file for the model."""
        return self.model / f"streetcover.geo{self.extension}"

def filepaths_from_yaml(f: Path):
    """Get file paths from a yaml file."""
    address_dict = yaml_load(f.read_text())
    address_dict['base_dir'] = Path(address_dict['base_dir'])
    addresses = FilePaths(**address_dict)
    return addresses

class FilePaths:
    """File paths class (manager for project, bbox and model)."""

    def __init__(self,
                 base_dir: Path, 
                 project_name: str,
                 bbox_bounds: tuple[float, float, float, float],
                 bbox_number: int | None, 
                 model_number: int | None, 
                 extension: str='parquet',
                 **kwargs):
        """Initialise the file paths.

        Args:
            base_dir (Path): The base directory.
            project_name (str): The name of the project.
            bbox_number (int): The bounding box number.
            model_number (int): The model number.
            extension (str): The extension for the files.
            **kwargs: Additional file paths.
        """
        
        self.project_paths = ProjectPaths(base_dir, project_name, extension)

        if not bbox_number:
            bbox_number = get_next_bbox_number(bbox_bounds, 
                                                self.project_paths.project)
        self.bbox_paths = BBoxPaths(self.project_paths, bbox_number, extension)
        bounding_box_info = {"bbox": bbox_bounds, 
                             "project": self.project_paths.project_name}
        if not (self.bbox_paths.bbox / 'bounding_box_info.json').exists():
            with open(self.bbox_paths.bbox / 'bounding_box_info.json', 'w') as info_file:
                json.dump(bounding_box_info, info_file, indent=2)
                
        if not model_number:
            model_number = next_directory('model', self.bbox_paths.bbox)
        self.model_paths = ModelPaths(self.bbox_paths, model_number, extension)

        self._overrides = {}
        for key, value in kwargs.items():
            value_path = Path(value)
            if not value_path.exists():
                raise FileNotFoundError(f"Path {value} does not exist.")
            self._overrides[key] = value_path
            
    def to_yaml(self, f: Path):
        """Convert a file to json."""
        address_dict = {}
        for attr in ['model_paths', 'bbox_paths', 'project_paths']:
            address_dict.update(getattr(self, attr).__dict__)
        address_dict.update(self._overrides)
        yaml_dump(address_dict, f.open('w'))
    
    def __getattr__(self, name: str):
        """Get an attribute.
        
        Check if the attribute is in the overrides, then check the project, bbox 
        and model paths.

        Args:
            name (str): The attribute name.
        """
        if name in self._overrides:
            return self._overrides[name]
        for paths in [self.project_paths, self.bbox_paths, self.model_paths]:
            if hasattr(paths, name):
                return getattr(paths, name)
        raise AttributeError(f"""'{self.__class__.__name__}' object has no 
                             attribute '{name}'""")

    def __setattr__(self, name, value):
        """Set an attribute.

        Set the attribute. Updating the base attributes, otherwise store in
        the overrides.

        Args:
            name (str): The attribute name.
            value (Any): The attribute value.
        """
        if name in ['project_paths', 'bbox_paths', 'model_paths','_overrides']:
            super().__setattr__(name, value)
        elif name == 'model_number':
            self.model_paths.model_number = value
        elif name == 'bbox_number':
            self.bbox_paths.bbox_number = value
            self.model_paths.base_dir = self.bbox_paths.bbox
        else:
            self._overrides[name] = Path(value)



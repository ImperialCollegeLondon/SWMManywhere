"""File paths module for SWMMAnywhere."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from swmmanywhere.logging import logger
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
    existing_dirs = [int(d.name.split("_")[-1]) for d in directory.glob(f"{keyword}_*")]
    return 1 if not existing_dirs else max(existing_dirs) + 1


def check_bboxes(bbox: tuple[float, float, float, float], data_dir: Path) -> int | bool:
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
    info_fids = data_dir.glob("*/*bounding_box_info.json")

    # Iterate over info files
    for info_fid in info_fids:
        # Read bounding_box_info.json
        with info_fid.open("r") as info_file:
            bounding_info = json.load(info_file)
        # Check if the bounding box coordinates match
        if Counter(bounding_info.get("bbox")) == Counter(bbox):
            bbox_full_dir = info_fid.parent
            bbox_dir = bbox_full_dir.name
            bbox_number = int(bbox_dir.replace("bbox_", ""))
            return bbox_number

    return False


def get_next_bbox_number(
    bbox: tuple[float, float, float, float], data_dir: Path
) -> int:
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
        return next_directory("bbox", data_dir)
    return bbox_number


def get_overrides(klass: type, overrides: dict[str, Path]) -> dict[str, Path]:
    """Get overrides for a class."""
    out = {}
    for p in overrides.copy().keys():
        if not hasattr(klass, p):
            continue
        out[p] = overrides.pop(p)
    return out


class ProjectPaths:
    """Paths for the project folder (within base_dir)."""

    def __init__(
        self, base_dir: Path, project_name: str, extension: str = "parquet", **kwargs
    ):
        """Initialise the project paths.

        Args:
            base_dir (Path): The base directory.
            project_name (str): The name of the project.
            extension (str): The extension for the files.
            **kwargs: Additional file paths to override.
        """
        self.project_name = project_name
        self.extension = extension
        self.base_dir = base_dir
        self.overrides: dict[str, Path] = get_overrides(ProjectPaths, kwargs)

        self.project.mkdir(exist_ok=True)
        self.national.mkdir(exist_ok=True)

    @property
    def project(self):
        """The project folder (sits in the base_dir)."""
        return self.overrides.get("project", self.base_dir / self.project_name)

    @property
    def national(self):
        """The national folder (for national scale downloads)."""
        return self.overrides.get("national", self.project / "national")

    @property
    def national_building(self):
        """The national scale building file."""
        return self.overrides.get(
            "national_building", self.national / f"building.{self.extension}"
        )


class BBoxPaths:
    """Paths for the bounding box folder (within project folder)."""

    def __init__(
        self,
        project_paths: ProjectPaths,
        bbox_bounds: tuple[float, float, float, float],
        bbox_number: int | None = None,
        extension: str = "parquet",
        **kwargs,
    ):
        """Initialise the bounding box paths.

        Args:
            project_paths (ProjectPaths): The project paths.
            bbox_bounds (tuple[float, float, float, float]): Bounding box
                coordinates in the format (minx, miny, maxx, maxy).
            bbox_number (int, optional): The bounding box number. Defaults to
                None, in which case if the existing bounding box has already
                been created it will be found, otherwise the next number will
                be used.
            extension (str, optional): The extension for the files. Defaults to
                'parquet'.
            **kwargs: Additional file paths to override.
        """
        if not bbox_number:
            bbox_number = get_next_bbox_number(bbox_bounds, project_paths.project)

        self.base_dir = project_paths.project
        self.bbox_number = bbox_number
        self.extension = extension
        self.bbox_bounds = bbox_bounds
        self.overrides: dict[str, Path] = get_overrides(BBoxPaths, kwargs)

        self.bbox.mkdir(exist_ok=True)
        self.download.mkdir(exist_ok=True)

        bounding_box_info = {"bbox": bbox_bounds, "project": project_paths.project_name}

        bbox_info_file = self.bbox / "bounding_box_info.json"
        if not bbox_info_file.exists():
            with bbox_info_file.open("w") as info_file:
                json.dump(bounding_box_info, info_file, indent=2)

    @property
    def bbox(self):
        """The bounding box folder (specific to a bounding box)."""
        return self.overrides.get("bbox", self.base_dir / f"bbox_{self.bbox_number}")

    @property
    def download(self):
        """The download folder (for bbox specific downloaded data)."""
        return self.overrides.get("download", self.bbox / "download")

    @property
    def river(self):
        """The river graph for the bounding box."""
        return self.overrides.get("river", self.download / f"river.{self.extension}")

    @property
    def street(self):
        """The street graph for the bounding box."""
        return self.overrides.get("street", self.download / f"street.{self.extension}")

    @property
    def elevation(self):
        """The elevation file for the bounding box."""
        return self.overrides.get("elevation", self.download / "elevation.tif")

    @property
    def building(self):
        """The building file for the bounding box (clipped from national scale)."""
        return self.overrides.get(
            "building", self.download / f"building.geo{self.extension}"
        )

    @property
    def precipitation(self):
        """The precipitation data."""
        return self.overrides.get(
            "precipitation", self.download / f"precipitation.{self.extension}"
        )


class ModelPaths:
    """Paths for the model folder (within bbox folder)."""

    def __init__(
        self,
        bbox_paths: BBoxPaths,
        model_number: int | None = None,
        extension: str = "parquet",
        **kwargs,
    ):
        """Initialise the model paths.

        Args:
            bbox_paths (BBoxPaths): The bounding box paths.
            model_number (int, None): The model number. Defaults to None, in
                which case the next number in the bbox directory will be used.
            extension (str): The extension for the files.
            **kwargs: Additional file paths to override.
        """
        if model_number is None:
            model_number = next_directory("model", bbox_paths.bbox)

        self.base_dir = bbox_paths.bbox
        self.model_number = model_number
        self.extension = extension
        self.overrides: dict[str, Path] = get_overrides(ModelPaths, kwargs)

        self.model.mkdir(exist_ok=True)

    @property
    def model(self):
        """The model folder (one specific synthesised model)."""
        return self.overrides.get("model", self.base_dir / f"model_{self.model_number}")

    @property
    def inp(self):
        """The synthesised SWMM input file for the model."""
        return self.overrides.get("inp", self.model / f"model_{self.model_number}.inp")

    @property
    def subcatchments(self):
        """The subcatchments file for the model."""
        return self.overrides.get(
            "subcatchments", self.model / f"subcatchments.geo{self.extension}"
        )

    @property
    def graph(self):
        """The graph file for the model."""
        return self.overrides.get("graph", self.model / f"graph.{self.extension}")

    @property
    def nodes(self):
        """The nodes file for the model."""
        return self.overrides.get("nodes", self.model / f"nodes.geo{self.extension}")

    @property
    def edges(self):
        """The edges file for the model."""
        return self.overrides.get("edges", self.model / f"edges.geo{self.extension}")

    @property
    def streetcover(self):
        """The street cover file for the model."""
        return self.overrides.get(
            "streetcover", self.model / f"streetcover.geo{self.extension}"
        )


def filepaths_from_yaml(f: Path):
    """Get file paths from a yaml file."""
    address_dict = yaml_load(f.read_text())
    address_dict["base_dir"] = Path(address_dict["base_dir"])
    overrides = address_dict.pop("overrides", {})
    addresses = FilePaths(**address_dict, **overrides)
    return addresses


class FilePaths:
    """File paths class (manager for project, bbox and model)."""

    def __init__(
        self,
        base_dir: Path,
        project_name: str,
        bbox_bounds: tuple[float, float, float, float],
        bbox_number: int | None = None,
        model_number: int | None = None,
        extension: str = "parquet",
        **kwargs,
    ):
        """Initialise the file paths.

        Args:
            base_dir (Path): The base directory.
            project_name (str): The name of the project.
            bbox_bounds (tuple[float, float, float, float]): Bounding box
                coordinates in the format (minx, miny, maxx, maxy).
            bbox_number (int, optional): The bounding box number. Defaults to
                None, in which case if the existing bounding box has already
                been created it will be found, otherwise the next number will
                be used.
            model_number (int, optional): The model number. Defaults to None,
                in which case the next number in the bbox directory will be used.
            extension (str): The extension for the files.
            **kwargs: Additional file paths.
        """
        # Validate overrides and convert to paths
        for p, value in kwargs.items():
            value = Path(value)
            if not value.exists():
                logger.warning(f"Override path for {p}, {value} does not yet exist.")
            kwargs[p] = value

        # Create project paths and apply overrides
        self.project_paths = ProjectPaths(base_dir, project_name, extension, **kwargs)

        # Create bbox paths and apply overrides
        self.bbox_paths = BBoxPaths(
            self.project_paths, bbox_bounds, bbox_number, extension, **kwargs
        )

        # Create model paths and apply overrides
        self.model_paths = ModelPaths(
            self.bbox_paths, model_number, extension, **kwargs
        )

        self._overrides = kwargs

    def to_yaml(self, f: Path):
        """Convert a file to json."""
        address_dict = {}
        for attr in ["model_paths", "bbox_paths", "project_paths"]:
            address_dict.update(getattr(self, attr).__dict__)
            address_dict.update(**getattr(getattr(self, attr), "overrides"))
        address_dict.update(self._overrides)
        yaml_dump(address_dict, f.open("w"))

    def get_path(self, name: str) -> Path:
        """Get a path from _overrides."""
        path = self._overrides.get(name, None)
        if not path:
            raise FileExistsError(f"No file found for `{name}` attribute.")
        return path

    def set_bbox_number(self, number):
        """Set the bounding box number."""
        self.bbox_paths.bbox_number = number
        self.model_paths.base_dir = self.bbox_paths.bbox

    def set_model_number(self, number):
        """Set the model number."""
        self.model_paths.model_number = number

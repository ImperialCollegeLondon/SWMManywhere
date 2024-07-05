"""Parameters and file paths module for SWMManywhere."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from swmmanywhere.utilities import yaml_dump, yaml_load


def get_full_parameters():
    """Get the full set of parameters."""
    return {
        "subcatchment_derivation": SubcatchmentDerivation(),
        "outlet_derivation": OutletDerivation(),
        "topology_derivation": TopologyDerivation(),
        "hydraulic_design": HydraulicDesign(),
        "metric_evaluation": MetricEvaluation()
    }

def get_full_parameters_flat():
    """Get the full set of parameters in a flat format."""
    parameters = get_full_parameters()
    # Flatten
    # parameters_flat = {k : {**y, **{'category' : cat}} 
    #                    for cat,v in parameters.items() 
    #                     for k, y in v.model_json_schema()['properties'].items()}
    parameters_flat = {}
    for cat, v in parameters.items():
        for k, y in v.model_json_schema()['properties'].items():
            parameters_flat[k] = {**y, **{'category' : cat}}

    return parameters_flat

class SubcatchmentDerivation(BaseModel):
    """Parameters for subcatchment derivation."""
    subbasin_streamorder: int = Field(default = None,
            ge = 1,
            le = 20,
            unit = "-",
            description = "Stream order for subbasin derivation.")
    
    subbasin_membership: float = Field(default = 0.5,
            ge = 0,
            le = 1,
            unit = "-",
            description = "Membership threshold for subbasin derivation.")
    
    subbasin_clip_method: str = Field(default = 'subbasin',
        unit = '-',
        description = "Method to clip subbasins, can be `subbasin` or `community`.")
    
    lane_width: float = Field(default = 3.5,
            ge = 2.0,
            le = 5.0,
            unit = "m", 
            description = "Width of a road lane.")

    carve_depth: float = Field(default = 2.0,
            ge = 1.0,
            le = 3.0,
            unit = "m", 
            description = "Depth of road/river carve for flow accumulation.")

    max_street_length: float = Field(default = 60.0,
            ge = 40.0,
            le = 100.0,
            unit = "m", 
            description = "Distance to split streets into segments.")

    node_merge_distance: float = Field(default = 10,
                ge = 1,
                le = 39.9, # should be less than max_street_length
                unit = 'm',
                description = "Distance within which to merge street nodes.")
    
class OutletDerivation(BaseModel):
	"""Parameters for outlet derivation."""
	method: str = Field(default = 'separate',
        unit = '-',
        description = """Method to derive outlet locations, 
            can be 'separate' or 'withtopo'.""")

	river_buffer_distance: float = Field(default = 150.0,
		ge = 10.0,
		le = 500.0,
		unit = "m",
		description = "Buffer distance to link rivers to streets.")

	outlet_length: float = Field(default = 40.0,
		ge = 0.0,
		le = 600.0,
		unit = "-",
		description = "Weight to discourage street drainage into river buffers.")

class TopologyDerivation(BaseModel):
    """Parameters for topology derivation."""
    allowable_networks: list = Field(default = ['walk', 'drive'],
                                     min_items = 1,
                        unit = "-",
                        description = "OSM networks to consider")
    
    weights: list = Field(default = ['chahinian_slope',
                                      'chahinian_angle',
                                      'length',
                                      'contributing_area'],
                        min_items = 1,
                        unit = "-",
                        description = "Weights for topo derivation")

    omit_edges: list = Field(default = ['motorway', 
                                        'motorway_link',
                                        'bridge', 
                                        'tunnel',
                                        'corridor'],
                        min_items = 1,
                        unit = "-",
                        description = "OSM paths pipes are not allowed under")

    chahinian_slope_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to surface slope in topo derivation")
    
    chahinian_angle_scaling: float = Field(default = 0,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to chahinian angle in topo derivation")
    
    length_scaling: float = Field(default = 0.1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to length in topo derivation")
    
    contributing_area_scaling: float = Field(default = 0.1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to contributing area in topo derivation")
    
    chahinian_slope_exponent: float = Field(default = 1,
        le = 2,
        ge = 0,
        unit = "-",
        description = "Exponent to apply to surface slope in topo derivation")
    
    chahinian_angle_exponent: float = Field(default = 1,
        le = 2,
        ge = 0,
        unit = "-",
        description = "Exponent to apply to chahinian angle in topo derivation")

    length_exponent: float = Field(default = 1,
        le = 2,
        ge = 0,
        unit = "-",
        description = "Exponent to apply to length in topo derivation")
    
    contributing_area_exponent: float = Field(default = 1,
        le = 2,
        ge = 0,
        unit = "-",
        description = "Exponent to apply to contributing area in topo derivation")
    

    
    @model_validator(mode='after')
    def check_weights(cls, values):
        """Check that weights have associated scaling and exponents."""
        for weight in values.weights:
                if not hasattr(values, f'{weight}_scaling'):
                        raise ValueError(f"Missing {weight}_scaling")
                if not hasattr(values, f'{weight}_exponent'):
                        raise ValueError(f"Missing {weight}_exponent")
        return values

class HydraulicDesign(BaseModel):
    """Parameters for hydraulic design."""
    diameters: list = Field(default = 
                            np.linspace(0.15,3,int((3-0.15)/0.075) + 1).tolist(),
                            min_items = 1,
                            unit = "m",
                            description = """Diameters to consider in 
                            pipe by pipe method""")
    max_fr: float = Field(default = 0.8,
		le = 1,
		ge = 0,
		unit = "-",
		description = "Maximum filling ratio in pipe by pipe method")
    min_shear: float = Field(default = 2,
		le = 3,
		ge = 0,
		unit = "Pa",
		description = "Minimum wall shear stress in pipe by pipe method")
    min_v: float = Field(default = 0.75,
		le = 2,
		ge = 0,
		unit = "m/s",
		description = "Minimum velocity in pipe by pipe method")
    max_v: float = Field(default = 5,
		le = 10,
		ge = 3,
		unit = "m/s",
		description = "Maximum velocity in pipe by pipe method")
    min_depth: float = Field(default = 0.5,
		le = 1,
		ge = 0,
		unit = "m",
		description = "Minimum excavation depth in pipe by pipe method")
    max_depth: float = Field(default = 5,
		le = 10,
		ge = 2,
		unit = "m",
		description = "Maximum excavation depth in pipe by pipe method")
    precipitation: float = Field(default = 0.006,
		le = 0.010,
		ge = 0.001,
		description = "Depth of design storm in pipe by pipe method",
		unit = "m")

class MetricEvaluation(BaseModel):
    """Parameters for metric evaluation."""
    grid_scale: float = Field(default = 100,
                        le = 5000,
                        ge = 10,
                        unit = "m",
                        description = "Scale of the grid for metric evaluation")
    
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
                 bbox_number: int, 
                 model_number: int, 
                 extension: str='json',
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
        self.bbox_paths = BBoxPaths(self.project_paths, bbox_number, extension)
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
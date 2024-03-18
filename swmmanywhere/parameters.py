# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator


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
        for k, y in v.schema()['properties'].items():
            parameters_flat[k] = {**y, **{'category' : cat}}

    return parameters_flat

class SubcatchmentDerivation(BaseModel):
    """Parameters for subcatchment derivation."""
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
            ge = 20.0,
            le = 100.0,
            unit = "m", 
            description = "Distance to split streets into segments.")

class OutletDerivation(BaseModel):
	"""Parameters for outlet derivation."""
	max_river_length: float = Field(default = 30.0,
		ge = 5.0,
		le = 100.0,
		unit = "m",
		description = "Distance to split rivers into segments.")   

	river_buffer_distance: float = Field(default = 150.0,
		ge = 50.0,
		le = 300.0,
		unit = "m",
		description = "Buffer distance to link rivers to streets.")

	outlet_length: float = Field(default = 40.0,
		ge = 10.0,
		le = 600.0,
		unit = "m",
		description = "Length to discourage street drainage into river buffers.")

class TopologyDerivation(BaseModel):
    """Parameters for topology derivation."""
    weights: list = Field(default = ['chahinian_slope',
                                      'chahinian_angle',
                                      'length',
                                      'contributing_area'],
                        min_items = 1,
                        unit = "-",
                        description = "Weights for topo derivation")

    chahinian_slope_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to surface slope in topo derivation")
    
    chahinian_angle_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to chahinian angle in topo derivation")
    
    length_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to length in topo derivation")
    
    contributing_area_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to contributing area in topo derivation")
    
    chahinian_slope_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to surface slope in topo derivation")
    
    chahinian_angle_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to chahinian angle in topo derivation")

    length_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to length in topo derivation")
    
    contributing_area_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
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
    diameters: list = Field(default = np.linspace(0.15,3,int((3-0.15)/0.075) + 1),
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
                        le = 10,
                        ge = 5000,
                        unit = "m",
                        description = "Scale of the grid for metric evaluation")



class FilePaths:
    """Parameters for file path lookup."""

    def __init__(self, 
                 base_dir: Path, 
                 project_name: str, 
                 bbox_number: int, 
                 model_number: int, 
                 extension: str='json'):
        """Initialise the class."""
        self.base_dir = base_dir
        self.project_name = project_name
        self.bbox_number = bbox_number
        self.model_number = model_number
        self.extension = extension
    
    def __getattr__(self, name):
        """Fetch the address."""
        return self._fetch_address(name)
    
    def _generate_path(self, *subdirs):
        """Generate a path."""
        return self.base_dir.joinpath(*subdirs)

    def _generate_property(self, 
                           property_name: str, 
                           location: str):
        """Generate a property.
        
        Check if the property exists in the class, otherwise generate it.
        
        Args:
            property_name (str): Name of the folder/file.
            location (str): Name of the folder that the property_name exists 
                in.
            
        Returns:
            Path: Path to the property.
        """
        if property_name in self.__dict__.keys():
             return self.__dict__[property_name]
        
        return self._generate_path(self.project_name, 
                                    getattr(self, location),
                                    property_name)

    def _generate_project(self):
        return self._generate_path(self.project_name)

    def _generate_national(self):
        return self._generate_property('national', 'project')

    def _generate_national_building(self):
        return self._generate_property('building.parquet',
                                        'national')
    def _generate_bbox(self):
        return self._generate_property(f'bbox_{self.bbox_number}', 
                                        'project')
    def _generate_model(self):
        return self._generate_property(f'model_{self.model_number}', 
                                        'bbox')
    def _generate_inp(self):
        return self._generate_property(f'model_{self.model_number}.inp',
                                        'model')
    def _generate_subcatchments(self):
        return self._generate_property(f'subcatchments.geo{self.extension}', 
                                        'model')
    def _generate_graph(self):
        return self._generate_property(f'graph.{self.extension}', 
                                        'model')
    def _generate_nodes(self):
        return self._generate_property(f'nodes.geo{self.extension}', 
                                        'model')
    def _generate_edges(self):
        return self._generate_property(f'edges.geo{self.extension}', 
                                        'model')
    def _generate_download(self):
        return self._generate_property('download', 
                                        'bbox')
    def _generate_river(self):
        return self._generate_property('river.json', 
                                        'download')
    def _generate_street(self):
        return self._generate_property('street.json', 
                                        'download')
    def _generate_elevation(self):
        return self._generate_property('elevation.tif', 'download')
    def _generate_building(self):
        return self._generate_property(f'building.geo{self.extension}', 
                                        'download')
    def _generate_precipitation(self):
        return self._generate_property(f'precipitation.{self.extension}', 
                                        'download')

    def _fetch_address(self, name):
        """Fetch the address.
        
        Generate a path to the folder/file described by name. If the 
        folder/file has already been set, then it will be returned. Otherwise
        it will be generated according to the default structure defined below.

        Args:
            name (str): Name of the folder/file.
            
        Returns:
            Path: Path to the folder/file.
        """
        try:
            return getattr(self, f"_generate_{name}")()
        except AttributeError:
            raise AttributeError(
                f"Generate path for '{name}' failed. Attribute not found.")
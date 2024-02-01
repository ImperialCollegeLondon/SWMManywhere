# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


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
    weights: list = Field(default = ['surface_slope',
                                      'chahinan_angle',
                                      'length',
                                      'contributing_area'],
                        min_items = 1,
                        unit = "-",
                        description = "Weights for topo derivation")

    surface_slope_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to surface slope in topo derivation")
    
    chahinan_angle_scaling: float = Field(default = 1,
        le = 1,
        ge = 0,
        unit = "-",
        description = "Constant to apply to chahinan angle in topo derivation")
    
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
    
    surface_slope_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to surface slope in topo derivation")
    
    chahinan_angle_exponent: float = Field(default = 1,
        le = 2,
        ge = -2,
        unit = "-",
        description = "Exponent to apply to chahinan angle in topo derivation")

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

# TODO move this to tests and run it if we're happy with this way of doing things
class NewTopo(TopologyDerivation):
     """Demo for changing weights that should break the validator."""
     weights: list = Field(default = ['surface_slope',
                                      'chahinan_angle',
                                      'length',
                                      'contributing_area',
                                'test'],
                        min_items = 1,
                        unit = "-",
                        description = "Weights for topo derivation")
     
class Addresses:
    """Parameters for address lookup.

    TODO: this doesn't validate addresses to allow for un-initialised data
    (e.g., subcatchments are created by a graph and so cannot be validated).
    """

    def __init__(self, 
                 base_dir: Path, 
                 project_name: str, 
                 bbox_number: int, 
                 model_number: str, 
                 extension: str='json'):
        """Initialise the class."""
        self.base_dir = base_dir
        self.project_name = project_name
        self.bbox_number = bbox_number
        self.model_number = model_number
        self.extension = extension

    def _generate_path(self, *subdirs):
        return self.base_dir.joinpath(*subdirs)

    def _generate_property(self, folder_name, location):
        return property(lambda self: self._generate_path(self.project_name, 
                                                         location, 
                                                         folder_name))

    def _generate_properties(self):
        self.project = self._generate_path(self.project_name)
        self.national = self._generate_property('national', 
                                                'project')
        self.bbox = self._generate_property(f'bbox_{self.bbox_number}', 
                                            'project')
        self.model = self._generate_property(f'model_{self.model_number}', 
                                             'bbox')
        self.subcatchments = self._generate_property('subcatchments', 
                                                     'model')
        self.download = self._generate_property('download', 
                                                'bbox')
        self.elevation = self._generate_property('elevation.tif', 
                                                 'download')
        self.building = self._generate_property(f'building.{self.extension}', 
                                                'download')
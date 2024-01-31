# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

from pathlib import Path

from pydantic import BaseModel, Field


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
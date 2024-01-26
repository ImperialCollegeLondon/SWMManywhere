# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

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

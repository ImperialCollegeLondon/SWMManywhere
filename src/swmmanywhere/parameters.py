"""Parameters module for SWMManywhere."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator


def get_full_parameters():
    """Get the full set of parameters."""
    return {
        "subcatchment_derivation": SubcatchmentDerivation(),
        "outfall_derivation": OutfallDerivation(),
        "topology_derivation": TopologyDerivation(),
        "hydraulic_design": HydraulicDesign(),
        "metric_evaluation": MetricEvaluation(),
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
        for k, y in v.model_json_schema()["properties"].items():
            parameters_flat[k] = {**y, **{"category": cat}}

    return parameters_flat


class SubcatchmentDerivation(BaseModel):
    """Parameters for subcatchment derivation."""

    subbasin_streamorder: int = Field(
        default=None,
        ge=1,
        le=20,
        unit="-",
        description="Stream order for subbasin derivation.",
    )

    subbasin_membership: float = Field(
        default=0.5,
        ge=0,
        le=1,
        unit="-",
        description="Membership threshold for subbasin derivation.",
    )

    subbasin_clip_method: str = Field(
        default="subbasin",
        unit="-",
        description="Method to clip subbasins, can be `subbasin` or `community`.",
    )

    lane_width: float = Field(
        default=3.5, ge=2.0, le=5.0, unit="m", description="Width of a road lane."
    )

    carve_depth: float = Field(
        default=2.0,
        ge=1.0,
        le=3.0,
        unit="m",
        description="Depth of road/river carve for flow accumulation.",
    )

    max_street_length: float = Field(
        default=60.0,
        ge=40.0,
        le=100.0,
        unit="m",
        description="Distance to split streets into segments.",
    )

    node_merge_distance: float = Field(
        default=10,
        ge=1,
        le=39.9,  # should be less than max_street_length
        unit="m",
        description="Distance within which to merge street nodes.",
    )


class OutfallDerivation(BaseModel):
    """Parameters for outfall derivation."""

    method: str = Field(
        default="separate",
        unit="-",
        description="""Method to derive outfall locations, 
            can be 'separate' or 'withtopo'.""",
    )

    river_buffer_distance: float = Field(
        default=150.0,
        ge=10.0,
        le=500.0,
        unit="m",
        description="Buffer distance to link rivers to streets.",
    )

    outfall_length: float = Field(
        default=40.0,
        ge=0.0,
        le=600.0,
        unit="-",
        description="Weight to discourage street drainage into river buffers.",
    )


class TopologyDerivation(BaseModel):
    """Parameters for topology derivation."""

    allowable_networks: list = Field(
        default=["walk", "drive"],
        min_items=1,
        unit="-",
        description="OSM networks to consider",
    )

    weights: list = Field(
        default=["chahinian_slope", "chahinian_angle", "length", "contributing_area"],
        min_items=1,
        unit="-",
        description="Weights for topo derivation",
    )

    omit_edges: list = Field(
        default=["motorway", "motorway_link", "bridge", "tunnel", "corridor"],
        min_items=1,
        unit="-",
        description="OSM paths pipes are not allowed under",
    )

    chahinian_slope_scaling: float = Field(
        default=1,
        le=1,
        ge=0,
        unit="-",
        description="Constant to apply to surface slope in topo derivation",
    )

    chahinian_angle_scaling: float = Field(
        default=0,
        le=1,
        ge=0,
        unit="-",
        description="Constant to apply to chahinian angle in topo derivation",
    )

    length_scaling: float = Field(
        default=0.1,
        le=1,
        ge=0,
        unit="-",
        description="Constant to apply to length in topo derivation",
    )

    contributing_area_scaling: float = Field(
        default=0.1,
        le=1,
        ge=0,
        unit="-",
        description="Constant to apply to contributing area in topo derivation",
    )

    chahinian_slope_exponent: float = Field(
        default=1,
        le=2,
        ge=0,
        unit="-",
        description="Exponent to apply to surface slope in topo derivation",
    )

    chahinian_angle_exponent: float = Field(
        default=1,
        le=2,
        ge=0,
        unit="-",
        description="Exponent to apply to chahinian angle in topo derivation",
    )

    length_exponent: float = Field(
        default=1,
        le=2,
        ge=0,
        unit="-",
        description="Exponent to apply to length in topo derivation",
    )

    contributing_area_exponent: float = Field(
        default=1,
        le=2,
        ge=0,
        unit="-",
        description="Exponent to apply to contributing area in topo derivation",
    )

    @model_validator(mode="after")
    def check_weights(cls, values):
        """Check that weights have associated scaling and exponents."""
        for weight in values.weights:
            if not hasattr(values, f"{weight}_scaling"):
                raise ValueError(f"Missing {weight}_scaling")
            if not hasattr(values, f"{weight}_exponent"):
                raise ValueError(f"Missing {weight}_exponent")
        return values


class HydraulicDesign(BaseModel):
    """Parameters for hydraulic design."""

    diameters: list = Field(
        default=np.linspace(0.15, 3, int((3 - 0.15) / 0.075) + 1).tolist(),
        min_items=1,
        unit="m",
        description="""Diameters to consider in 
                            pipe by pipe method""",
    )
    max_fr: float = Field(
        default=0.8,
        le=1,
        ge=0,
        unit="-",
        description="Maximum filling ratio in pipe by pipe method",
    )
    min_shear: float = Field(
        default=2,
        le=3,
        ge=0,
        unit="Pa",
        description="Minimum wall shear stress in pipe by pipe method",
    )
    min_v: float = Field(
        default=0.75,
        le=2,
        ge=0,
        unit="m/s",
        description="Minimum velocity in pipe by pipe method",
    )
    max_v: float = Field(
        default=5,
        le=10,
        ge=3,
        unit="m/s",
        description="Maximum velocity in pipe by pipe method",
    )
    min_depth: float = Field(
        default=0.5,
        le=1,
        ge=0,
        unit="m",
        description="Minimum excavation depth in pipe by pipe method",
    )
    max_depth: float = Field(
        default=5,
        le=10,
        ge=2,
        unit="m",
        description="Maximum excavation depth in pipe by pipe method",
    )
    precipitation: float = Field(
        default=0.006,
        le=0.010,
        ge=0.001,
        description="Depth of design storm in pipe by pipe method",
        unit="m",
    )


class MetricEvaluation(BaseModel):
    """Parameters for metric evaluation."""

    grid_scale: float = Field(
        default=100,
        le=5000,
        ge=10,
        unit="m",
        description="Scale of the grid for metric evaluation",
    )

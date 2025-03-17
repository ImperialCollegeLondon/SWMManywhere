"""Test parameters."""

from __future__ import annotations

from pathlib import Path

from swmmanywhere import parameters
from swmmanywhere.swmmanywhere import import_module


def test_custom_parameters():
    """Test register_parameter_group."""
    import_module(Path(__file__).parent / "test_data" / "custom_parameters.py")

    assert "new_params" in parameters.get_full_parameters()
    assert "new_param" in parameters.get_full_parameters_flat()


def test_replace_parameter_group():
    """Test replacing a parameter group."""

    @parameters.register_parameter_group("topology_derivation")
    class NewTopologyDerivation(parameters.TopologyDerivation):
        new_weight: float = parameters.Field(
            default=1,
            le=1,
            ge=0,
        )

    # Check has new_weight
    assert hasattr(
        parameters.get_full_parameters()["topology_derivation"], "new_weight"
    )

    # Check retained existing weight
    assert hasattr(
        parameters.get_full_parameters()["topology_derivation"],
        "chahinian_angle_scaling",
    )

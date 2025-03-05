"""Test parameters."""

from pathlib import Path

from swmmanywhere import parameters
from swmmanywhere.swmmanywhere import import_module


def test_custom_parameters():
    """Test register_parameter_group."""
    import_module(Path(__file__).parent / "test_data" / "custom_parameters.py")

    assert "new_params" in parameters.get_full_parameters()
    assert "new_param" in parameters.get_full_parameters_flat()

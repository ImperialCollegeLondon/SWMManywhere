"""Test parameters."""
from swmmanywhere import parameters

def test_custom_parameters():
    """Test register_parameter_group."""
    @parameters.register_parameter_group(name="new_params")
    class new_params(parameters.BaseModel):
        """New parameters."""
        new_param: int = parameters.Field(
            default=1,
            ge=0,
            le=10,
            unit="-",
            description="A new parameter.",
        )
    
    assert "new_params" in parameters.get_full_parameters()
    assert "new_param" in parameters.get_full_parameters_flat()

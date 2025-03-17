from __future__ import annotations

from swmmanywhere import parameters


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

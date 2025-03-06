# Parameters guide

SWMManywhere is a deliberately highly parameterised workflow, with the goal of enabling users to create a diverse range of UDMs. This guide is to explain the logic of the implemented parameters and how to customise them, as what each parameter does is highly specific to the [`graphfcn`](graphfcns_guide.md) that uses it. Instead, to understand specific parameter purposes, you can view all available parameters at the [API](reference-parameters.md).

## Using parameters

Let's look at a [parameter group](reference-parameters.md#swmmanywhere.parameters.OutfallDerivation), which is a group of parameters related to identifying outfall locations.

:::swmmanywhere.parameters.OutfallDerivation
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false
      show_docstring_attributes: false
      show_docstring_description: false
      show_docstring_examples: false
      show_docstring_parameters: false
      show_docstring_returns: false
      show_docstring_raises: false

We can see here three related parameters and relevant metadata, grouped together in a [`pydantic.BaseModel`](https://docs.pydantic.dev/latest/api/base_model/) object. Parameters in SWMManywhere are grouped together because `graphfcns` that need one of them tend to need the others. Let's look at [`identify_outfalls`](reference-graph-utilities.md#swmmanywhere.graphfcns.outfall_graphfcns.identify_outfalls), which needs these parameters.

:::swmmanywhere.graphfcns.outfall_graphfcns.identify_outfalls
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false
      show_docstring_attributes: false
      show_docstring_description: false
      show_docstring_examples: false
      show_docstring_parameters: false
      show_docstring_returns: false
      show_docstring_raises: false

When calling [`iterate_graphfcns`](reference-graph-utilities.md#swmmanywhere.graph_utilities.iterate_graphfcns), for more information see [here](graphfcns_guide.md#lists-of-graph-functions), SWMManywhere will automatically provide any parameters that have been registered to any graphfcn.

## Registering parameters

When you create a new parameter, it will need to belong to an existing or new parameter group.

### Creating a new parameter group(s)

You create a new module(s) that can contain multiple parameter groups. See below as a template of such amodule.

```python
{%
    include-markdown "../tests/test_data/custom_parameters.py"
    comments=false
%}
```

### Adjust config file

We will add the required lines to the
[minimum viable config](config_guide.md#minimum-viable-configuration) template.

```yml
{%
    include-markdown "snippets/minimum_viable_template.yml"
    comments=false
%}
custom_parameter_modules: 
  - /path/to/custom_parameters.py
```

Now when we run our `config` file, these parameters will be registered and any [custom graphfcns](graphfcns_guide.md#add-a-new-graph-function) will have access to them.

### Changing existing parameter groups

There may be cases where you want to change existing parameter groups, such as introducing new weights to the [`calculate_weights`](reference-graph-utilities.md#swmmanywhere.graphfcns.topology_graphfcns.calculate_weights) step so that they are minimized during the shortest path optimization. In this example, we want the [`TopologyDerivation`](reference-parameters.md#swmmanywhere.parameters.TopologyDerviation) group to include some new parameters. We can do this in a similar way to [above](#creating-a-new-parameter-groups), but being mindful to inherit from `TopologyDerivation` rather than `BaseModel`:

```python
from swmmanywhere.parameters import register_parameter_group, TopologyDerivation, Field

@register_parameter_group("topology_derivation")
class NewTopologyDerivation(TopologyDerivation):
    new_weight_scaling: float = Field(
        default=1,
        le=1,
        ge=0,
    )
    new_weight_exponent: float = Field(
        default=1,
        le=2,
        ge=0,
    )
```

Now the `calculate_weights` function will have access to these new weighting parameters, as well as existing ones.

Note, in this specific example of adding custom weights, you will also have to:

- Update the `weights` parameter in your `config` file, for example:

```yaml
parameter_overrides:
  topology_derviation:
    weights:
      - new_weight
      - length
```

- [Create and register a `graphfcn`](graphfcns_guide.md#add-a-new-graph-function) that adds the `new_weight` parameter to the graph.

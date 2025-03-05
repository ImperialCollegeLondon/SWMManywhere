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

We can see here three related parameters and relevant metadata, grouped together in a [`pydantic.BaseModel`](https://docs.pydantic.dev/latest/api/base_model/) object. Parameters in SWMManywhere are grouped together because `graphfcns` that need one of them tend to need the others. Let's look at [`identify_outfalls`](reference-graphfcns.md#swmmanywhere.graphfcns.outfall_graphfcns.identify_outfalls), which needs these parameters.

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

When calling [`iterate_graphfcns`](reference-graph-utilities.md#swmmanywhere.graph_utilities.iterate_graphfcns), for more information see [here](graphfcns_guide.md#lists-of-graph-functions) SWMManywhere will automatically provide any parameters that have been registered.

## Registered parameters and registering parameters

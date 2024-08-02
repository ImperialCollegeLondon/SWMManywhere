# Graph functions guide

SWMManywhere works by starting with a graph of plausible pipe locations
(typically the street network) and iteratively applying functions to transform
that network gradually into a UDM. A graph function is actually a class, of type
[`BaseGraphFunction`](reference-graph-utilities.md#swmmanywhere.graph_utilities.BaseGraphFunction), that can
be called with a function that takes a graph (and some arguments) and returns
an updated graph.

## Using graph functions

Let's look at a [graph function](reference-graph-utilities.md#swmmanywhere.graph_utilities.to_undirected)
that is simply a wrapper for
[`networkx.to_undirected`](https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.to_undirected.html):

> :::swmmanywhere.graph_utilities.to_undirected
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false

We can see that this graph function is a class that can be called with a graph
and returns a graph. Note that the class has been registered with
`@register_graphfcn`.

### Registered graph functions

The `GraphFunctionRegistry` is a dictionary called `graphfcns` that contains all
registered graph functions to be called from one place.

``` py
>>> from swmmanywhere.graph_utilities import graphfcns
>>> print(graphfcns.keys())
dict_keys(['assign_id', 'remove_parallel_edges', 'remove_non_pipe_allowable_links', 
'calculate_streetcover', 'double_directed', 'to_undirected', 'split_long_edges', 
'merge_street_nodes', 'fix_geometries', 'clip_to_catchments', 
'calculate_contributing_area', 'set_elevation', 'set_surface_slope',
'set_chahinian_slope', 'set_chahinian_angle', 'calculate_weights', 'identify_outlets',
'derive_topology', 'pipe_by_pipe'])
```

We will later demonstrate how to [add a new graph function](#add-a-new-graph-function)
to the registry.

### Arguments

In the previous example, we saw that, in addition to a graph, the function
takes `**kwargs`, which are ignored.

> :::swmmanywhere.graph_utilities.to_undirected
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false

While this graph function does not require any information that is not contained
within the graph, most require parameters or file path information to be completed.
A graph function can receive a `FilePaths` object or any number of parameter
categories, which we will briefly explain below.
A full explanation of these is outside scope of this guide, but for now you can
view the [`parameters`](reference-parameters.md) and
[`FilePaths`](reference-filepaths.md) APIs.

We can see an example of using a parameter category with this
[graph function](reference-graph-utilities.md#swmmanywhere.graph_utilities.remove_non_pipe_allowable_links):

> :::swmmanywhere.graph_utilities.remove_non_pipe_allowable_links
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false

We can see that `remove_non_pipe_allowable_links` uses the `omit_edges` parameter,
which is contained in the `parameters.TopologyDerivation` object that
the graph function takes as an argument. Although we recommend changing parameter
values in SWMManywhere with the [configuration file](config_guide.md#changing-parameters)
we will give an example below explain how a parameter can be changed 'manually'
to better understand what is happening at the graph function level.

``` py
>>> from swmmanywhere.examples.data import demo_graph as G
>>> from swmmanywhere.graph_utilities import graphfcns
>>> from swmmanywhere.parameters import TopologyDerivation
>>> G_ = graphfcns.remove_non_pipe_allowable_links(G, TopologyDerivation())
>>> print(f"{len(G.edges) - len(G_.edges)} edges removed")
2 edges removed
>>> G_ = graphfcns.remove_non_pipe_allowable_links(G, 
        TopologyDerivation(omit_edges=["primary", "bridge"])
    )
>>> print(f"{len(G.edges) - len(G_.edges)} edges removed")
16 edges removed
```

We can see that, by changing the parameter to remove more edge types, the
graph function produced a different graph.

## Lists of graph functions

Graph functions are intended to be applied in a sequence, gradually transforming
the graph. SWMManywhere provides a function to do this called
[`iterate_graphfcns`](reference-graph-utilities.md#swmmanywhere.graph_utilities.iterate_graphfcns).

For example:

```python
>>> from swmmanywhere.examples.data import demo_graph as G
>>> from swmmanywhere.graph_utilities import iterate_graphfcns
>>> print(len(G.edges))
22
>>> G = iterate_graphfcns(G, ["assign_id", "remove_non_pipe_allowable_links"])
>>> print(len(G.edges))
20
```

We have applied a list of two graph functions to the graph `G`, which has made
some changes (in this case checking the edge `id` and removing links as above).

In the [configuration file](config_guide.md#customise-graphfcns) we can specify
the list of graph functions to be applied as a `graphfcn_list`.

### Validating graph functions

Furthermore, this `graphfcn_list` also provides opportunities for validation.
For example, see the
[following graph function](reference-graph-utilities.md#swmmanywhere.graph_utilities.set_surface_slope):

> :::swmmanywhere.graph_utilities.set_surface_slope
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false

Critically, we can see that the `set_surface_slope` graph function has a
parameter `required_node_attributes` (not shown above but see also
`required_edge_attributes`), which specify what parameters in the graph
are required to perform the graph function. Although providing this information
does not guarantee that the graph function will behave as intended, if it is
not provided then the graph function is guaranteed to fail. To check the
feasibility of a set of graph functions a-priori, the parameter
`adds_edge_attributes` (not shown above but see also `adds_node_attributes`),
can be used to specify what, if any, parameters are added to the graph by the
graph function.

## Add a new graph function

Adding a custom graph function can be done by creating a graph function in the
appropriate style, see below for example to create a new graph function and then
specifying to use it with the [`config` file](config_guide.md).

### Write the graph function

You create a new module that can contain multiple graph functions. See below
as a template of that module.

```python
{%
    include-markdown "../tests/test_data/custom_graphfcns.py"
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
custom_graphfcn_modules: /path/to/custom_graphfcns.py
graphfcn_list: 
  - assign_id
  - fix_geometries
  - remove_non_pipe_allowable_links
  - calculate_streetcover
  - remove_parallel_edges
  - to_undirected
  - split_long_edges
  - merge_street_nodes
  - assign_id
  - clip_to_catchments
  - calculate_contributing_area
  - set_elevation
  - double_directed
  - fix_geometries
  - set_surface_slope
  - set_chahinian_slope
  - set_chahinian_angle
  - calculate_weights
  - identify_outlets
  - derive_topology
  - pipe_by_pipe
  - fix_geometries
  - assign_id
  - new_graphfcn
```

We can see that we now provide the `graphfcn_list` with `new_graphfcn` in the
list. This list (except for `new_graphfcn`) is reproduced from the
[`demo_config.yml`](reference-defs.md#demo-configuration-file). Any number of
new graph functions can be inserted at any points in the `graphfcn_list`.

And we provide
the path to the `custom_graphfcns.py` module that contains our `new_graphfcn`
under the `custom_graphfcn_module` entry.
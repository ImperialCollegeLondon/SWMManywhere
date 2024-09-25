# Metrics guide

If you have real data to compare against your synthesised UDM, you can take
advantage of the multiple `metrics` implemented in SWMManywhere. Metrics are
used to compare the similarity of either the synthesised UDM with the real, or
the accompanying SWMM simulation results. Thus, a `metric` is a function that
can take a variety of [arguments](#arguments) and returns the metric value.

## Using metrics

Let's look at a [metric](reference-metric-utilities.md#swmmanywhere.metric_utilities.nc_deltacon0)
that is simply a wrapper for [`netcomp.deltacon0`](https://arxiv.org/pdf/2010.16019):

:::swmmanywhere.metric_utilities.nc_deltacon0
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

We can see that this metric requires the synthesised and real graphs as
arguments, that is because it is a metric to compare the similarity of two graphs.
Note that the function has been registered with `@metrics.register`.

### Registered metrics

The [`MetricRegistry`](reference-metric-utilities.md#swmmanywhere.metric_utilities.MetricRegistry)
is a dictionary subclass called `metrics` that contains all
registered metrics to be called from one place.

``` py
>>> from swmmanywhere.metric_utilities import metrics
>>> print(metrics.keys())
dict_keys(['outfall_nse_flow', 'outfall_kge_flow', 'outfall_relerror_flow',
'outfall_relerror_length', 'outfall_relerror_npipes', 'outfall_relerror_nmanholes',
'outfall_relerror_diameter', 'outfall_nse_flooding', 'outfall_kge_flooding',
'outfall_relerror_flooding', 'grid_nse_flooding', 'grid_kge_flooding',
'grid_relerror_flooding', 'subcatchment_nse_flooding',
'subcatchment_kge_flooding', 'subcatchment_relerror_flooding', 'nc_deltacon0',
'nc_laplacian_dist', 'nc_laplacian_norm_dist', 'nc_adjacency_dist',
'nc_vertex_edge_distance', 'nc_resistance_distance', 'bias_flood_depth',
'kstest_edge_betweenness', 'kstest_betweenness', 'outfall_kstest_diameters'])
```

We will later demonstrate how to [add a new metric](#add-a-new-metric) to the
registry.

### Arguments

In the [previous example](#using-metrics), we saw that, in addition to the synthesised and real
graphs, the function takes `**kwargs`, which are ignored. While `nc_deltacon0` only requires `real_G` and `synthesised_G` to be calculated,
any `metric` has access to a range of arguments for calculation:

- the synthesised and real graphs (`real_G` and `synthesised_G`),
- the synthesised and real simulation results (`real_results` and
`synthesised_results`),
- the synthesised and real sub-catchments (`real_subs` and `synthesised_subs`),
- the [`MetricEvaluation`](reference-parameters.md#swmmanywhere.parameters.MetricEvaluation)
parameters category.

For example, see the [following metric](reference-metric-utilities.md#swmmanywhere.metric_utilities.outfall_kstest_diameters)

:::swmmanywhere.metric_utilities.outfall_kstest_diameters
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

We can see that this metric also requires the real results (`real_results`) and
real subcatchments (`real_subs`) to be evaluated, which are passed as arguments.

## Lists of metrics

Metrics are intended to be applied as part of a list, `metric_list` with the
SWMManywhere function
[`iterate_metrics`](reference-metric-utilities.md#swmmanywhere.metric_utilities.iterate_metrics).

For example:

```python
>>> from swmmanywhere.examples.data import demo_graph as G
>>> from swmmanywhere.metric_utilities import iterate_metrics
>>> iterate_metrics(
...     real_G = G,
...     synthetic_G = G, 
...     metric_list = ['nc_deltacon0','nc_resistance_distance']
... )
{'nc_deltacon0': 0.0, 'nc_resistance_distance': 0.0}
```

In this example only graph comparison metrics are included in `metric_list`, and
so we only need to provide `synthesised_G`, `real_G` and `metric_list`.
We see that the metrics have returned `0.0`, because the two graphs are identical.

In the [configuration file](config_guide.md#performance-metrics) we can specify
the list of metrics to be applied as a `metric_list`. By default this list will
be populated from [`demo_config.yml`](reference-defs.md#demo-configuration-file).

## Add a new metric

Adding a custom metric can be done by creating a metric in the appropriate style,
see below for example to create a new metric and then specifying to use it with
the [`config` file](config_guide.md)

### Write the metric

You create a new module that can contain multiple metrics.
See below as a template of that module.

```python
{%
    include-markdown "../tests/test_data/custom_metrics.py"
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
real:
  inp: /path/to/real/model.inp
  graph: /path/to/real/graph.json
  subcatchments: /path/to/real/subcatchments.geojson
  results: null
custom_metric_modules: /path/to/custom_metrics.py
metric_list: 
  - new_metric
```

To enable metrics to be calculated we must provide information on the `real`
UDM (reproduced from the
[`demo_config.yml`](reference-defs.md#demo-configuration-file)
).

We can see that we now provide the `metric_list` with `new_metric` in the
list. Any number of custom metrics may be provided across one or multiple modules.
Only the metrics specified in `metric_list` will be calculated, use the
[metric registry](#registered-metrics) to identify allowable metrics.

And we provide
the path to the `custom_metrics.py` module that contains our `new_metric`
under the `custom_metric_modules` entry.

## Generalised behaviour of metrics

Because of the large number of potential metrics that can plausibly be calculated,
owing to the wide number of variations that might be applied, a combination of
approaches are used to streamline metric creation.

### Metric factory

Metrics can be created as self-contained functions, as with the example
[earlier](#using-metrics). However, most metrics are created with the
[`metric_factory`](reference-metric-utilities.md#swmmanywhere.metric_utilities.metric_factory).
This is a function that takes a metric as a `str` which contains the metric's
`<scale>_<coefficient>_<variable>`. [Coefficients](#coefficients) and
[scales](#scales) are explained below, while the `variable` is simply the name
of the variable (whether timeseries or graph property) to be calculated. Let
us create a metric with `metric_factory`:

``` py
>>> from swmmanywhere.metric_utilities import metric_factory
>>> metric_factory('outfall_nse_flow')
<function metric_factory.<locals>.new_metric at 0x000001EECEA7C220>
```

We have created a function that is a valid metric, it calculates the Nash-Sutcliffe
Efficiency (`nse`) value for `flow` timeseries at `outlet` scale. Note that
creating the metric with the `metric_factory` does not automatically add it to
the registry.

We do not currently support adding new coefficient or scales via the
[configuration file](config_guide.md). Thus, the following sections
[coefficients](#coefficients) and [scales](#scales) will explain how to manually
accommodate custom behaviour.

### Coefficients

The coefficient portion of a metric is the equation that is applied to two arrays.
See for example the
[`nse`](reference-metric-utilities.md#swmmanywhere.metric_utilities.nse):

:::swmmanywhere.metric_utilities.nse
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

Coefficients are stored in a registry which is a dictionary containing all
registered coefficients:

``` py
>>> from swmmanywhere.metric_utilities import coef_registry
>>> print(coef_registry.keys())
dict_keys(['relerror', 'nse', 'kge'])
```

We can see here the registered coefficients. Thus, the `coefficient` portion of a
string being passed to `metric_factory` must take one of these values.

To register a new coefficient, we can use `register_coef`, and we can then use
it in `metric_factory` to create a metric. Since this new metric is not included
in `metrics` we can also register that.

``` py
>>> from swmmanywhere.metric_utilities import (
...     register_coef, 
...     coef_registry,
...     metric_factory,
...     metrics
... )
>>> import numpy as np
>>> metric_factory('outfall_rmse_flow') # Try creating the metric
Traceback (most recent call last):
    ...
KeyError: 'rmse'
>>> def rmse(y: np.array, yhat: np.array): np.sqrt(np.mean(np.pow(y-yhat,2)))
... 
>>> register_coef(rmse) # Register new coefficient
<function rmse at 0x000001DC38ABC540>
>>> print(coef_registry.keys())
dict_keys(['relerror', 'nse', 'kge', 'rmse'])
>>> metrics.register(metric_factory('outfall_rmse_flow')) # Create and register new metric
<function metric_factory.<locals>.new_metric at 0x00000227D219E020>
>>> 'outfall_rmse_flow' in metrics # Check that the metric is available for use
True
```

### Scales

SWMManywhere supports a variety of spatial scales for which metrics may be calculated.
As with coefficients, these are stored in a registry.

``` py
>>> from swmmanywhere.metric_utilities import scale_registry
>>> print(scale_registry.keys())
dict_keys(['subcatchment', 'grid', 'outfall'])
```

For example, `subcatchment` aligns `real` and `synthesised` subcatchments
together and calculates the coefficient for each subcatchment (returning the
median coefficient value over all matched subcatchments).

:::swmmanywhere.metric_utilities.subcatchment
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

You will have to read the [API](reference-metric-utilities.md) to understand the
differences between scales. However, custom scales may be created in much the
same way as custom coefficients, albeit with more arguments required and more
complexity for the function to interpret different argument values.

### Restrictions

Because of the complexity in interpretation, a key element of the `metric_factory`
is restrictions on certain combinations of scales/coefficients/variables.

For example, conceptually it makes no sense to apply the `nse` coefficient to
the `npipes` variable - as `nse` is used to compare timeseries while `npipes`
is a description of the designed UDM.

``` py
>>> from swmmanywhere.metric_utilities import metric_factory
>>> metric_factory('outfall_nse_npipes')
Traceback (most recent call last):
    ... , in restriction_on_metric
    raise ValueError(f"Variable {variable} only valid with relerror metric")
ValueError: Variable npipes only valid with relerror metric
```

Restrictions are stored in a register as with coefficients and scales,
and we can see that the restriction triggered above was the
[`restriction_on_metric`](reference-metric-utilities.md#swmmanywhere.metric_utilities.restriction_on_metric):

:::swmmanywhere.metric_utilities.restriction_on_metric
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

Custom restrictions can be added as with coefficients and scales.

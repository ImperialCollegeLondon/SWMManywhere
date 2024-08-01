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

We can see that this metric requires the synthesised and real graphs as
arguments, that is because it is a metric to compare the similarity of two graphs.
Note that the function has been registered with `@register_metric`.

### Registered metrics

The `MetricRegistry` is a dictionary called `metrics` that contains all registered
metrics to be called from one place.

``` py
>>> from swmmanywhere.metric_utilities import metrics
>>> print(metrics.keys())
dict_keys(['outlet_nse_flow', 'outlet_kge_flow', 'outlet_relerror_flow',
'outlet_relerror_length', 'outlet_relerror_npipes', 'outlet_relerror_nmanholes',
'outlet_relerror_diameter', 'outlet_nse_flooding', 'outlet_kge_flooding',
'outlet_relerror_flooding', 'grid_nse_flooding', 'grid_kge_flooding',
'grid_relerror_flooding', 'subcatchment_nse_flooding',
'subcatchment_kge_flooding', 'subcatchment_relerror_flooding', 'nc_deltacon0',
'nc_laplacian_dist', 'nc_laplacian_norm_dist', 'nc_adjacency_dist',
'nc_vertex_edge_distance', 'nc_resistance_distance', 'bias_flood_depth',
'kstest_edge_betweenness', 'kstest_betweenness', 'outlet_kstest_diameters'])
```

We will later demonstrate how to [add a new metric](#add-a-new-metric) to the
registry.

### Arguments

In the previous example, we saw that, in addition to the synthesised and real
graphs, the function takes `**kwargs`, which are ignored.

:::swmmanywhere.metric_utilities.nc_deltacon0
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false

While this metric only requires `real_G` and `synthesised_G` to be calculated,
any `metric` has access to a range of arguments for calculation:

- the synthesised and real graphs (`real_G` and `synthesised_G`),
- the synthesised and real simulation results (`real_results` and
`synthesised_results`),
- the synthesised and real sub-catchments (`real_subs` and `synthesised_subs`),
- the [`MetricEvaluation`](reference-parameters.md#swmmanywhere.parameters.MetricEvaluation)
parameters category.

For example, see the [following metric](reference-metric-utilities.md#swmmanywhere.metric_utilities.outlet_kstest_diameters)

:::swmmanywhere.metric_utilities.outlet_kstest_diameters
    handler: python
    options:
      members: no
      show_root_heading: false
      show_bases: false
      show_source: true
      show_root_toc_entry: false

## Lists of metrics

## Generalised behaviour of metrics

### Restrictions, coefficients and scales

### Metric factory

## Add a new metric

### Write the metric

### Adjust config file
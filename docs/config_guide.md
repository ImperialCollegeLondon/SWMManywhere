# Configuration file guide

The `config` file is the intended way for users to interact with SWMManywhere,
enabling a variety of features to customise the synthesis process, which we will
describe in this guide. You can view an example configuration file that contains
entries for all settable attributes at
[`demo_config.yml`](reference-defs.md#demo-configuration-file) and the schema
that must be followed at [`schema.yml`](reference-defs.md#schema-for-configuration-file).

## Minimum viable configuration

The minimum requirements for a user to provide are simply:

- a base directory,
- a project name,
- a bounding box that specifies the latitude and longitude (EPSG:4326) of the bottom left and upper right corners of the region within which to create the Urban Drainage Model (UDM).

We can define a simple configuration `.yml` file here:

```yml
{%
    include-markdown "snippets/minimum_viable_template.yml"
    comments=false
%}
```

## Customising your synthetic UDM

Unless you are exceptionally lucky, it is likely that you will want to change
something about a UDM synthesised by SWMManywhere. There are three key tools
inside the SWMManywhere `config` file that can be helpful to do this, described
in this section.

### Changing parameters

Changing parameter values is by far the easiest way to change your derived
network. You can view available [parameters](reference-parameters.md) and determine
which ones may help to fix what you are unhappy with, for example, if you feel that
there are too many manholes, then you may want to increase `max_street_length` (see
source code of
[`SubcatchmentDerivation`](reference-parameters.md#swmmanywhere.parameters.SubcatchmentDerivation)).

To change parameters via the `config` file, we can use the `parameter_overrides`
field:

```yml
parameter_overrides:
  subcatchment_derivation:
    max_street_length: 40
```

Note that we must provide the parameter category for the parameter that we are
changing (`subcatchment_derivation` above).

As our SWMManywhere paper [link preprint](https://doi.org/10.31223/X5GT5X) demonstrates, you can capture an enormously wide range of UDM behaviours through changing parameters. However, if your system is particularly unusual, or you are testing out new behaviours then you may need to adopt a more elaborate approach.

### Customise `graphfcns`

Graph functions are the way that operations are applied to create a synthetic UDM
in SWMManywhere. You can read more about them [here](graphfcns_guide.md), but a primary feature of
the `config` file is to provide a `graphfcn_list`. By default `graphfcn_list` is
selected from [`demo_config.yml`](reference-defs.md#demo-configuration-file).
Although we believe that the default list makes sense, you may instead provide
your own `graphfcn_list`, this is essential if you plan to [add a `graphfcn`](graphfcns_guide.md#add-a-new-graph-function).

Sometimes, it doesn't matter how clever your functionality is, because your
initial graph is missing something, in which case you will need to...

### Change `starting_graph`

By default SWMManywhere uses Open Street Map (OSM) data to create a starting graph,
to which subsequent `graphfcns` are applied to. A key limitation of SWMManywhere is
that the plausibility of a synthetic pipe being placed in a given location requires
links to exist in the original `starting_graph`. If you are in a location where
the quality of OSM is questionable, or perhaps you have knowledge that some pipes
exist in unusual locations, then you can provide the address to a custom graph
with the `starting_graph` entry in the `config` file. Note, for information on the
format that this graph should take, see
[`save_graph`](reference-graph-utilities.md#swmmanywhere.graph_utilities.save_graph).

If the default workflow is missing something that isn't to do with parameters,
functionality, or the starting graph, then it is possible you will require additional
data sources beyond those that are provided by default...

### Use your own data

A user may want to change some of the key data files that underpin SWMManywhere, such
as the base `elevation` file (i.e., a DEM from NASADEM) if you have higher resolution
data for your region. You can explore the default file structure of a SWMManywhere
project at [`FilePaths`](reference-filepaths.md), but a user may provide their own file paths
via the config file through the `address_overrides` entry, for example:

```yml
address_overrides:
  elevation: /new/path/to/elevation.tif
```

## Evaluating your synthetic UDM

If you are lucky enough to have a pre-existing UDM for your region, then you may
instead be using SWMManywhere to explore uncertainties. If this is the case then
it is likely that you will be evaluating how the synthesised UDM compare to the
pre-existing ones. To do this, you will need to specify the pre-existing (or 'real')
UDM file paths in the `config` file, and the performance metrics to be calculated.

### Specifying the 'real' UDM

In SWMManywhere, a pre-existing UDM is referred to as the 'real' model (although
if you are using SWMManywhere you are presumably aware that describing any UDM
as real is tenuous). To enable SWMManywhere to compare against a real model, we
use the `real` entry of the `config` file, see the `real` entry in
[`demo_config.yml`](reference-defs.md#demo-configuration-file) for example.
The path to a `subcatchments` geometry file, and a `graph` file (see
[`save_graph`](reference-graph-utilities.md#swmmanywhere.graph_utilities.save_graph)
for format) must be provided - however this is temporary and will be unnecessary
following the fixing of
[this](https://github.com/ImperialCollegeLondon/SWMManywhere/issues/84).
The user can then provide either an `inp` path to the SWMM `.inp`
model file, or if the file has already been run, directly to the `results` file. If
a `results` file is provided this will always be used for metric calculation. If
`results` is not provided but `inp` is, then SWMManywhere will run the `inp` model
file provided. Currently the user must ensure precipitation timeseries are aligned
and comparable for both the real and synthetic networks.

Once you have some real
simulations to compare against your synthetic ones, you must specify how to evaluate
the synthetic model...

### Performance metrics

The SWMManywhere package comes with a wide variety of performance
metrics that can be used to make this comparison, explained
[here](metrics_guide.md). In the `config`
file you can specify which metrics should be calculated under the `metric_list`
entry. The [`demo_config.yml`](reference-defs.md#demo-configuration-file)
`metric_list` contains all metrics that come with SWMManywhere, although you may
want to choose a subselection of these if you have a very large network (say >5000
nodes) because some of the graph-based metrics can be slow to calculate (anything
starting with `nc` or containing `betweenness`). You may also be unsatisfied with
the built in metrics, in which case you can
[add your own](metrics_guide.md#add-a-new-metric),
although these must be specified under `metric_list` for them to be calculated.

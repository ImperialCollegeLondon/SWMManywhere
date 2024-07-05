# Configuration file guide

The `config` file is the intended way for users to interact with SWMManywhere,
enabling a variety of features to customise the synthesis process, which we will
describe in this guide. You can view an example configuration file that contains
entries for all settable attributes at
[`demo_config.yml`](../reference-defs/#demo-configuration-file) and the schema
that must be followed at [`schema.yml`](../reference-defs/#schema-for-configuration-file).

## Minimum viable configuration

The minimum requirements for a user to provide are simply:

- a base directory,
- a project name,
- a bounding box that specifies the latitude and longitude (EPSG:4326) of the bottom left and upper right corners of the region within which to create the UDM.

We can define a simple configuration `.yml` file here:

```yml
base_dir: /path/to/base/directory
project: my_first_swmm
bbox: [1.52740,42.50524,1.54273,42.51259]
```

## Customising your synthetic UDM

Unless you are exceptionally lucky, it is likely that you will want to change
something about a UDM synthesised by SWMManywhere. There are three key tools
inside the SWMManywhere `config` file that can be helpful to do this, described
in this section.

### 1. Changing parameters

Changing parameter values is by far the easiest way to change your derived
network. You can view available [parameters](../reference-parameters) and determine
which ones may help to fix what you are unhappy with, for example, if you feel that
there are too many manholes, then you may want to reduce `max_street_length` (see
source code of
[`SubcatchmentDerivation`](../reference-parameters/#swmmanywhere.parameters.SubcatchmentDerivation)).

To change parameters via the `config` file, we can use the `parameter_overrides`
field:

```yml
parameter_overrides:
  subcatchment_derivation:
    max_street_length: 40
```

Note that we must provide the parameter category for the parameter that we are
changing (`subcatchment_derivation` above). As our SWMManywhere paper [link preprint]
demonstrates, you can capture an enormously wide range of UDM behaviours through
changing parameters. However, if your system is particularly unusual, or you are
testing out new behaviours then you may need to...

### 2. Customise `graphfcns`

Graph functions are the way that operations are applied to create a synthetic UDM
in SWMManywhere. You can read more about them [here], but a primary feature of
the `config` file is to provide a `graphfcn_list`. By default `graphfcn_list` is
selected from [`demo_config.yml`](../reference-defs/#demo-configuration-file).
Although we believe that the default list makes sense, you may instead provide
your own `graphfcn_list`, this is essential if you plan to [add a `graphfcn`].
Sometimes, it doesn't matter how clever your functionality is, because your
initial graph is missing something, in which case you will need to...

### 3. Change `starting_graph`

By default SWMManywhere uses Open Street Map (OSM) data to create a starting graph,
to which subsequent `graphfcns` are applied to. A key limitation of SWMManywhere is
that the plausibility of a synthetic pipe being placed in a given location requires
links to exist in the original `starting_graph`. If you are in a location where
the quality of OSM is questionable, or perhaps you have knowledge that some pipes
exist in unusual locations, then you can provide the address to a custom graph
with the `starting_graph` entry in the `config` file. Note, for information on the
format that this graph should take, see
[`save_graph`](../reference-graph-utilities/#swmmanywhere.graph_utilities.save_graph).

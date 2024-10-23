# %% [markdown]
# # Extended Demo
# Note - this script can also be opened in interactive Python if you wanted to
# play around. On the GitHub it is in [docs/notebooks](https://github.com/ImperialCollegeLondon/SWMManywhere/blob/main/docs/notebooks/extended_demo.py)
# . To run this on your
# local machine, you will need to install the optional dependencies for `doc`:
#
# `pip install swmmanywhere[doc]`
#
# %% [markdown]
# ## Introduction
# This script demonstrates a simple use case of `swmmanywhere`, building on the
# [quickstart](https://imperialcollegelondon.github.io/SWMManywhere/quickstart/)
# example, but including plotting and alterations.
#
# Since this is a notebook, we will define [`config`](https://imperialcollegelondon.github.io/SWMManywhere/config_guide/)
# as a dictionary rather than a `yaml` file, but the same principles apply.
#
# ## Initial run
#
# Here we will run the [quickstart](https://imperialcollegelondon.github.io/SWMManywhere/config_guide/)
# configuration, keeping everything in a temporary directory.
# %%
# Imports
from __future__ import annotations

import tempfile
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

from swmmanywhere.logging import set_verbose
from swmmanywhere.swmmanywhere import swmmanywhere
from swmmanywhere.utilities import plot_map

# Create temporary directory
temp_dir = tempfile.TemporaryDirectory()
base_dir = Path(temp_dir.name)

# Define minimum viable config (with shorter duration so better inspect results)
config = {
    "base_dir": base_dir,
    "project": "my_first_swmm",
    "bbox": [1.52740, 42.50524, 1.54273, 42.51259],
    "run_settings": {"duration": 3600},
}

# Run SWMManywhere
outputs = swmmanywhere(config)

# Verify the output
model_file = outputs[0]
if not model_file.exists():
    raise FileNotFoundError(f"Model file not created: {model_file}")


# %% [markdown]
# ## Plotting output
#
# If you do not have a real UDM, the majority of your interpretation will be
# around the synthesised `nodes` and `edges`. These are
# created in the same directory as the `model_file`. Let's have a look at them.
# Note that the `outfall` that each node drains to is specified in the `outfall`
# attribute, we will plot these in red and other nodes in black with the built-
# in `swmmanywhere.utilities.plot_map` function.
# %%
# Create a folium map and add the nodes and edges
plot_map(model_file.parent)

# %% [markdown]
# OK, it's done something! Though perhaps we're not super satisfied with the output.
#
# ## Customising outputs
#
# Some things stick out on first glance,
#
# - Probably we do not need pipes in the hills to the South, these seem to be along
# pedestrian routes, which can be adjusted with the `allowable_networks` parameter.
# - We will also remove any types under the `omit_edges` entry, here you can specify
# to not allow pipes to cross bridges, tunnels, motorways, etc., however, this is
# such a small area we probably don't want to restrict things so much.
# - We have far too few outfalls, it seems implausible that so many riverside streets
# would not have outfalls. Furthermore, there are points that are quite far from the
# river that have been assigned as outfalls. We can reduce the `river_buffer_distance`
# to make nodes nearer the river more likely to be outfalls, but also reduce the
# `outfall_length` distance parameter to enable `swmmanywhere` to more freely select
# outfalls that are adjacent to the river.
#
# Let's just demonstrate that using the
# [`parameter_overrides` functionality](https://imperialcollegelondon.github.io/SWMManywhere/config_guide/#changing-parameters).
#
# %%
config["parameter_overrides"] = {
    "topology_derivation": {
        "allowable_networks": ["drive"],
        "omit_edges": ["bridge"],
    },
    "outfall_derivation": {
        "outfall_length": 5,
        "river_buffer_distance": 30,
    },
}
outputs = swmmanywhere(config)
plot_map(outputs[0].parent)

# %% [markdown]
# OK that clearly helped, although we have appear to have stranded pipes (e.g., along
# *Carrer dels Canals* in North West), presumably due to some mistake in the
# OSM specifying that it
# is connected via a pedestrian route. We won't remedy this in the tutorial, but you can
# manually provide your
# [`starting_graph`](https://imperialcollegelondon.github.io/SWMManywhere/config_guide/#change-starting_graph)
# via the configuration file to address such mistakes.
#
# More importantly we can see some distinctive unconnected network in the South West.
# What is going on there? To explain this we will have to turn on verbosity to print the
# intermediate files used in model derivation.
#
# To do this with a command line call we simply add the flag `--verbose=True`.
# Though in code we will have to use `set_verbose` from the `logging` module.

# %%
# Make verbose
set_verbose(True)  # Set verbosity

# Run again
outputs = swmmanywhere(config)
model_dir = outputs[0].parent
m = plot_map(model_dir)

# %% [markdown]
# That's a lot of information! However, the reason we are currently interested
# in this is because the files associated with
# each workflow step are saved when `verbose=True`.
#
# We will load a file called `subbasins` and add it to the map.

# %%
subbasins = gpd.read_file(model_dir / "subbasins.geoparquet")
folium.GeoJson(subbasins, fill_opacity=0, color="blue", weight=2).add_to(m)
m

# %% [markdown]
# Although this can be customised, the default behaviour of `swmmanywhere` is to not
# allow edges to cross hydrological subbasins. It is now super clear why these
# unconnected  networks have appeared, and are ultimately due to the underlying DEM.
# If you did desperately care about these streets, then you should probably
# widen your bounding box.
#
# ## Plotting results
#
# Because we have run the model with `verbose=True` we will also see that a new
# `results` file has appeared, which contains all of the simulation results from SWMM.

# %%
df = pd.read_parquet(model_dir / "results.parquet")
df.head()

# %% [markdown]
# `results` contains all simulation results in long format, with `flooding` at
# nodes and `flow` at edges. We will plot a random `flow`.

# %%
flows = df.loc[df.variable == "flow"]
flows.loc[flows.id == flows.iloc[0].id].set_index("date").value.plot(
    ylabel="flow (l/s)"
)


# %% [markdown]
# If `results` are present in the `model_dir`, `plot_map` will make clickable
# elements to view plots,
# now you can inspect your results in a much more elegant way than the SWMM GUI.
# Just click a node or link to view the flooding or flow timeseries!


# %%
plot_map(model_dir)

# %% [markdown]
# If we explore around, clicking on edges, we can see that flows are often
# looking sensible, though we can definitely some areas that have been hampered
# by our starting street graph (e.g., in the Western portion of *Carrer del Sant Andreu*
# in North West we can see negative flows meaning the direction is different from
# what the topology derivation assumed flow would be going in!).
# The first suggestion here would be to examine the starting graph,
# however, if you want to
# make more sophisticated customisations then your probably want to learn about
# [graph functions](https://imperialcollegelondon.github.io/SWMManywhere/graphfcns_guide/).

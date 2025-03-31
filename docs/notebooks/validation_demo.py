# %% [markdown]
# # Network comparison demo
# Note - this script can also be opened in interactive Python if you wanted to
# play around. On the GitHub it is in [docs/notebooks](https://github.com/ImperialCollegeLondon/SWMManywhere/blob/main/docs/notebooks/extended_demo.py)
# . To run this on your
# local machine, you will need to install the optional dependencies for `doc`:
#
# `pip install swmmanywhere[doc]`
#
# %% [markdown]
# ## Introduction
# This script demonstrates how to use `swmmanywhere` when you have a real network.
#
# Since this is a notebook, we will define [`config`](https://imperialcollegelondon.github.io/SWMManywhere/config_guide/)
# as a dictionary rather than a `yaml` file, but the same principles apply.
#
# ## Initial setup
#
# Here we will run one of the Bellinge sub-networks which is provided in the test data.
# We will keep everything in a temporary directory.
# %%
# Imports
from __future__ import annotations

import tempfile
from pathlib import Path

from swmmanywhere.logging import set_verbose
from swmmanywhere.swmmanywhere import swmmanywhere
from swmmanywhere.utilities import plot_map

# Create temporary directory
temp_dir = tempfile.TemporaryDirectory()
base_dir = Path(temp_dir.name)

bellinge_test_folder = Path(__file__).parent.parent / "tests" / "test_data" / "bellinge_small"

# %% [markdown]
# We update our config file to use new coordinates, and provide the real data. As set
# out in the [schema](reference-defs.md#schema-for-configuration-file), a variety of
# data entries can be provided to describe the real network, all CRS must be in the UTM
# CRS for the project:
#
# - `graph` (essential) - a graph file in JSON format created from [`nx.node_link_data`](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html)
# - `subcatchments` (essential) - a GeoJSON file with subcatchment outlines, with #
# headings for `id`, and `impervious_area` (only needed if calculating the metric,
# `bias_flood_depth`).
# - `inp` - an inp file to run the model with, matching the appropriate graph
# and subcatchments. 
# - `results` - a results file to compare the model with. 
#
# At least one of `results` or `inp` must be provided. If `inp` is provided then the
# model will be run and the results compared to the real data, otherwise `results` will
# be loaded directly.
# %%
# Define config
config = {
    "base_dir": base_dir,
    "project": "bellinge_small",
    "bbox":[
    10.308864591221067,
    55.332756215349825,
    10.31747911327979,
    55.33917324071062
  ],
    "run_settings": {"duration": 3600},
    "real":
    {
        "graph": bellinge_test_folder / "graph.json", 
        "inp": bellinge_test_folder / "bellinge_small.inp",
        "subcatchments": bellinge_test_folder / "subcatchments.geojson",
    }
}

# %% [markdown]
# We make the `swmmanywhere` call as normal, but can observe that there is an additional
# model run where the real model `inp` is being run.
# %%
## Run SWMManywhere
set_verbose(True) 
outputs = swmmanywhere(config)

# %% [markdown]
# We can plot the real network data and simulation
# %%
## View real data
plot_map(bellinge_test_folder)

# %% [markdown]
# ... and we can plot the synthesised network
# %%
## View output
model_file = outputs[0]
plot_map(model_file.parent)

# %% [markdown]
# .. but of course we can see that the two networks do not perfectly line up. So we
# can't exactly plot our timeseries side-by-side. To quantify this properly we need to
# draw on SWMManywhere's ability to compare two networks that don't line up, which is
# done using the `metrics` output, automatically calculated when `real` data is
# provided.
# %%
# View metrics
metrics = outputs[1]
print(metrics)

# %% [markdown]
# For more information on using metrics see our [metrics guide](metrics_guide.md). To
# understand how to make use of such information, see our [paper](https://doi.org/10.1016/j.envsoft.2025.106358)
# for example.
#
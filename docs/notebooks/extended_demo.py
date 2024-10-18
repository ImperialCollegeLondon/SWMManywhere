"""A demo."""

# %% [markdown]
# # Extended Demo
# Note - this script can also be opened in interactive Python if you wanted to
# play around. On the GitHub it is in [docs/notebooks]. To run this on your
# local machine, you will need to install the optional dependencies for `doc`:
#
# `pip install swmmanywhere[doc]`
#
# %% [markdown]
# ## Introduction
# This script demonstrates a simple use case of `SWMManywhere`, building on the
# [quickstart](quickstart.md) example, but including plotting and alterations.
#
# Since this is a notebook, we will define [`config`](config_guide.md) as a
# dictionary rather than a `yaml` file, but the same principles apply.
#
# ## Initial run
#
# Here we will run the [quickstart](quickstart.md) configuration, keeping
# everything in a temporary directory.
# %%
# Imports
from __future__ import annotations

import tempfile
from pathlib import Path

import folium
import geopandas as gpd

from swmmanywhere.swmmanywhere import swmmanywhere

# Create temporary directory
temp_dir = tempfile.TemporaryDirectory()
base_dir = Path(temp_dir.name)

# Define minimum viable config
config = {
    "base_dir": base_dir,
    "project": "my_first_swmm",
    "bbox": [1.52740, 42.50524, 1.54273, 42.51259],
    "address_overrides": {
        "elevation": Path(
            r"C:\Users\bdobson\Downloads\test\my_first_swmm\bbox_1\download\elevation.tif"
        )
    },
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
# created in the same directory as the `model_file``.
#
# Let's have a look at these
# %%

# Load and inspect results
nodes = gpd.read_file(model_file.parent / "nodes.geoparquet")
edges = gpd.read_file(model_file.parent / "edges.geoparquet")

# Convert to EPSG 4326 for plotting
nodes = nodes.to_crs(4326)
edges = edges.to_crs(4326)

# Create a folium map and add the nodes and edges
m = folium.Map(location=[nodes.y.mean(), nodes.x.mean()], zoom_start=16)
folium.GeoJson(nodes).add_to(m)
folium.GeoJson(edges).add_to(m)

# Display the map
m

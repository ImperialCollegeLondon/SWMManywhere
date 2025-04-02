# %% [markdown]
# # Custom data demo
# Note - this script can also be opened in interactive Python if you wanted to
# play around. On the GitHub it is in [docs/notebooks](https://github.com/ImperialCollegeLondon/SWMManywhere/blob/main/docs/notebooks/extended_demo.py)
# . To run this on your
# local machine, you will need to install the optional dependencies for `doc`:
#
# `pip install swmmanywhere[doc]`
#
# %% [markdown]
# ## Introduction
# This script demonstrates how to use `swmmanywhere` when you have custom data.
#
# Since this is a notebook, we will define [`config`](https://imperialcollegelondon.github.io/SWMManywhere/config_guide/)
# as a dictionary rather than a `yaml` file, but the same principles apply.
#
# ## Initial setup
#
# We will use the same example as the [extended demo](extended_demo.py), but with a 
# custom elevation dataset. Let's start by rerunning it.
# %%
# Imports
from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from swmmanywhere.logging import set_verbose
from swmmanywhere.swmmanywhere import swmmanywhere
from swmmanywhere.utilities import plot_map

# Create temporary directory
temp_dir = tempfile.TemporaryDirectory()
base_dir = Path(temp_dir.name)

# Define minimum viable config
bbox = [1.52740, 42.50524, 1.54273, 42.51259]
config = {
    "base_dir": base_dir,
    "project": "my_first_swmm",
    "bbox": bbox,
    "run_settings": {"duration": 3600},
    "parameter_overrides": {
        "topology_derivation": {
            "allowable_networks": ["drive"],
            "omit_edges": ["bridge"],
        },
        "outfall_derivation": {
            "outfall_length": 5,
            "river_buffer_distance": 30,
        },
    },
}
set_verbose(True)  # Set verbosity

# Run SWMManywhere
outputs = swmmanywhere(config)
model_dir = outputs[0].parent

# %%
plot_map(model_dir)

# %% [markdown]
# ## Supply custom elevation data
#
# To keep things simple, we will just download some elevation data for the same area
# and perturb it, though in practice you are likely to use some higher resolution
# or more accurate data.
#
# %%

# Import NASADEM downloader and reprojection tools
from swmmanywhere.prepare_data import download_elevation
from swmmanywhere.geospatial_utilities import reproject_raster, get_utm_epsg

# Download and reproject the correct elevation to UTM
download_elevation(base_dir / "elevation.tif", config["bbox"])
reproject_raster(
    get_utm_epsg(bbox[0],bbox[1])
    base_dir / "elevation.tif",
    base_dir / "elevation_utm.tif",
)

# Flip it
import rasterio
import numpy as np

with rasterio.open(base_dir / "elevation_utm.tif") as src:
    data = np.fliplr(src.read(1))
    with rasterio.open(base_dir / "fake_elevation.tif", "w", **src.profile) as dst:
        dst.write(data, 1)

# %% [markdown]
# ## Update config and run again
#
# Now we update the `elevation` entry in the `address_overrides` part of the
# `config` to point to the new elevation data, then rerun `swmmanywhere`.
# %%
# Update config
config["address_overrides"] = {
    "elevation": str(base_dir / "fake_elevation.tif"),
}

# Run again
outputs = swmmanywhere(config)
model_dir = outputs[0].parent

# %%
plot_map(model_dir)


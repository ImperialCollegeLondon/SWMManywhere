# %% [markdown]
# # Custom data demo
# Note - this script can also be opened in interactive Python if you wanted to
# play around. On the GitHub it is in [docs/notebooks](./notebooks/custom_data_demo.py).
# To run this on your local machine, you will need to install the optional dependencies
# for `doc`:
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
# We will use the same example as the [extended demo](./notebooks/extended_demo.py), but
# with a custom elevation dataset. Let's start by rerunning it.
# %%
# Imports
from __future__ import annotations

import tempfile
from pathlib import Path

import folium
import geopandas as gpd

from swmmanywhere.logging import set_verbose
from swmmanywhere.swmmanywhere import swmmanywhere
from swmmanywhere.utilities import plot_map

# Create temporary directory
temp_dir = tempfile.TemporaryDirectory(dir=".")
base_dir = Path(temp_dir.name)

# Define minimum viable config
bbox = (1.52740, 42.50524, 1.54273, 42.51259)
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

# %% [markdown]
# ## Plotting output
#
# Now we can plot the output. To highlight the differences in the supplied data that we
# are about to demonstrate, we also plot the subbasins.
# %%
m = plot_map(model_dir)
subbasins = gpd.read_file(model_dir / "subbasins.geoparquet")
folium.GeoJson(subbasins, fill_opacity=0, color="blue", weight=2).add_to(m)
m


# %% [markdown]
# ## Supply custom elevation data
#
# To keep things simple, we will just download some elevation data for the same area
# and perturb it, though in practice you are likely to use some higher resolution
# or more accurate data.
#
# You don't need to worry about your files lining up perfectly (though of course if they
# do not overlap at all then you will run into problems).
# %%

# Import NASADEM downloader and reprojection tools
from swmmanywhere.geospatial_utilities import (  # noqa: E402
    get_utm_epsg,
    reproject_raster,
)
from swmmanywhere.prepare_data import download_elevation  # noqa: E402

# Download and reproject the correct elevation to UTM
download_elevation(base_dir / "elevation.tif", bbox)
reproject_raster(
    get_utm_epsg(bbox[0], bbox[1]),
    base_dir / "elevation.tif",
    base_dir / "elevation_utm.tif",
)

# Flip it
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

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

# %% [markdown]
# ## Plotting output
#
# This time we will include both the original (blue) and the new (red) subbasins to
# highlight the impact of flipping the elevation data.
# %%
m = plot_map(model_dir)
subbasins_new = gpd.read_file(model_dir / "subbasins.geoparquet")
folium.GeoJson(subbasins_new, fill_opacity=0, color="red", weight=2).add_to(m)
folium.GeoJson(subbasins, fill_opacity=0, color="blue", weight=2).add_to(m)
m

# %% [markdown]
#
# ## In Summary
#
# This tutorial has demonstrated how we can supply our own data in place of the existing
# files that are downloaded by SWMManywhere. This method works for any
# [filepath](https://imperialcollegelondon.github.io/SWMManywhere/reference-filepaths/),
# which are the files used by the current workflow.
#
# You can also use `address_overrides` to you add your own custom data that
# might be used by a [custom graphfcn](https://imperialcollegelondon.github.io/SWMManywhere/graphfcns_guide/)
# , for example. If you are adding new data to the SWMManywhere workflow in this way,
# you will have to access it from the `FilePaths` object with the `get_path` method.
#

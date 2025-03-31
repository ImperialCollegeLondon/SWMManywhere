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
# ## Initial run
#
# Here we will run one of the Bellinge sub-networks, 
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

# rbellinge_test_folder = Path(__file__).parent.parent / "tests" / "test_data" / "bellinge_small"
bellinge_test_folder = Path(r"C:\Users\bdobson\Documents\GitHub\SWMManywhere\tests\test_data\bellinge_small")

# Define minimum viable config (with shorter duration so better inspect results)
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

# Run SWMManywhere
set_verbose(True) 
outputs = swmmanywhere(config)


# View real data
plot_map(bellinge_test_folder)

# View output
model_file = outputs[0]
plot_map(model_file.parent)

# View metrics
metrics = outputs[1]

print(metrics)

# %%

# %%

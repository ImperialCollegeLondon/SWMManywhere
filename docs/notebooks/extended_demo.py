# %%
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

import base64
import tempfile
from io import BytesIO
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt

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
# created in the same directory as the `model_file`. Let's have a look at them.
# %%
# Create a folium map and add the nodes and edges
def basic_map(model_dir):
    # Load and inspect results
    nodes = gpd.read_file(model_dir / "nodes.geoparquet")
    edges = gpd.read_file(model_dir / "edges.geoparquet")
    

    # Convert to EPSG 4326 for plotting
    nodes = nodes.to_crs(4326)
    edges = edges.to_crs(4326)
    
    m = folium.Map(location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()], zoom_start=16)
    folium.GeoJson(edges, color='black',weight=1).add_to(m)
    folium.GeoJson(nodes, marker=folium.CircleMarker(radius = 3, # Radius in metres
                                               weight = 0, #outline weight
                                               fill_color = 'black', 
                                               fill_opacity = 1)).add_to(m)
    
    # Display the map
    return m

basic_map(model_file.parent)

# %% [markdown]
# OK, it's done something! Though perhaps we're not super satisfied with the output.
#
# ## Customising outputs
#
# Some things stick out on first glance:
# - Probably we do not need pipes in the hills to the South, these seem to be along pedestrian routes, which can be adjusted with the `allowable_networks` parameter.
# - We will also remove any types under the `omit_edges` entry, here you can specify to not allow pipes to cross bridges, tunnels, motorways, etc., however, this is such a small area we probably don't want to restrict things so much.
# - The density of points seems a bit extreme, ultimately we'd like to walk around and figure out where the manholes are, but for now we can reduce density by increasing `node_merge_distance`.
# Let's just demonstrate that using the [`parameter_overrides` functionality](config_guide.md/#changing-parameters).

# %%
config['parameter_overrides'] = {'topology_derivation' : {'allowable_networks' : ['drive'], 'omit_edges' : []}, 'subcatchment_derivation' : {'node_merge_distance':15 }}
outputs = swmmanywhere(config)
basic_map(outputs[0].parent)

# %% [markdown]
# OK that clearly helped, although we have appear to have stranded pipes along *Carrer de la Grella*, presumably due to some mistake in the OSM specifying that it is connected via a pedestrian route. We won't remedy this in the tutorial, but you can manually provide your [`starting_graph`](config_guide.md/#change-starting_graph) via the configuration file to address such mitakes. 
#
# More importantly we can see some distinctive unconnected network in the South West. What is going on there? To explain this we will have to turn on verbosity to print the intermediate files used in model derivation.
#
# To do this with a command line call we simply add the flag `--verbose=True`.

# %%
# Make verbose
from swmmanywhere import logging
logging.set_verbose(True) # Set verbosity

# Run again
outputs = swmmanywhere(config)
model_dir = outputs[0].parent
m = basic_map(model_dir)

# %% [markdown]
# OK that's a lot of information! We can see `swmmanywhere` iterating through the various graph functions and a variety of other messages. However, the reason we are currently interested in this is because the files associated with each step are saved when `verbose=True`. 
#
# We will load a file called `subbasins` and add it to the map.

# %%
subbasins = gpd.read_file(model_dir / "subbasins.geoparquet")
folium.GeoJson(subbasins,fill_opacity=0, color='blue',weight=2).add_to(m)
m

# %% [markdown]
# Although this can be customised, the default behaviour of `swmmanywhere` is to not allow edges to cross hydrological subbasins. It is now super clear why these unconnected networks have appeared, and are ultimately due to the underlying DEM. If you did desperately care about these streets, then you should probably widen your bounding box.
#
# ## Plotting results
#
# Because we have run the model with `verbose=True` we will also see that a new `results` file has appeared, which contains all of the simulation results from SWMM.

# %%
df = pd.read_parquet(model_dir / "results.parquet")
df.head()

# %% [markdown]
# `results` contains all simulation results in long format, with `flooding` at nodes and `flow` at edges. We will plot a random `flow`.

# %%
floods = df.loc[df.variable == 'flooding']
flows = df.loc[df.variable == 'flow']
flows.loc[flows.id == flows.iloc[0].id].set_index('date').value.plot(ylabel='flow (m3/s)')


# %% [markdown]
# Since folium is super clever, we can make these clickable on our map - and now you can inspect your results in a much more elegant way than the SWMM GUI.

# %%
# Create a folium map and add the nodes and edges
def clickable_map(model_dir):
    # Load and inspect results
    nodes = gpd.read_file(model_dir / "nodes.geoparquet")
    edges = gpd.read_file(model_dir / "edges.geoparquet")
    df = pd.read_parquet(model_dir / "results.parquet")
    df.id = df.id.astype(str)
    floods = df.loc[df.variable == 'flooding'].groupby('id')
    flows = df.loc[df.variable == 'flow'].groupby('id')


    # Convert to EPSG 4326 for plotting
    nodes = nodes.to_crs(4326).set_index('id')
    edges = edges.to_crs(4326).set_index('id')

    # Create map
    m = folium.Map(location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()], zoom_start=16)

    # Add nodes
    for node, row in nodes.iterrows():
        grp = floods.get_group(str(node))
        grp.set_index('date').value.plot(ylabel='flooding (m3)', title = node)
        img = BytesIO()
        f = plt.gcf()
        f.savefig(img, format="png",dpi=94)
        plt.close(f)
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode()
        img_html = f'<img src="data:image/png;base64,{img_base64}">'
        folium.CircleMarker(
            [nodes.loc[node].geometry.y, nodes.loc[node].geometry.x],
            color="black",
            radius=3,
            weight=0,
            fill_color='black',
            fill_opacity=1,
            popup=folium.Popup(img_html, max_width=450),
        ).add_to(m)

    # Add edges
    for edge, row in edges.iterrows():
        grp = flows.get_group(str(edge))
        grp.set_index('date').value.plot(ylabel='flow (m3/s)', title = edge)
        img = BytesIO()
        f = plt.gcf()
        f.savefig(img, format="png",dpi=94)
        plt.close(f)
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode()
        img_html = f'<img src="data:image/png;base64,{img_base64}">'
        folium.PolyLine(
            [[c[1],c[0]] for c in row.geometry.coords],
            color="black",
            weight=2,
            popup=folium.Popup(img_html, max_width=450),
        ).add_to(m)
    return m

clickable_map(model_dir)

# %% [markdown]
# If we explore around, clicking on edges, we can see that flows are often looking sensible, though we can definitely some areas that have been hampered by our starting street graph (e.g., *Carrer dels Canals*). The first suggestion here would be to widen your bounding box, however, if you want to make more sophisticated customisations then your probably want to learn about [graph functions](graphfcns_guide.md)

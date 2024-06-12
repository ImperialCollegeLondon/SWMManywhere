# %% [markdown]
# # Quickstart
# Note - this script can also be opened in interactive Python if you wanted to
# play around. On the GitHub it is in [docs/demo/scripts]()
#
# %%
# Import modules
from __future__ import annotations

import os
from pathlib import Path
from pprint import pprint as print

import pandas as pd
import yaml

from swmmanywhere import swmmanywhere

# Make a base directory
base_dir = Path.cwd() / 'swmmanywhere_models'
print(base_dir)
base_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
## API keys
# ... Information here about how to handle API keys
# TODO
# %%
api_keys = {'nasadem_key' : 'b206e65629ac0e53d599e43438560d28'}
with (base_dir / 'api_keys.yml').open('w') as f:
    yaml.dump(api_keys, f)

# %% [markdown]
## Define configuration file
# The standard use of SWMManywhere is achieved via a `configuration` file. This
# file defines a variety of parameters to use during the creation and running 
# of the synthetic SWMM model.
# %%
# Load the default config file
config = swmmanywhere.load_config(validation = False)
print(config)

# %% [markdown]
# The configuration file is a dictionary that contains a variety of parameters
# that are used to define the synthetic SWMM model and run. We won't go through
# all of them here, but we will explore some key ones in this demo.

# %%
# Update the information in it to match our current demo

# A real base directory is needed to store data
config['base_dir'] = base_dir

# A project name will be a folder in the base directory for this demo
config['project'] = 'my_first_swmm'

# The bounding box is a list of four values: [minx, miny, maxx, maxy], we will
# pick one in Andorra because the area is small and we can download the building
# data for it very quickly (larger countries will take longer)
config['bbox'] = [1.52740,42.50524,1.54273,42.51259]

# We need to locate the API keys file
config['api_keys'] = base_dir / 'api_keys.yml'

# The precipitation downloader is currently broken so we will just use the 
# design storm
config['address_overrides'] = {'precipitation' : 
                    Path(swmmanywhere.__file__).parent / 'defs' / 'storm.dat'}

# If you do not know the outline of your urban drainage network (as in this case)
# you will have to run the `clip_to_catchments` graph function. T
config['parameter_overrides'] = {'subcatchment_derivation' : 
                                 {'subbasin_streamorder' : 5}
                                 }

# We do not have a real SWMM model for this so we will delete that entry
del config['real']

# %% [markdown]
## Run SWMManywhere
# The `swmmanywhere` function is the main function that generates a synthetic
# SWMM model and runs it.
#
# We will turn on the verbose mode so we can see what is happening.

# %%
# Run SWMManywhere
os.environ["SWMMANYWHERE_VERBOSE"] = "true"
inp, _ = swmmanywhere.swmmanywhere(config)
print(f'Created SWMM model at: {inp}')

# %% [markdown]
# OK so we have run the model and we have an `inp` file. This is the SWMM model
# that we have created and can be loaded by the SWMM software. Here is a 
# screenshot of this generated run.
# 
# ![SWMM Model](../../images/andorra_swmm_screenshot.png)
# 
# The `swmmanywhere` function has also run a simulation, the results can be 
# found in the same folder as the input file, by default they are saved in a 
# `parquet` format, let's load them and plot something.

# %%
results = pd.read_parquet(inp.parent / 'results.parquet')
results.sample(10)
# %% [markdown]
# The results are a DataFrame that contains the results of the simulation, the 
# data is in narrow format, so each row is a simulation of a `variable` at a
# particular `date` for a particular `id` (i.e., manhole or pipe). 
# %%
gb = results.groupby(['variable','id'])
gb.get_group(('flow', results.iloc[-1].id)).plot(x='date', 
                                                 y='value',
                                                 title='Flow in a pipe',
                                                 xlabel='Date',
                                                 ylabel='Flow (l/s)'
                                                 )
# %% [markdown]
# Hooray! We have run a synthetic SWMM model and plotted some results. This is
# the end of the quickstart guide.
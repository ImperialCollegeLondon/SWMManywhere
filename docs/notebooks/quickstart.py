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
# pick one in Monaco because the area is small and we can download the building
# data for it very quickly (larger countries will take longer)
config['bbox'] = [7.41680,43.72647,7.43150,43.73578]

# We need to locate the API keys file
config['api_keys'] = base_dir / 'api_keys.yml'

# The precipitation downloader is currently broken so we will just use the 
# design storm
config['address_overrides'] = {'precipitation' : base_dir.parent.parent.parent /\
                                    'swmmanywhere' / 'defs' / 'storm.dat'}

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

# %%

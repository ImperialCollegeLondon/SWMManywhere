# %% [markdown]
# # Quickstart
# Note - this script can also be opened in interactive Python if you wanted to
# play around. On the GitHub it is in [docs/demo/scripts]()
#
# %%
# Import modules
from pathlib import Path

# Make a base directory
base_dir = Path.cwd() / 'swmmanywhere_models'
print(base_dir)
base_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
## API keys
# ... Information here about how to handle API keys
# TODO
# %% [markdown]
## Define configuration file
# The standard use of SWMManywhere is achieved via a `configuration` file. This
# file defines a variety of parameters to use during the creation and running 
# of the synthetic SWMM model.
# %%
# Make a config file
config = {'base_dir' : base_dir,
          'project' : 'my_first_swmm',
          'bbox' : [0.05428,51.55847,0.07193,51.56726],
          'api_keys' : base_dir / 'api_keys.yml',
          'graphfcn_list' : ['assign_id']
          }
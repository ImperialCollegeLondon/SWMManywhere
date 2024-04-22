"""Perform sensitivity analysis on the results of the model runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from SALib import ProblemSpec
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
from tqdm import tqdm

from swmmanywhere.logging import logger
from swmmanywhere.paper import experimenter
from swmmanywhere.paper import plotting as swplt
from swmmanywhere.preprocessing import check_bboxes
from swmmanywhere.swmmanywhere import load_config

# %% [markdown]
# ## Initialise directories and load results
# %%
# Load the configuration file and extract relevant data
project = 'cranbrook'
base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')
config_path = base_dir / project / f'{project}_hpc.yml'
config = load_config(config_path, validation = False)
config['base_dir'] = base_dir / project
objectives = config['metric_list']
parameters = config['parameters_to_sample']

# Load the results
bbox = check_bboxes(config['bbox'], config['base_dir'])
results_dir = config['base_dir'] / f'bbox_{bbox}' / 'results'
fids = list(results_dir.glob('*_metrics.csv'))
dfs = [pd.read_csv(fid) for fid in tqdm(fids, total = len(fids))]

# Calculate how many processors were used
nprocs = len(fids)

# Concatenate the results
df = pd.concat(dfs)

# Log deltacon0 because it can be extremely large
df['nc_deltacon0'] = np.log(df['nc_deltacon0'])
df = df.sort_values(by = 'iter')

# Make a directory to store plots in
plot_fid = results_dir / 'plots'
plot_fid.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Plot the objectives
# %%
# Highlight the behavioural indices 
# (i.e., KGE, NSE, PBIAS are in some preferred range)
behavioral_indices = swplt.create_behavioral_indices(df)

# Plot the objectives
swplt.plot_objectives(df, 
                        parameters, 
                        objectives, 
                        behavioral_indices,
                        plot_fid)

# %% [markdown]
# ## Perform Sensitivity Analysis
# %%

# Formulate the SALib problem
problem = experimenter.formulate_salib_problem(parameters)

# Calculate any missing samples
n_ideal = pd.DataFrame(
    experimenter.generate_samples(parameters_to_select=parameters,
    N=2**config['sample_magnitude'])
    ).iter.nunique()
missing_iters = set(range(n_ideal)).difference(df.iter)
if missing_iters:
    logger.warning(f"Missing {len(missing_iters)} iterations")

# Perform the sensitivity analysis for groups
problem['outputs'] = objectives
rg = {objective: sobol.analyze(problem, 
                    df[objective].iloc[0:
                                        (2**(config['sample_magnitude'] + 1) * 10)]
                                        .values,
                    print_to_console=False) 
                    for objective in objectives}

# Perform the sensitivity analysis for parameters
problemi = problem.copy()
del problemi['groups']
ri = {objective: sobol.analyze(problemi, 
                    df[objective].values,
                    print_to_console=False) 
                    for objective in objectives}



for r_, groups in zip([rg,ri],  ['groups','parameters']):
    f,axs = plt.subplots(len(objectives),1,figsize=(10,10))
    for ix, ax, (objective, r) in zip(range(len(objectives)), axs, r_.items()):
        total, first, second = r.to_df()
        total['sp'] = (total['ST'] - first['S1'])
        barplot(total,ax=ax)
        if ix == 0:
            ax.set_title('Total - First')
        if ix != len(objectives) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([x.replace('_','\n') for x in total.index], 
                                    rotation = 0)
            
        ax.set_ylabel(objective,rotation = 0,labelpad=20)
        ax.get_legend().remove()
    f.tight_layout()
    f.savefig(plot_fid / f'{groups}_indices.png')  
    plt.close(f)

f,axs = plt.subplots(2,len(ri),figsize=(8,8))
for r_, axs_ in zip([rg, ri],axs):
    for (objective, r), ax in zip(r_.items(),axs_):
        total, first, second = r.to_df()
        barplot(total,ax=ax)
        ax.set_title(objective)
plt.tight_layout()
f.savefig(plot_fid / 'overall_indices.png')


sp = ProblemSpec(problemi)
sp.samples = df[parameters].values
sp.results = df[objectives].values
sp.analyze_sobol(print_to_console=False,
                    calc_second_order=True)
for objective in objectives:
    ax = sp.heatmap(objective)
    f = ax.get_figure()
    f.savefig(plot_fid / f'heatmap_{objective}.png')

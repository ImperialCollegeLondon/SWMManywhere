"""Perform sensitivity analysis on the results of the model runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from tqdm import tqdm

from swmmanywhere.custom_logging import logger
from swmmanywhere.paper import experimenter
from swmmanywhere.paper import plotting as swplt
from swmmanywhere.preprocessing import check_bboxes
from swmmanywhere.swmmanywhere import load_config

# %% [markdown]
# ## Initialise directories and load results
# %%
# Load the configuration file and extract relevant data
if __name__ == 'main':
    project = 'bellinge_formatted'
    base_dir = Path.home() / "Documents" / "data" / "swmmanywhere"
    config_path = base_dir / project / 'bf_hpc.yml'
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
    plot_fid = results_dir.parent / 'plots'
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
                (2**(config['sample_magnitude'] + 1) *\
                  (len(set(problem['groups']))+1)*2)]
                .values,
            print_to_console=False
            ) 
                        for objective in objectives}

    # Perform the sensitivity analysis for parameters
    problemi = problem.copy()
    del problemi['groups']
    ri = {objective: sobol.analyze(problemi, 
                        df[objective].values,
                        print_to_console=False) 
                        for objective in objectives}

    # Barplot of sensitvitiy indices
    for r_, groups in zip([rg,ri],  ['groups','parameters']):
        swplt.plot_sensitivity_indices(r_, 
                                     objectives, 
                                     plot_fid / f'{groups}_indices.png')
        
    for r_, groups in zip([rg,ri],  ['groups','parameters']):
        swplt.heatmaps(r_, plot_fid / f'heatmap_{groups}_indices.png')
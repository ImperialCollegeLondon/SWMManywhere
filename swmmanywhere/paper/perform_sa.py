# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from SALib.analyze import sobol
from tqdm import tqdm

from swmmanywhere.paper.experimenter import formulate_salib_problem
from swmmanywhere.preprocessing import check_bboxes
from swmmanywhere.swmmanywhere import load_config

if __name__ == 'main':
    project = 'bellinge'
    base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')
    config_path = base_dir / project / f'{project}_hpc.yml'
    config = load_config(config_path, validation = False)
    config['base_dir'] = base_dir / project


    bbox = check_bboxes(config['bbox'], config['base_dir'])
    results_dir = config['base_dir'] / f'bbox_{bbox}' / 'results'

    # Load the results
    fids = list(results_dir.glob('*_metrics.csv'))

    df = [pd.read_csv(fid) for fid in tqdm(fids, total = len(fids))]
    df = pd.concat(df)

    df = df.sort_values(by = 'iter')
    
    objectives = config['metric_list']

    parameters = config['parameters_to_sample']

    plot_fid = results_dir / 'plots'
    plot_fid.mkdir(exist_ok=True, parents=True)

    # Plots
    for parameter in parameters:
        f,axs = plt.subplots(int(len(objectives)**0.5),
                             int(len(objectives)**0.5),
                             figsize = (10,10))
        for ax, objective in zip(axs.reshape(-1),objectives):
            ax.scatter(df[parameter], df[objective],s=0.5)
            ax.set_yscale('symlog')
            ax.set_title(objective)
        f.tight_layout()
        f.suptitle(parameter)
        f.savefig(plot_fid / (parameter + '.png'))

    # Overall indices
    problem = formulate_salib_problem(parameters)
    problem['outputs'] = objectives
    rg = {objective: sobol.analyze(problem, 
                       df[objective].iloc[0:
                                          (2**(config['sample_magnitude'] + 1) * 10)]
                                          .values,
                       print_to_console=False) 
                       for objective in objectives}

    problemi = problem.copy()
    del problemi['groups']
    ri = {objective: sobol.analyze(problemi, 
                        df[objective].values,
                        print_to_console=False) 
                        for objective in objectives}


    from SALib.plotting.bar import plot as barplot
    f,axs = plt.subplots(2,len(ri),figsize=(8,8))
    for r_, axs_ in zip([rg, ri],axs):
        for (objective, r), ax in zip(r_.items(),axs_):
            total, first, second = r.to_df()
            barplot(total,ax=ax)
            ax.set_title(objective)
    plt.tight_layout()
    f.savefig(plot_fid / 'overall_indices.png')

    from SALib import ProblemSpec
    sp = ProblemSpec(problemi)
    sp.samples = df[parameters].values
    sp.results = df[objectives].values
    sp.analyze_sobol(print_to_console=False,
                     calc_second_order=True)
    for objective in objectives:
        ax = sp.heatmap(objective)
        f = ax.get_figure()
        f.savefig(plot_fid / f'heatmap_{objective}.png')
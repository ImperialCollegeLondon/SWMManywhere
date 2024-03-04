# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
from glob import glob
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from SALib.analyze import sobol

from swmmanywhere.paper.experimenter import formulate_salib_problem

if __name__ == 'main':
    base_dir = Path(r"""C:\Users\bdobson\Documents\data\swmmanywhere
                    \bellinge\bbox_1\results""")

    # Load the results
    fids = glob(str(base_dir / '*_flooding.csv'))

    df = []
    for fid in fids:
        df.append(pd.read_csv(fid))
    df = pd.concat(df).drop(columns = 'Unnamed: 0')
    df = df.sort_values(by = 'iter')
    objectives = ['pbias', 'maxflow']

    parameters = df.columns.difference(objectives + ['iter']).tolist()
    #NB HAS TO BE IN THE SAME ORDER AS IN EXPERIMENTER

    plot_fid = base_dir / 'plots'
    plot_fid.mkdir(exist_ok=True, parents=True)

    # Plots
    for parameter in parameters:
        f,axs = plt.subplots(1,len(objectives))
        for ax, objective in zip(axs,objectives):
            ax.scatter(df[parameter], df[objective],s=0.5)
            ax.set_yscale('symlog')
            ax.set_title(f'{objective}-{parameter}')
        f.savefig(plot_fid / (parameter + '.png'))

    # 
    nx = 1
    while (2 ** nx) * 10 < df.shape[0]:
        nx += 1

    # Overall indices
    problem = formulate_salib_problem(parameters)
    problem['outputs'] = objectives
    rg = {objective: sobol.analyze(problem, 
                       df[objective].iloc[0:(2**(nx-1) * 10)].values,
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
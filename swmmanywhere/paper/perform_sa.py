# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from SALib.analyze import sobol
from tqdm import tqdm

from swmmanywhere.paper.experimenter import formulate_salib_problem
from swmmanywhere.preprocessing import check_bboxes
from swmmanywhere.swmmanywhere import load_config

if __name__ == 'main':
    project = 'cranbrook'
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
    df['nc_deltacon0'] = np.log(df['nc_deltacon0'])
    df = df.sort_values(by = 'iter')
    
    # Calc missing
    set([x % 1200 for x in set(range(7168)).difference(df.iter)])

    objectives = config['metric_list']

    parameters = config['parameters_to_sample']

    plot_fid = results_dir / 'plots'
    plot_fid.mkdir(exist_ok=True, parents=True)

    # Plots
    """
    metric_groups = {'design' : ['outlet_kstest_diameters',
                                 'outlet_pbias_length',
                                 'outlet_pbias_nmanholes',
                                 'outlet_pbias_npipes'
                                 ],
                     'simulation' : ['bias_flood_depth',
                                     'grid_nse_flooding',
                                     'outlet_nse_flooding',
                                     'outlet_nse_flow',
                                     'subcatchment_nse_flooding'
                                     ],
                     'topology' : ['kstest_betweenness',
                                   'kstest_edge_betweenness',
                                   'nc_adjacency_dist',
                                   'nc_deltacon0',
                                   'nc_laplacian_dist',
                                   'nc_laplacian_norm_dist',
                                   'nc_vertex_edge_distance'
                                   ]}
    for parameter in parameters:
        behavioural_ind1 = (df.loc[:,df.columns.str.contains('nse')] > 0).any(axis=1)
        behavioural_ind2 = (df.loc[:,df.columns.str.contains('pbias')].abs() < \
            0.1).any(axis=1)

        for group, gmetrics in metric_groups.items():
            f, axs = plt.subplots(len(gmetrics),1,figsize=(5,10))
            for ax, objective in zip(axs, gmetrics):
                ax.scatter(df[parameter], df[objective],s=0.5,c='b')
                ax.scatter(df.loc[behavioural_ind2,parameter],
                        df.loc[behavioural_ind2,objective],
                        s=2,
                        c = 'c')
                ax.scatter(df.loc[behavioural_ind1,parameter],
                            df.loc[behavioural_ind1,objective],
                            s=2,
                            c = 'r')
                if 'pbias' in objective:
                    ax.plot([df[parameter].min(),df[parameter].max()],
                            [-0.1,-0.1],
                            'k--')
                    ax.plot([df[parameter].min(),df[parameter].max()],
                            [0.1,0.1],
                            'k--')
                if 'nse' in objective:
                    ax.plot([df[parameter].min(),df[parameter].max()],
                            [0,0],
                            'k--')
                ax.set_yscale('symlog')
                if not df[objective].isna().all():
                    ax.set_yticks(df[objective].quantile(
                        [0.01, 0.25,0.5,0.75,0.99]))
                    ax.set_yticklabels(np.round(df[objective].quantile(
                        [0.01, 0.25,0.5,0.75,0.99]),
                                            2))
                    if ax == axs[-1]:
                        ax.set_xlabel(parameter)
                    elif ax == axs[0]:
                        ax.set_title(group)
                    else:
                        ax.set_xticklabels([])

                    ax.set_ylabel(objective.replace('_','\n'))
                    ax.grid(True)
                if 'nc_deltacon0' == objective:
                    objective = 'log(nc_deltacon0)'
            f.tight_layout()
            f.savefig(plot_fid / f'{group}_{parameter}.png')
            f.close()
    """
    n_y_ticks = 4
    for parameter in parameters:
        f,axs = plt.subplots(int(len(objectives)**0.5),
                             int(len(objectives)**0.5),
                             figsize = (10,10))
        behavioural_ind1 = ((df.loc[:,df.columns.str.contains('nse')] > 0) &\
                             (df.loc[:,df.columns.str.contains('nse')] < 1)
                             ).any(axis=1)
        behavioural_ind2 = (df.loc[:,df.columns.str.contains('bias')].abs() <\
                             0.1).any(axis=1)

        for ax, objective in zip(axs.reshape(-1),objectives):
            ax.scatter(df[parameter], df[objective],s=0.5,c='b')
            
            ax.scatter(df.loc[behavioural_ind2,parameter],
                       df.loc[behavioural_ind2,objective],
                       s=2,
                       c = 'c')
            ax.scatter(df.loc[behavioural_ind1,parameter],
                        df.loc[behavioural_ind1,objective],
                        s=2,
                        c = 'r')
            
            if 'bias' in objective:
                ax.plot([df[parameter].min(),df[parameter].max()],
                        [-0.1,-0.1],
                        'k--')
                ax.plot([df[parameter].min(),df[parameter].max()],
                        [0.1,0.1],
                        'k--')
            if 'nse' in objective:
                ax.plot([df[parameter].min(),df[parameter].max()],
                        [0,0],
                        'k--')
            ax.set_yscale('symlog')
            print(df.iloc[df['nc_vertex_edge_distance'].argmin()])
            if not df[objective].isna().all():
                #ax.set_yticks(df[objective].quantile([0.01, 0.25,0.5,0.75,0.99]))
                #ax.set_yticklabels(np.round(df[objective].quantile(
                # [0.01, 0.25,0.5,0.75,0.99]),
                #                        2))
                ax.grid(True)
            if 'nc_deltacon0' == objective:
                objective = 'log(nc_deltacon0)'
            if 'nse' in objective:
                ax.set_ylim([-10,1])
            ax.set_title(objective)
        f.tight_layout()
        f.suptitle(parameter)
        f.savefig(plot_fid / (parameter.replace('_','-') + '.png'), dpi = 500)
        plt.close(f)

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
    """
    for r_, groups in zip([rg,ri],  ['groups','parameters']):
        f,axs = plt.subplots(len(objectives),2,figsize=(10,10))
        for ix, axs_, (objective, r) in zip(range(len(objectives)), axs, r_.items()):
            total, first, second = r.to_df()
            
            barplot(total,ax=axs_[0])
            barplot(first,ax=axs_[1])
            if ix == 0:
                axs_[0].set_title('Total')
                axs_[1].set_title('First')
            if ix != len(objectives) - 1:
                axs_[0].set_xticklabels([])
                axs_[1].set_xticklabels([])
            else:
                axs_[0].set_xticklabels([x.replace('_','\n') for x in total.index], 
                                        rotation = 0)
                axs_[1].set_xticklabels([x.replace('_','\n') for x in total.index], 
                                        rotation =0 )
            axs_[0].set_ylabel(objective,rotation = 0,labelpad=20)
            axs_[0].get_legend().remove()
            axs_[1].get_legend().remove()
        f.tight_layout()
        f.savefig(plot_fid / f'{groups}_indices.png')  
        plt.close(f)
    """
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
    
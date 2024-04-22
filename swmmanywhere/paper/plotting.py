"""Plotting SWMManywhere.

A module with some built in plotting for SWMManywhere.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_behavioral_indices(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Create behavioral indices for a dataframe.

    Args:
        df (pd.DataFrame): A dataframe containing the results.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple of two series, the first is the
            behavioural indices for 'strict' objectives (KGE/NSE), the second 
            is the behavioural indices for less strict objectives (PBIAS).
    """
    behavioural_ind_nse = ((df.loc[:, df.columns.str.contains('nse')] > 0) & \
                           (df.loc[:, df.columns.str.contains('nse')] < 1)).any(axis=1)
    behavioural_ind_kge = ((df.loc[:, df.columns.str.contains('kge')] > -0.41) &\
                            (df.loc[:, df.columns.str.contains('kge')] < 1)).any(axis=1)
    behavioural_ind_bias = (df.loc[:, 
                                   df.columns.str.contains('bias')].abs() < 0.1
                            ).any(axis=1)
    return behavioural_ind_nse | behavioural_ind_kge, behavioural_ind_bias

def plot_objectives(df: pd.DataFrame, 
                    parameters: list[str], 
                    objectives: list[str], 
                    behavioral_indices: tuple[pd.Series, pd.Series],
                    plot_fid: Path):
    """Plot the objectives.

    Args:
        df (pd.DataFrame): A dataframe containing the results.
        parameters (list[str]): A list of parameters to plot.
        objectives (list[str]): A list of objectives to plot.
        behavioral_indices (tuple[pd.Series, pd.Series]): A tuple of two series
            see create_behavioral_indices.
        plot_fid (Path): The directory to save the plots to.
    """
    n_rows_cols = int(len(objectives)**0.5 + 1)
    for parameter in parameters:
        fig, axs = plt.subplots(n_rows_cols, n_rows_cols, figsize=(10, 10))
        for ax, objective in zip(axs.flat, objectives):
            setup_axes(ax, df, parameter, objective, behavioral_indices)
            add_threshold_lines(ax, 
                                objective, 
                                df[parameter].min(), 
                                df[parameter].max())
        fig.tight_layout()
        fig.suptitle(parameter)

        fig.savefig(plot_fid / (parameter.replace('_', '-') + '.png'), dpi=500)
        plt.close(fig)
        

def setup_axes(ax: plt.Axes, 
               df: pd.DataFrame,
               parameter: str, 
               objective: str, 
               behavioral_indices: tuple[pd.Series, pd.Series]
               ):
    """Set up the axes for plotting.

    Args:
        ax (plt.Axes): The axes to plot on.
        df (pd.DataFrame): A dataframe containing the results.
        parameter (list[str]): The parameter to plot.
        objective (list[str]): The objective to plot.
        behavioral_indices (tuple[pd.Series, pd.Series]): A tuple of two series
            see create_behavioral_indices.
    """
    ax.scatter(df[parameter], df[objective], s=0.5, c='b')
    ax.scatter(df.loc[behavioral_indices[1], parameter], 
               df.loc[behavioral_indices[1], objective], s=2, c='c')
    ax.scatter(df.loc[behavioral_indices[0], parameter], 
               df.loc[behavioral_indices[0], objective], s=2, c='r')
    ax.set_yscale('symlog')
    ax.set_title(objective)
    ax.grid(True)
    if 'nse' in objective:
        ax.set_ylim([-10, 1])

def add_threshold_lines(ax, objective, xmin, xmax):
    """Add threshold lines to the axes.

    Args:
        ax (plt.Axes): The axes to plot on.
        objective (list[str]): The objective to plot.
        xmin (float): The minimum x value.
        xmax (float): The maximum x value.
    """
    thresholds = {
        'bias': [-0.1, 0.1],
        'nse': [0],
        'kge': [-0.41]
    }
    for key, values in thresholds.items():
        if key in objective:
            for value in values:
                ax.plot([xmin, xmax], [value, value], 'k--')
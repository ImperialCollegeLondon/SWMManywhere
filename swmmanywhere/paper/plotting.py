"""Plotting SWMManywhere.

A module with some built in plotting for SWMManywhere.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from SALib.plotting.bar import plot as barplot

from swmmanywhere import metric_utilities
from swmmanywhere.geospatial_utilities import graph_to_geojson
from swmmanywhere.graph_utilities import load_graph
from swmmanywhere.parameters import MetricEvaluation
from swmmanywhere.preprocessing import create_project_structure
from swmmanywhere.swmmanywhere import load_config


class ResultsPlotter():
    """Plotter object."""
    def __init__(self, 
                 config_path: Path,
                 bbox_number: int | None = None,
                 model_number: int | None = None
                 ):
        """Initialise results plotter."""
        self.config = load_config(config_path)
        if model_number is not None:
            self.config['model_number'] = model_number
        if bbox_number is not None:
            self.config['bbox_number'] = bbox_number
        self.addresses = create_project_structure(self.config['bbox'],
                                self.config['project'],
                                self.config['base_dir'],
                                self.config['model_number']
                                )
        self.plotdir = self.addresses.model / 'plots'
        self.plotdir.mkdir(exist_ok = True)
        for key, val in self.config.get('address_overrides', {}).items():
            setattr(self.addresses, key, val)
        
        self._synthetic_results = pd.read_parquet(
            self.addresses.model / 'results.parquet')
        self._synthetic_results.id = self._synthetic_results.id.astype(str)

        if not self.config['real'].get('results',None):
            results_fid = self.config['real']['inp'].parent /\
                f'real_results.{self.addresses.extension}'
        else:
            results_fid = self.config['real']['results']
        self._real_results = pd.read_parquet(results_fid)
        self._real_results.id = self._real_results.id.astype(str)

        self._synthetic_G = load_graph(self.addresses.graph)
        self._synthetic_G = nx.relabel_nodes(self._synthetic_G,
                         {x : str(x) for x in self._synthetic_G.nodes})
        nx.set_node_attributes(self._synthetic_G,
            {u : str(d['outlet']) for u,d 
             in self._synthetic_G.nodes(data=True)},
            'outlet')
        calculate_slope(self._synthetic_G)
        
        self._real_G = load_graph(self.config['real']['graph'])
        self._real_G = nx.relabel_nodes(self._real_G,
                         {x : str(x) for x in self._real_G.nodes})
        calculate_slope(self._real_G)

        self._synthetic_subcatchments = gpd.read_file(self.addresses.subcatchments)
        self._real_subcatchments = gpd.read_file(self.config['real']['subcatchments'])
    
    def __getattr__(self, name):
        """For the large datasets, return a copy."""
        return getattr(self, f'_{name}').copy()

    def make_all_plots(self):
        """make_all_plots."""
        self.outlet_plot('flow')
        self.outlet_plot('flooding')
        self.shape_relerror_plot('grid')
        self.shape_relerror_plot('subcatchment')
        self.design_distribution(value='diameter')
        self.design_distribution(value='chamber_floor_elevation')
        self.design_distribution(value='slope')
        self.annotate_flows_and_depths()

    def annotate_flows_and_depths(self):
        """annotate_flows_and_depths."""
        synthetic_max = self.synthetic_results.groupby(['id','variable']).max()
        real_max = self.real_results.groupby(['id','variable']).max()

        syn_G = self.synthetic_G
        for u,v,d in syn_G.edges(data=True):
            d['flow'] = synthetic_max.loc[(d['id'],'flow'),'value']
        
        real_G = self.real_G
        for u,v,d in real_G.edges(data=True):
            d['flow'] = real_max.loc[(d['id'],'flow'),'value']

        for u,d in syn_G.nodes(data=True):
            d['flood'] = synthetic_max.loc[(u,'flooding'),'value']
        
        for u,d in real_G.nodes(data=True):
            d['flood'] = real_max.loc[(u,'flooding'),'value']

        graph_to_geojson(syn_G, 
                         self.plotdir / 'synthetic_graph_nodes.geojson',
                         self.plotdir / 'synthetic_graph_edges.geojson',
                         syn_G.graph['crs'])
        graph_to_geojson(real_G, 
                         self.plotdir / 'real_graph_nodes.geojson',
                         self.plotdir / 'real_graph_edges.geojson',
                         real_G.graph['crs'])
        

    def outlet_plot(self, 
                    var: str = 'flow',
                    fid: Path | None = None,):
        """Plot flow at outlet."""
        if not fid:
            fid = self.plotdir / f'outlet-{var}.png'
        sg_syn, syn_outlet = metric_utilities.best_outlet_match(self.synthetic_G, 
                                                                self.real_subcatchments)
        sg_real, real_outlet = metric_utilities.dominant_outlet(self.real_G, 
                                                                self.real_results)
        if var == 'flow':
            # Identify synthetic and real arcs that flow into the best outlet node
            syn_arc = [d['id'] for u,v,d in self.synthetic_G.edges(data=True)
                        if v == syn_outlet]
            real_arc = [d['id'] for u,v,d in self.real_G.edges(data=True)
                    if v == real_outlet]
        elif var == 'flooding':
            # Use all nodes in the subgraphs
            syn_arc = list(sg_syn.nodes)
            real_arc = list(sg_real.nodes)
        df = metric_utilities.align_by_id(self.synthetic_results,
                                           self.real_results,
                                           var,
                                           syn_arc,
                                           real_arc
                                           )
        f, ax = plt.subplots()
        df.plot(ax=ax)
        f.savefig(fid)

    def shape_relerror_plot(self, 
                        shape: str = 'grid',
                        fid: Path | None = None):
        """shape_relerror_plot."""
        if not fid:
            fid = self.plotdir / f'{shape}-relerror.geojson'
        variable = 'flooding'
        if shape == 'grid':
            scale = self.config.get('metric_evaluation', {}).get('grid_scale',1000)
            shapes = metric_utilities.create_grid(self.real_subcatchments.total_bounds,
                                                scale)
            shapes.crs = self.real_subcatchments.crs
        elif shape == 'subcatchment':
            shapes = self.real_subcatchments
            shapes = shapes.rename(columns={'id':'sub_id'})
        else:
            raise ValueError("shape must be 'grid' or 'subcatchment'")
        
        results = metric_utilities.align_by_shape(variable,
                                synthetic_results = self.synthetic_results,
                                real_results = self.real_results,
                                shapes = shapes,
                                synthetic_G = self.synthetic_G,
                                real_G = self.real_G)
        val = (
            results
            .groupby('sub_id')
            .apply(lambda x: metric_utilities.relerror(x.value_real, x.value_syn))
            .rename('relerror')
            .reset_index()
        )
        total = (
            results
            .groupby('sub_id')
            [['value_real','value_syn']]
            .sum()
        )
        
        shapes = pd.merge(shapes[['geometry','sub_id']],
                          val,
                          on ='sub_id')
        shapes = pd.merge(shapes, 
                          total,
                          on = 'sub_id')
        shapes.to_file(fid,driver='GeoJSON')

    def recalculate_metrics(self, metric_list: list[str] | None = None):
        """recalculate_metrics."""
        if not metric_list:
            metric_list_ = self.config['metric_list']
        else:
            metric_list_ = metric_list
        if 'metric_evaluation' in self.config.get('parameter_overrides', {}):
            metric_evaluation = MetricEvaluation(
                **self.config['parameter_overrides']['metric_evaluation'])
        else:
            metric_evaluation = MetricEvaluation()

        return metric_utilities.iterate_metrics(self.synthetic_results, 
                                  self.synthetic_subcatchments,
                                  self.synthetic_G,
                                  self.real_results,
                                  self.real_subcatchments,
                                  self.real_G,
                                  metric_list_,
                                  metric_evaluation
                                  )
    
    def design_distribution(self, 
                            fid: Path | None = None,
                            value: str = 'diameter',
                            weight: str='length'):
        """design_distribution."""
        if not fid:
            fid = self.plotdir / f'{value}_{weight}_distribution.png'
        syn_v, syn_cdf = weighted_cdf(self.synthetic_G,value,weight)
        real_v, real_cdf = weighted_cdf(self.real_G,value,weight)
        f, ax = plt.subplots()
        ax.plot(real_v,
                 real_cdf, 
                 'b')
        ax.plot(syn_v,
                 syn_cdf, 
                 '--k')
        ax.set_xlabel(f'{value.title()} (m)')
        ax.set_ylabel('P(X <= x)')
        plt.legend(['real','synthetic'])

        f.savefig(fid)  
    
def calculate_slope(G):
    """calculate_slope."""
    for u,v,d in G.edges(data=True):
        d['slope'] = (G.nodes[v]['chamber_floor_elevation'] - \
                      G.nodes[u]['chamber_floor_elevation'])/d['length']

def weighted_cdf(G, value: str = 'diameter', weight: str = 'length'):
    """weighted_cdf."""
    # Create a DataFrame from the provided lists
    if value in ['diameter','slope']:
        data = pd.DataFrame([
            {value: d[value], 'weight': d.get(weight,1)}
            for u,v,d in G.edges(data=True)
        ])
    elif value == 'chamber_floor_elevation':
        data = pd.DataFrame([
                    {value: d[value], 'weight': d.get(weight,1)}
                    for u,d in G.nodes(data=True)
        ])        

    # Sort by diameter
    data_sorted = data.sort_values(by=value)

    # Calculate cumulative weights
    cumulative_weights = data_sorted['weight'].cumsum()

    # Normalize the cumulative weights to form the CDF
    cumulative_weights /= cumulative_weights.iloc[-1]

    return data_sorted[value].tolist(), cumulative_weights.tolist()

def create_behavioral_indices(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Create behavioral indices for a dataframe.

    Args:
        df (pd.DataFrame): A dataframe containing the results.

    Returns:
        tuple[pd.Series, pd.Series]: A tuple of two series, the first is the
            behavioural indices for 'strict' objectives (KGE/NSE), the second 
            is the behavioural indices for less strict objectives (relerror).
    """
    behavioural_ind_nse = ((df.loc[:, df.columns.str.contains('nse')] > 0) & \
                           (df.loc[:, df.columns.str.contains('nse')] < 1)).any(axis=1)
    behavioural_ind_kge = ((df.loc[:, df.columns.str.contains('kge')] > -0.41) &\
                            (df.loc[:, df.columns.str.contains('kge')] < 1)).any(axis=1)
    behavioural_ind_relerror = (df.loc[:, 
                                   df.columns.str.contains('relerror')].abs() < 0.1
                            ).any(axis=1)
    return behavioural_ind_nse | behavioural_ind_kge, behavioural_ind_relerror

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

        fig.savefig(plot_fid / f"{parameter.replace('_', '-')}.png", dpi=500)
        plt.close(fig)
    return fig

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
        'relerror': [-0.1, 0.1],
        'nse': [0],
        'kge': [-0.41]
    }
    for key, values in thresholds.items():
        if key in objective:
            for value in values:
                ax.plot([xmin, xmax], [value, value], 'k--')

def plot_sensitivity_indices(r_: dict[str, pd.DataFrame],
                             objectives: list[str],
                             plot_fid: Path):
    """Plot the sensitivity indices.

    Args:
        r_ (dict[str, pd.DataFrame]): A dictionary containing the sensitivity 
            indices as produced by SALib.analyze.
        objectives (list[str]): A list of objectives to plot.
        plot_fid (Path): The directory to save the plots to.
    """
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
    f.savefig(plot_fid)  
    plt.close(f)
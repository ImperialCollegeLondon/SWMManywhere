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
from swmmanywhere.parameters import MetricEvaluation, filepaths_from_yaml


class ResultsPlotter():
    """Plotter object."""
    def __init__(self, 
                 address_path: Path,
                 real_dir: Path,
                 ):
        """Initialise results plotter.
        
        This plotter loads the results, graphs and subcatchments from the two
        yaml files. It provides a central point for plotting the results without
        needing to reload data.

        Args:
            address_path (Path): The path to the address yaml file.
            real_dir (Path): The path to the directory containing the real data.
        """
        # Load the addresses
        self.addresses = filepaths_from_yaml(address_path)

        # Create the plot directory
        self.plotdir = self.addresses.model / 'plots'
        self.plotdir.mkdir(exist_ok = True)

        # Load synthetic and real results
        self._synthetic_results = pd.read_parquet(
            self.addresses.model / 'results.parquet')
        self._synthetic_results.id = self._synthetic_results.id.astype(str)

        self._real_results = pd.read_parquet(real_dir / 'real_results.parquet')
        self._real_results.id = self._real_results.id.astype(str)

        # Load the synthetic and real graphs
        self._synthetic_G = load_graph(self.addresses.graph)
        self._synthetic_G = nx.relabel_nodes(self._synthetic_G,
                         {x : str(x) for x in self._synthetic_G.nodes})
        nx.set_node_attributes(self._synthetic_G,
            {u : str(d.get('outlet',None)) for u,d 
             in self._synthetic_G.nodes(data=True)},
            'outlet')

        self._real_G = load_graph(real_dir / 'graph.json')
        self._real_G = nx.relabel_nodes(self._real_G,
                         {x : str(x) for x in self._real_G.nodes})
        
        # Calculate the slope
        calculate_slope(self._synthetic_G)
        calculate_slope(self._real_G)

        # Load the subcatchments
        self._synthetic_subcatchments = gpd.read_file(self.addresses.subcatchments)
        self._real_subcatchments = gpd.read_file(real_dir / 'subcatchments.geojson')

    def __getattr__(self, name):
        """Because these are large datasets, return a copy."""
        return getattr(self, f'_{name}').copy()

    def make_all_plots(self):
        """make_all_plots."""
        f,axs = plt.subplots(2,3,figsize = (10,7.5))
        self.outlet_plot('flow', ax_ = axs[0,0])
        self.outlet_plot('flooding', ax_ = axs[0,1])
        self.shape_relerror_plot('grid')
        self.shape_relerror_plot('subcatchment')
        self.design_distribution(value='diameter', ax_ = axs[0,2])
        self.design_distribution(value='chamber_floor_elevation', ax_ = axs[1,0])
        self.design_distribution(value='slope', ax_ = axs[1,1])
        self.annotate_flows_and_depths()
        f.tight_layout()
        f.savefig(self.plotdir / 'all_plots.png')

    def annotate_flows_and_depths(self):
        """annotate_flows_and_depths.
        
        Annotate maximum flow and flood values on the edges/nodes of the graph.
        Save these in the plotdir.
        """
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
                    fid: Path | None = None,
                    ax_ = None):
        """Plot flow/flooding at outlet.

        If an ax is provided, plot on that ax, otherwise create a new figure and
        save it to the provided fid (or plot directory if not provided).
        
        Args:
            var (str, optional): The variable to plot (flow or flooding). 
                Defaults to 'flow'.
            fid (Path, optional): The file to save the plot to. Defaults to None.
            ax_ ([type], optional): The axes to plot on. Defaults to None.
        """            
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
            # Use all nodes in the outlet match subgraphs
            syn_arc = list(sg_syn.nodes)
            real_arc = list(sg_real.nodes)
        df = metric_utilities.align_by_id(self.synthetic_results,
                                           self.real_results,
                                           var,
                                           syn_arc,
                                           real_arc
                                           )
        if not ax_:
            f, ax = plt.subplots()
        else:
            ax = ax_
        df.value_real.plot(ax=ax, color = 'b', linestyle = '-')
        df.value_syn.plot(ax=ax, color = 'r', linestyle = '--')
        plt.legend(['synthetic','real'])
        ax.set_xlabel('time')
        if var == 'flow':
            unit = 'l/s'
        elif var == 'flooding':
            unit = 'l'
        ax.set_ylabel(f'{var.title()} ({unit})')
        if not ax_:
            f.savefig(self.plotdir / f'outlet-{var}.png')

    def shape_relerror_plot(self, shape: str = 'grid'):
        """shape_relerror_plot.
        
        Plot the relative error of the shape. Either at 'grid' or 'subcatchment' 
        scale. Saves results to the plotdir.
        
        Args:
            shape (str, optional): The shape to plot. Defaults to 'grid'.
        """            
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
        # Align the results
        results = metric_utilities.align_by_shape(variable,
                                synthetic_results = self.synthetic_results,
                                real_results = self.real_results,
                                shapes = shapes,
                                synthetic_G = self.synthetic_G,
                                real_G = self.real_G)
        # Calculate the relative error
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
        # Merge with shapes
        shapes = pd.merge(shapes[['geometry','sub_id']],
                          val,
                          on ='sub_id')
        shapes = pd.merge(shapes, 
                          total,
                          on = 'sub_id')
        shapes.to_file(self.plotdir / f'{shape}-relerror.geojson',
                       driver='GeoJSON')

    def recalculate_metrics(self, metric_list: list[str] | None = None):
        """recalculate_metrics.
        
        Recalculate the metrics for the synthetic and real results, if no
        metric_list is provided, use the default metric_list from the config.

        Args:
            metric_list (list[str], optional): The metrics to recalculate. 
                Defaults to None.

        Returns:
            dict: A dictionary of the recalculated metrics.
        """
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
                            value: str = 'diameter',
                            weight: str='length',
                            ax_ = None):
        """design_distribution.

        Plot the distribution of a value in the graph. Saves the plot to the
        provided axes, if not, saves to plotdir.

        Args:
            value (str, optional): The value to plot. Defaults to 'diameter'.
            weight (str, optional): The weight to use. Defaults to 'length'.
            ax_ ([type], optional): The axes to plot on. Defaults to None.
        """
        syn_v, syn_cdf = weighted_cdf(self.synthetic_G,value,weight)
        real_v, real_cdf = weighted_cdf(self.real_G,value,weight)
        if not ax_:
            f, ax = plt.subplots()
        else:
            ax = ax_
        ax.plot(real_v,real_cdf, 'b')
        ax.plot(syn_v,syn_cdf, '--r')
        if value == 'slope':
            unit = 'm/m'
        elif value == 'chamber_floor_elevation':
            unit = 'mASL'
        else:
            unit = 'm'
        ax.set_xlabel(f'{value.title()} ({unit})')
        ax.set_ylabel('P(X <= x)')
        plt.legend(['real','synthetic'])

        if not ax_:
            f.savefig(self.plotdir / f'{value}_{weight}_distribution.png')  

def calculate_slope(G):
    """calculate_slope.
    
    Calculate the slope of the edges in the graph in place.
    
    Args:
        G ([type]): The graph to calculate the slope for.
    """
    for u,v,d in G.edges(data=True):
        d['slope'] = (G.nodes[v]['chamber_floor_elevation'] - \
                      G.nodes[u]['chamber_floor_elevation'])/d['length']

def weighted_cdf(G: nx.Graph, value: str = 'diameter', weight: str = 'length'):
    """weighted_cdf.
    
    Calculate the weighted cumulative distribution function of a value in the
    graph.
    
    Args:
        G (nx.Graph): The graph to calculate the cdf for.
        value (str, optional): The value to calculate the cdf for. Defaults to 
            'diameter'.
        weight (str, optional): The weight to use. Defaults to 'length'.

    Returns:
        tuple[list, list]: The values and the cdf.
    """
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
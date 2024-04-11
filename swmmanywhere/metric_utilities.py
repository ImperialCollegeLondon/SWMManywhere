"""Metric utilities module for SWMManywhere.

A module for metrics, the metrics registry object and utilities for calculating
metrics (such as NSE or timeseries data alignment) used in SWMManywhere.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Callable, Optional, get_type_hints

import cytoolz.curried as tlz
import geopandas as gpd
import joblib
import netcomp
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy import stats

from swmmanywhere.logging import logger
from swmmanywhere.parameters import MetricEvaluation


class MetricRegistry(dict): 
    """Registry object.""" 
    def _log_completion(self, func):
        def _wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logger.info(f'{func.__name__} completed')
            return result
        return _wrapper
    
    def register(self, func: Callable) -> Callable:
        """Register a metric."""
        if func.__name__ in self:
            raise ValueError(f"{func.__name__} already in the metric registry!")

        allowable_params = {"synthetic_results": pd.DataFrame,
                            "real_results": pd.DataFrame,
                            "synthetic_subs": gpd.GeoDataFrame,
                            "real_subs": gpd.GeoDataFrame,
                            "synthetic_G": nx.Graph,
                            "real_G": nx.Graph,
                            "metric_evaluation": MetricEvaluation}

        # Use get_type_hints to resolve annotations, 
        # considering 'from __future__ import annotations'
        type_hints = get_type_hints(func)

        for param, annotation in type_hints.items():
            if param in ('kwargs', 'return'):
                continue
            if param not in allowable_params:
                raise ValueError(f"{param} of {func.__name__} not allowed.")
            if annotation != allowable_params[param]:
                raise ValueError(f"""{param} of {func.__name__} should be of
                                 type {allowable_params[param]}, not 
                                 {annotation}.""")
        self[func.__name__] = self._log_completion(func)
        return func

    def __getattr__(self, name):
        """Get a metric from the graphfcn dict."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{name} NOT in the metric registry!")
metrics = MetricRegistry()

def iterate_metrics(synthetic_results: pd.DataFrame, 
                    synthetic_subs: gpd.GeoDataFrame,
                    synthetic_G: nx.Graph,
                    real_results: pd.DataFrame,
                    real_subs: gpd.GeoDataFrame,
                    real_G: nx.Graph,
                    metric_list: list[str],
                    metric_evaluation: MetricEvaluation) -> dict[str, float]:
    """Iterate a list of metrics over a graph.

    Args:
        synthetic_results (pd.DataFrame): The synthetic results.
        synthetic_subs (gpd.GeoDataFrame): The synthetic subcatchments.
        synthetic_G (nx.Graph): The synthetic graph.
        real_results (pd.DataFrame): The real results.
        real_subs (gpd.GeoDataFrame): The real subcatchments.
        real_G (nx.Graph): The real graph.
        metric_list (list[str]): A list of metrics to iterate.
        metric_evaluation (MetricEvaluation): The metric evaluation parameters.

    Returns:
        dict[str, float]: The results of the metrics.
    """
    not_exists = [m for m in metric_list if m not in metrics]
    if not_exists:
        raise ValueError(f"Metrics are not registered:\n{', '.join(not_exists)}")
    
    kwargs = {
        "synthetic_results": synthetic_results,
        "synthetic_subs": synthetic_subs,
        "synthetic_G": synthetic_G,
        "real_results": real_results,
        "real_subs": real_subs,
        "real_G": real_G,
        "metric_evaluation": metric_evaluation
    }

    return {m : metrics[m](**kwargs) for m in metric_list}

def extract_var(df: pd.DataFrame,
                     var: str) -> pd.DataFrame:
    """Extract var from a dataframe."""
    df_ = df.loc[df.variable == var].copy()
    df_.loc[:,'duration'] = (df_.date - \
                        df_.date.min()).dt.total_seconds()
    return df_

# Coefficient Registry
coef_registry = {}

def register_coef(coef_func: Callable):
    """Register a coefficient function.
    
    Register a coefficient function to the coef_registry. The function should
    take two arguments, 'y' and 'yhat', and return a float. The function should
    be registered with the '@register_coef' decorator.
    
    Args:
        coef_func (Callable): The coefficient function to register.
    """
    name = coef_func.__name__

    # Check if the function is already registered
    if name in coef_registry:
        raise ValueError(f"Coefficient function '{name}' already registered.")
    
    # Validate the function
    args = list(get_type_hints(coef_func).keys())
    if 'y' != args[0] or 'yhat' != args[1]:
        raise ValueError(f"Coef {coef_func.__name__} requires args ('y', 'yhat').")
    
    # Add the function to the registry
    coef_registry[name] = coef_func
    return coef_func

@register_coef
def pbias(y: np.ndarray,
          yhat: np.ndarray) -> float:
    r"""Percentage bias, PBIAS.

    Calculate the percent bias:

    .. math::

        pbias = \\frac{{\sum(synthetic) - \sum(real)}}{{\sum(real)}}

    where:
    - :math:`synthetic` is the synthetic data,
    - :math:`real` is the real data.

    Args:
        y (np.ndarray): The real data.
        yhat (np.ndarray): The synthetic data.

    Returns:
        float: The PBIAS value.
    """
    total_observed = y.sum()
    if total_observed == 0:
        return np.inf
    return (yhat.sum() - total_observed) / total_observed

@register_coef
def nse(y: np.ndarray,
        yhat: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe efficiency (NSE).
    
    Args:
        y (np.array): Observed data array.
        yhat (np.array): Simulated data array.

    Returns:
        float: The NSE value.
    """
    if np.std(y) == 0:
        return np.inf
    return 1 - np.sum(np.square(y - yhat)) / np.sum(np.square(y - np.mean(y)))

@register_coef
def kge(y: np.ndarray,yhat: np.ndarray) -> float:
    """Calculate the Kling-Gupta Efficiency (KGE) between simulated and observed data.
    
    Args:
        y (np.array): Observed data array.
        yhat (np.array): Simulated data array.
    
    Returns:
        float: The KGE value.
    """
    if (np.std(y) == 0) | (np.mean(y) == 0):
        return np.inf
    if np.std(yhat) == 0:
        r = 0
    else:
        r = np.corrcoef(yhat, y)[0, 1]
    alpha = np.std(yhat) / np.std(y)
    beta = np.mean(yhat) / np.mean(y)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def align_calc_coef(synthetic_results: pd.DataFrame, 
                  real_results: pd.DataFrame, 
                  variable: str, 
                  syn_ids: list,
                  real_ids: list,
                  coef_func: Callable = nse) -> float:
    """Align and calculate coef_func.

    Aggregate synthetic and real results by date for specifics ids. 
    Align the synthetic and real dates and calculate the coef_func metric
    of the variable over time. In cases where the synthetic
    data is does not overlap the real data, the value is interpolated.

    Args:
        synthetic_results (pd.DataFrame): The synthetic results.
        real_results (pd.DataFrame): The real results.
        variable (str): The variable to align and calculate coef_func for.
        syn_ids (list): The ids of the synthetic data to subselect for.
        real_ids (list): The ids of the real data to subselect for.
        coef_func (Callable, optional): The coefficient to calculate. 
            Defaults to nse.

    Returns:
        float: The coef_func value.
    """
    synthetic_results = synthetic_results.copy()
    real_results = real_results.copy()

    # Format dates
    synthetic_results['date'] = pd.to_datetime(synthetic_results['date'])
    real_results['date'] = pd.to_datetime(real_results['date'])

    # Help alignment
    synthetic_results["id"] = synthetic_results["id"].astype(str)
    real_results["id"] = real_results["id"].astype(str)
    syn_ids = [str(x) for x in syn_ids]
    real_ids = [str(x) for x in real_ids]

    # Extract data
    syn_data = extract_var(synthetic_results, variable)
    syn_data = syn_data.loc[syn_data["id"].isin(syn_ids)]
    syn_data = syn_data.groupby('date').value.sum()

    real_data = extract_var(real_results, variable)
    real_data = real_data.loc[real_data["id"].isin(real_ids)]
    real_data = real_data.groupby('date').value.sum()
    
    # Align data
    df = pd.merge(syn_data, 
                  real_data, 
                  left_index = True,
                  right_index = True,
                  suffixes=('_syn', '_real'), 
                  how='outer').sort_index()

    # Interpolate to time in real data
    df['value_syn'] = df.value_syn.interpolate().to_numpy()
    df = df.dropna(subset=['value_real'])

    # Calculate coef_func
    return coef_func(df.value_real, df.value_syn)

def create_subgraph(G: nx.Graph,
                    nodes: list) -> nx.Graph:
    """Create a subgraph.
    
    Create a subgraph of G based on the nodes list. Taken from networkx 
    documentation: https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.subgraph.html
    
    Args:
        G (nx.Graph): The original graph.
        nodes (list): The list of nodes to include in the subgraph.

    Returns:
        nx.Graph: The subgraph.
    """
    # Create a subgraph SG based on a (possibly multigraph) G
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in nodes)
    if SG.is_multigraph():
        SG.add_edges_from((n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in nodes
            for nbr, keydict in nbrs.items() if nbr in nodes
            for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in nodes
            for nbr, d in nbrs.items() if nbr in nodes)
    SG.graph.update(G.graph)
    return SG

def median_coef_by_group(results: pd.DataFrame,
                        gb_key: str,
                        coef_func: Callable = nse) -> float:
    """Median coef_func value by group.

    Calculate the median coef_func value of a variable over time
    for each group in the results dataframe, and return the median of these
    values.

    Args:
        results (pd.DataFrame): The results dataframe.
        gb_key (str): The column to group by.
        coef_func (Callable): The coefficient to calculate. Default is nse.

    Returns:
        float: The median coef_func value.
    """
    val = (
        results
        .groupby(['date',gb_key])
        .sum()
        .reset_index()
        .groupby(gb_key)
        .apply(lambda x: coef_func(x.value_real, x.value_syn))
    )
    val = val[val.map(np.isfinite)]
    return val.median()


def nodes_to_subs(G: nx.Graph,
                  subs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Nodes to subcatchments.

    Classify the nodes of the graph to the subcatchments of the subs dataframe.

    Args:
        G (nx.Graph): The graph.
        subs (gpd.GeoDataFrame): The subcatchments.

    Returns:
        gpd.GeoDataFrame: A dataframe from the nodes and data, and the 
            subcatchment information, distinguished by the column 'sub_id'.
    """
    nodes_df = pd.DataFrame([{'id' :x, **d} for x,d in G.nodes(data=True)])
    nodes_joined = (
        gpd.GeoDataFrame(nodes_df, 
                        geometry=gpd.points_from_xy(nodes_df.x, 
                                                    nodes_df.y),
                        crs = G.graph['crs'])
        .sjoin(subs.rename(columns = {'id' : 'sub_id'}), 
               how="inner", 
               predicate="within")
    )
    return nodes_joined

def best_outlet_match(synthetic_G: nx.Graph,
                      real_subs: gpd.GeoDataFrame) -> tuple[nx.Graph,int]:
    """Best outlet match.
    
    Identify the outlet with the most nodes within the real_subs and return the
    subgraph of the synthetic graph of nodes that drain to that outlet.

    Args:
        synthetic_G (nx.Graph): The synthetic graph.
        real_subs (gpd.GeoDataFrame): The real subcatchments.

    Returns:
        nx.Graph: The subgraph of the synthetic graph for the outlet with the 
            most nodes within the real_subs.
        int: The id of the outlet.
    """
    nodes_joined = nodes_to_subs(synthetic_G, real_subs)
    
    # Select the most common outlet
    outlet = nodes_joined.outlet.value_counts().idxmax()

    # Subselect the matching graph
    outlet_nodes = [n for n, d in synthetic_G.nodes(data=True) 
                    if d['outlet'] == outlet]
    sg = create_subgraph(synthetic_G,outlet_nodes)
    return sg, outlet

def dominant_outlet(G: nx.DiGraph,
                    results: pd.DataFrame) -> tuple[nx.DiGraph,int]:
    """Dominant outlet.

    Identify the outlet with highest flow along it and return the
    subgraph of the graph of nodes that drain to that outlet.

    Args:
        G (nx.DiGraph): The graph.
        results (pd.DataFrame): The results, which include a 'flow' and 'id' 
            column.

    Returns:
        nx.Graph: The subgraph of nodes/arcs that the reach max flow outlet
        int: The id of the outlet.
    """
    # Identify outlets of the graph
    outlets = [n for n, outdegree in G.out_degree() 
               if outdegree == 0]
    outlet_arcs = [d['id'] for u,v,d in G.edges(data=True) 
                   if v in outlets]
    
    # Identify the outlet with the highest flow
    outlet_flows = results.loc[(results.variable == 'flow') &
                               (results["id"].isin(outlet_arcs))]
    max_outlet_arc = outlet_flows.groupby('id').value.median().idxmax()
    max_outlet = [v for u,v,d in G.edges(data=True) 
                  if d['id'] == max_outlet_arc][0]
    
    # Subselect the matching graph
    sg = create_subgraph(G, nx.ancestors(G, max_outlet) | {max_outlet})
    return sg, max_outlet

def nc_compare(G1, G2, funcname, **kw):
    """Compare two graphs using netcomp."""
    A1,A2 = [nx.adjacency_matrix(G) for G in (G1,G2)]
    return getattr(netcomp, funcname)(A1,A2,**kw)

def edge_betweenness_centrality(G: nx.Graph, 
                                normalized: bool = True,
                                weight: Optional[str] = "weight", 
                                njobs: int = -1):
    """Parallel betweenness centrality function."""
    njobs = joblib.cpu_count(True) if njobs == -1 else njobs
    node_chunks = tlz.partition_all(G.order() // njobs, G.nodes())
    bt_func = tlz.partial(nx.edge_betweenness_centrality_subset, 
                          G=G, 
                          normalized=normalized, 
                          weight=weight)
    bt_sc = joblib.Parallel(n_jobs=njobs)(
        joblib.delayed(bt_func)(sources=nodes, 
                                targets=G.nodes()) for nodes in node_chunks
    )

    # Merge the betweenness centrality results
    bt_c: dict[int, float] = defaultdict(float)
    for bt in bt_sc:
        for n, v in bt.items():
            bt_c[n] += v
    return bt_c

def align_by_shape(var,
                          synthetic_results: pd.DataFrame,
                          real_results: pd.DataFrame,
                          shapes: gpd.GeoDataFrame,
                          synthetic_G: nx.Graph,
                          real_G: nx.Graph) -> pd.DataFrame:
    """Align by subcatchment.

    Align synthetic and real results by shape and return the results.

    Args:
        var (str): The variable to align.
        synthetic_results (pd.DataFrame): The synthetic results.
        real_results (pd.DataFrame): The real results.
        shapes (gpd.GeoDataFrame): The shapes to align by (e.g., grid or real_subs).
        synthetic_G (nx.Graph): The synthetic graph.
        real_G (nx.Graph): The real graph.
    """
    synthetic_joined = nodes_to_subs(synthetic_G, shapes)
    real_joined = nodes_to_subs(real_G, shapes)

    # Extract data
    real_results = extract_var(real_results, var)
    synthetic_results = extract_var(synthetic_results, var)

    # Format to help alignment
    real_results["id"] = real_results["id"].astype(str)
    synthetic_results["id"] = synthetic_results["id"].astype(str)
    real_joined["id"] = real_joined["id"].astype(str)
    synthetic_joined["id"] = synthetic_joined["id"].astype(str)


    # Align data
    synthetic_results = pd.merge(synthetic_results,
                                 synthetic_joined[['id','sub_id']],
                                 on='id')
    real_results = pd.merge(real_results,
                            real_joined[['id','sub_id']],
                            on='id')
    
    results = pd.merge(real_results[['date','sub_id','value']],
                            synthetic_results[['date','sub_id','value']],
                            on = ['date','sub_id'],
                            suffixes = ('_real', '_syn'),
                            how = 'outer'
                            )
    
    results['value_syn'] = results.value_syn.interpolate().to_numpy()
    results = results.dropna(subset=['value_real'])

    return results

def create_grid(bbox: tuple,
                scale: float | tuple[float,float]) -> gpd.GeoDataFrame:
    """Create a grid of polygons.
    
    Create a grid of polygons based on the bounding box and scale.

    Args:
        bbox (tuple): The bounding box coordinates in the format (minx, miny, 
            maxx, maxy).
        scale (float | tuple): The scale of the grid. If a tuple, the scale is 
            (dx, dy). Otherwise, the scale is dx = dy = scale.
    
    Returns:
        gpd.GeoDataFrame: A geodataframe of the grid.
    """
    minx, miny, maxx, maxy = bbox

    if isinstance(scale, tuple):
        dx, dy = scale
    else: 
        dx = dy = scale
    xmins = np.arange(minx, maxx, dx)
    ymins = np.arange(miny, maxy, dy)
    grid = [{'geometry' : shapely.box(x, y, x + dx, y + dy),
             'sub_id' : i} for i, (x, y) in enumerate(product(xmins, ymins))]

    return gpd.GeoDataFrame(grid)

scale_registry = {}
def register_scale(scale_func: Callable):
    """Register a scale function.
    
    Register a scale function to the scale_registry. The function should
    take the same arguments as the scale functions and return a float. The 
    function should be registered with the '@register_scale' decorator. A scale
    function is called as a metric, but with some additional arguments provided
    (i.e., the variable name and the coefficient function to use). The function
    should return a float.
    
    Args:
        scale_func (Callable): The scale function to register.
    """
    name = scale_func.__name__

    # Check if the function is already registered
    if name in scale_registry:
        raise ValueError(f"Scale function '{name}' already registered.")
    
    # Validate the function
    args = list(get_type_hints(scale_func).keys())
    if args != ['synthetic_results', 'synthetic_subs', 'synthetic_G', 
                'real_results', 'real_subs', 'real_G', 'metric_evaluation',
                'var', 'coef_func']:
        raise ValueError(f"""Scale {scale_func.__name__} requires args 
                         ('synthetic_results', 'synthetic_subs', 'synthetic_G', 
                         'real_results', 'real_subs', 'real_G', 
                         'metric_evaluation', 'var', 'coef_func').""")
    
    # Add the function to the registry
    scale_registry[name] = scale_func
    return scale_func

@register_scale
def subcatchment(synthetic_results: pd.DataFrame,
                synthetic_subs: gpd.GeoDataFrame,
                synthetic_G: nx.Graph,
                real_results: pd.DataFrame,
                real_subs: gpd.GeoDataFrame,
                real_G: nx.Graph,
                metric_evaluation: MetricEvaluation,
                var: str,
                coef_func: Callable,
                ):
    """Subcatchment scale metric.

    Calculate the coefficient (coef_func) of a variable over time for aggregated 
    to real subcatchment scale. The metric produced is the median coef_func 
    across all subcatchments.

    Args:
        synthetic_results (pd.DataFrame): The synthetic results.
        synthetic_subs (gpd.GeoDataFrame): The synthetic subcatchments.
        synthetic_G (nx.Graph): The synthetic graph.
        real_results (pd.DataFrame): The real results.
        real_subs (gpd.GeoDataFrame): The real subcatchments.
        real_G (nx.Graph): The real graph.
        metric_evaluation (MetricEvaluation): The metric evaluation parameters.
        var (str): The variable to calculate the coefficient for.
        coef_func (Callable): The coefficient to calculate.

    Returns:
        float: The median coef_func value.
    """
    results = align_by_shape(var,
                                    synthetic_results = synthetic_results,
                                    real_results = real_results,
                                    shapes = real_subs,
                                    synthetic_G = synthetic_G,
                                    real_G = real_G)
    
    return median_coef_by_group(results, 'sub_id', coef_func=coef_func)

@register_scale
def grid(synthetic_results: pd.DataFrame,
                synthetic_subs: gpd.GeoDataFrame,
                synthetic_G: nx.Graph,
                real_results: pd.DataFrame,
                real_subs: gpd.GeoDataFrame,
                real_G: nx.Graph,
                metric_evaluation: MetricEvaluation,
                var: str,
                coef_func: Callable,
                ):
    """Grid scale metric.

    Classify synthetic nodes to a grid and calculate the coef_func of a variable over
    time for each grid cell. The metric produced is the median coef_func across all
    grid cells.

    Args:
        synthetic_results (pd.DataFrame): The synthetic results.
        synthetic_subs (gpd.GeoDataFrame): The synthetic subcatchments.
        synthetic_G (nx.Graph): The synthetic graph.
        real_results (pd.DataFrame): The real results.
        real_subs (gpd.GeoDataFrame): The real subcatchments.
        real_G (nx.Graph): The real graph.
        metric_evaluation (MetricEvaluation): The metric evaluation parameters.
        var (str): The variable to calculate the coefficient for.
        coef_func (Callable): The coefficient to calculate.

    Returns:
        float: The median coef_func value.
    """
    # Create a grid (GeoDataFrame of polygons)
    scale = metric_evaluation.grid_scale
    grid = create_grid(real_subs.total_bounds,
                       scale)
    grid.crs = real_subs.crs

    # Align results
    results = align_by_shape(var,
                                    synthetic_results = synthetic_results,
                                    real_results = real_results,
                                    shapes = grid,
                                    synthetic_G = synthetic_G,
                                    real_G = real_G)
    # Calculate coefficient
    return median_coef_by_group(results, 'sub_id', coef_func=coef_func)

@register_scale
def outlet(synthetic_results: pd.DataFrame,
                synthetic_subs: gpd.GeoDataFrame,
                synthetic_G: nx.Graph,
                real_results: pd.DataFrame,
                real_subs: gpd.GeoDataFrame,
                real_G: nx.Graph,
                metric_evaluation: MetricEvaluation,
                var: str,
                coef_func: Callable,
                ):
    """Outlet scale metric.

    Calculate the coefficient of a variable for the subgraph that
    drains to the dominant outlet node. The dominant outlet node of the 'real'
    network is calculated by dominant_outlet, while the dominant outlet node of
    the 'synthetic' network is calculated by best_outlet_match.

    Args:
        synthetic_results (pd.DataFrame): The synthetic results.
        synthetic_subs (gpd.GeoDataFrame): The synthetic subcatchments.
        synthetic_G (nx.Graph): The synthetic graph.
        real_results (pd.DataFrame): The real results.
        real_subs (gpd.GeoDataFrame): The real subcatchments.
        real_G (nx.Graph): The real graph.
        metric_evaluation (MetricEvaluation): The metric evaluation parameters.
        var (str): The variable to calculate the coefficient for.
        coef_func (Callable): The coefficient to calculate.

    Returns:
        float: The median coef_func value.
    """
    # Identify synthetic and real arcs that flow into the best outlet node
    sg_syn, syn_outlet = best_outlet_match(synthetic_G, real_subs)
    sg_real, real_outlet = dominant_outlet(real_G, real_results)
    
    allowable_var = ['nmanholes', 'npipes', 'length', 'flow', 'flooding']
    if var not in allowable_var:
        raise ValueError(f"Invalid variable {var}. Can be {allowable_var}")

    if var == 'nmanholes':
        # Calculate the coefficient based on the number of manholes
        return coef_func(np.atleast_1d(sg_real.number_of_nodes()),
                    np.atleast_1d(sg_syn.number_of_nodes()))
    if var == 'npipes':
        # Calculate the coefficient based on the number of pipes
        return coef_func(np.atleast_1d(sg_real.number_of_edges()),
                    np.atleast_1d(sg_syn.number_of_edges()))
    if var == 'length':
        # Calculate the coefficient based on the total length of the pipes
        return coef_func(
            np.array(
                list(nx.get_edge_attributes(sg_real, 'length').values())
                ),
            np.array(
                list(nx.get_edge_attributes(sg_syn, 'length').values())
                )
            )
    if var == 'flow':
        # Identify synthetic and real arcs that flow into the best outlet node
        syn_arc = [d['id'] for u,v,d in synthetic_G.edges(data=True)
                    if v == syn_outlet]
        real_arc = [d['id'] for u,v,d in real_G.edges(data=True)
                if v == real_outlet]
    elif var == 'flooding':
        # Use all nodes in the subgraphs
        syn_arc = list(sg_syn.nodes)
        real_arc = list(sg_real.nodes)

    # Calculate the coefficient
    return align_calc_coef(synthetic_results, 
                         real_results, 
                         var,
                         syn_arc, 
                         real_arc,
                         coef_func = coef_func)

def metric_factory(name: str):
    """Create a metric function.
    
    A factory function to create a metric function based on the name. The first
    part of the name is the scale, the second part is the metric, and the third
    part is the variable. For example, 'grid_nse_flooding' is a metric function
    that calculates the NSE of flooding at the grid scale.

    Args:
        name (str): The name of the metric.

    Returns:
        Callable: The metric function.
    """
    # Split the name
    parts = name.split('_')
    if len(parts) != 3:
        raise ValueError("Invalid metric name. Expected 'scale_metric_variable'")
    scale, metric, variable = parts

    # Get coefficient
    coef_func = coef_registry[metric]

    # Get scale
    func = scale_registry[scale]
    
    # Further validation
    design_variables = ['length', 'nmanholes', 'npipes']
    if variable in design_variables:
        if scale != 'outlet':
            raise ValueError(f"Variable {variable} only supported at the outlet scale")
        if metric != 'pbias':
            raise ValueError(f"Variable {variable} only valid with pbias metric")

    # Create the metric function
    def new_metric(**kwargs):
        return func(coef_func = coef_func,
                    var = variable,
                    **kwargs)
    new_metric.__name__ = name
    return new_metric

metrics.register(metric_factory('outlet_nse_flow'))
metrics.register(metric_factory('outlet_kge_flow'))
metrics.register(metric_factory('outlet_pbias_flow'))

metrics.register(metric_factory('outlet_pbias_length'))
metrics.register(metric_factory('outlet_pbias_npipes'))
metrics.register(metric_factory('outlet_pbias_nmanholes'))

metrics.register(metric_factory('outlet_nse_flooding'))
metrics.register(metric_factory('outlet_kge_flooding'))
metrics.register(metric_factory('outlet_pbias_flooding'))

metrics.register(metric_factory('grid_nse_flooding'))
metrics.register(metric_factory('grid_kge_flooding'))
metrics.register(metric_factory('grid_pbias_flooding'))

metrics.register(metric_factory('subcatchment_nse_flooding'))
metrics.register(metric_factory('subcatchment_kge_flooding'))
metrics.register(metric_factory('subcatchment_pbias_flooding'))

@metrics.register
def nc_deltacon0(synthetic_G: nx.Graph,
                  real_G: nx.Graph,
                  **kwargs) -> float:
    """Run the evaluated metric."""
    return nc_compare(synthetic_G, 
                      real_G, 
                      'deltacon0',
                      eps = 1e-10)

@metrics.register
def nc_laplacian_dist(synthetic_G: nx.Graph,
                  real_G: nx.Graph,
                  **kwargs) -> float:
    """Run the evaluated metric."""
    return nc_compare(synthetic_G, 
                      real_G, 
                      'lambda_dist',
                      k=10,
                      kind = 'laplacian')

@metrics.register
def nc_laplacian_norm_dist(synthetic_G: nx.Graph,
                  real_G: nx.Graph,
                  **kwargs) -> float:
    """Run the evaluated metric."""
    return nc_compare(synthetic_G, 
                      real_G, 
                      'lambda_dist',
                      k=10,
                      kind = 'laplacian_norm')

@metrics.register
def nc_adjacency_dist(synthetic_G: nx.Graph,
                  real_G: nx.Graph,
                  **kwargs) -> float:
    """Run the evaluated metric."""
    return nc_compare(synthetic_G, 
                      real_G, 
                      'lambda_dist',
                      k=10,
                      kind = 'adjacency')

@metrics.register
def nc_vertex_edge_distance(synthetic_G: nx.Graph,
                  real_G: nx.Graph,
                  **kwargs) -> float:
    """Run the evaluated metric.
    
    Do '1 -' because this metric is similarity not distance.
    """
    return 1 - nc_compare(synthetic_G, 
                           real_G, 
                          'vertex_edge_distance')

@metrics.register
def nc_resistance_distance(synthetic_G: nx.Graph,
                  real_G: nx.Graph,
                  **kwargs) -> float:
    """Run the evaluated metric."""
    return nc_compare(synthetic_G,
                        real_G,
                        'resistance_distance',
                        check_connected = False,
                        renormalized = True)

@metrics.register
def bias_flood_depth(
                 synthetic_results: pd.DataFrame,
                 real_results: pd.DataFrame,
                 synthetic_subs: gpd.GeoDataFrame,
                 real_subs: gpd.GeoDataFrame,
                 **kwargs) -> float:
        """Run the evaluated metric."""
        
        def _f(x):
            return np.trapz(x.value,x.duration)

        syn_flooding = extract_var(synthetic_results,
                                    'flooding').groupby('id').apply(_f)
        syn_area = synthetic_subs.impervious_area.sum()
        syn_tot = syn_flooding.sum() / syn_area

        real_flooding = extract_var(real_results,
                                    'flooding').groupby('id').apply(_f)
        real_area = real_subs.impervious_area.sum()
        real_tot = real_flooding.sum() / real_area

        return (syn_tot - real_tot) / real_tot

@metrics.register
def kstest_edge_betweenness( 
                 synthetic_G: nx.Graph,
                 real_G: nx.Graph,
                 **kwargs) -> float:
        """Run the evaluated metric."""
        syn_betweenness = edge_betweenness_centrality(synthetic_G, weight=None)
        real_betweenness = edge_betweenness_centrality(real_G, weight=None)

        #TODO does it make more sense to use statistic or pvalue?
        return stats.ks_2samp(list(syn_betweenness.values()),
                              list(real_betweenness.values())).statistic

@metrics.register
def kstest_betweenness( 
                 synthetic_G: nx.Graph,
                 real_G: nx.Graph,
                 **kwargs) -> float:
        """Run the evaluated metric."""
        syn_betweenness = nx.betweenness_centrality(synthetic_G, weight=None)
        real_betweenness = nx.betweenness_centrality(real_G, weight=None)

        #TODO does it make more sense to use statistic or pvalue?
        return stats.ks_2samp(list(syn_betweenness.values()),
                              list(real_betweenness.values())).statistic

@metrics.register
def outlet_kstest_diameters(real_G: nx.Graph,
                            synthetic_G: nx.Graph,
                            real_results: pd.DataFrame,
                            real_subs: gpd.GeoDataFrame,
                            **kwargs) -> float:
    """Outlet KStest diameters.

    Calculate the Kolmogorov-Smirnov statistic of the diameters in the subgraph
    that drains to the dominant outlet node. The dominant outlet node of the
    'real' network is calculated by dominant_outlet, while the dominant outlet
    node of the 'synthetic' network is calculated by best_outlet_match.
    """
    # Identify synthetic and real outlet arcs
    sg_syn, _ = best_outlet_match(synthetic_G, real_subs)
    sg_real, _ = dominant_outlet(real_G, real_results)
    
    # Extract the diameters
    syn_diameters = nx.get_edge_attributes(sg_syn, 'diameter')
    real_diameters = nx.get_edge_attributes(sg_real, 'diameter')
    return stats.ks_2samp(list(syn_diameters.values()),
                         list(real_diameters.values())).statistic
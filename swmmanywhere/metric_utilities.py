# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
from collections import defaultdict
from inspect import signature
from typing import Callable, Optional

import cytoolz.curried as tlz
import geopandas as gpd
import joblib
import netcomp
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


class MetricRegistry(dict): 
    """Registry object.""" 
    
    def register(self, func: Callable) -> Callable:
        """Register a metric."""
        if func.__name__ in self:
            raise ValueError(f"{func.__name__} already in the metric registry!")

        allowable_params = {"synthetic_results": pd.DataFrame,
                            "real_results": pd.DataFrame,
                            "synthetic_subs": gpd.GeoDataFrame,
                            "real_subs": gpd.GeoDataFrame,
                            "synthetic_G": nx.Graph,
                            "real_G": nx.Graph}

        sig = signature(func)
        for param, obj in sig.parameters.items():
            if param == 'kwargs':
                continue
            if param not in allowable_params:
                raise ValueError(f"{param} of {func.__name__} not allowed.")
            if obj.annotation != allowable_params[param]:
                raise ValueError(f"""{param} of {func.__name__} should be of
                                 type {allowable_params[param]}, not 
                                 {obj.__class__}.""")
        self[func.__name__] = func
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
                    metric_list: list[str]) -> dict[str, float]:
    """Iterate a list of metrics over a graph.

    Args:
        synthetic_results (pd.DataFrame): The synthetic results.
        synthetic_subs (gpd.GeoDataFrame): The synthetic subcatchments.
        synthetic_G (nx.Graph): The synthetic graph.
        real_results (pd.DataFrame): The real results.
        real_subs (gpd.GeoDataFrame): The real subcatchments.
        real_G (nx.Graph): The real graph.
        metric_list (list[str]): A list of metrics to iterate.

    Returns:
        dict[str, float]: The results of the metrics.
    """
    results = {}
    for metric in metric_list:
        assert metric in metrics.keys(), f"Metric {metric} not registered in metrics."
        results[metric] = metrics[metric](synthetic_results = synthetic_results,
                                       synthetic_subs = synthetic_subs,
                                       synthetic_G = synthetic_G,
                                       real_results = real_results,
                                       real_subs = real_subs,
                                       real_G = real_G)
    return results

def extract_var(df: pd.DataFrame,
                     var: str) -> pd.DataFrame:
    """Extract var from a dataframe."""
    df_ = df.loc[df.variable == var]
    df_['duration'] = (df_.date - \
                        df_.date.min()).dt.total_seconds()
    return df_

def align_calc_nse(synthetic_results: pd.DataFrame, 
                  real_results: pd.DataFrame, 
                  variable: str, 
                  syn_ids: list,
                  real_ids: list) -> float:
    """Align and calculate NSE.

    Align the synthetic and real data and calculate the Nash-Sutcliffe
    efficiency (NSE) of the variable over time. In cases where the synthetic
    data is does not overlap the real data, the value is interpolated.
    """
    # Format dates
    synthetic_results['date'] = pd.to_datetime(synthetic_results['date'])
    real_results['date'] = pd.to_datetime(real_results['date'])

    # Extract data
    syn_data = extract_var(synthetic_results, variable)
    syn_data = syn_data.loc[syn_data.object.isin(syn_ids)]
    syn_data = syn_data.groupby('date').value.sum()

    real_data = extract_var(real_results, variable)
    real_data = real_data.loc[real_data.object.isin(real_ids)]
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

    # Calculate NSE
    return nse(df.value_real, df.value_syn)

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

def nse(y: np.ndarray,
        yhat: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe efficiency (NSE)."""
    return 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)

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
    # Identify which nodes fall within real_subs
    nodes_df = pd.DataFrame([d for x,d in synthetic_G.nodes(data=True)],
                             index = synthetic_G.nodes)
    nodes_joined = (
        gpd.GeoDataFrame(nodes_df, 
                         geometry=gpd.points_from_xy(nodes_df.x, 
                                                    nodes_df.y),
                         crs = synthetic_G.graph['crs'])
        .sjoin(real_subs, 
               how="right", 
               predicate="within")
    )

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
        G (nx.Graph): The graph.
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
                               (results.object.isin(outlet_arcs))]
    max_outlet_arc = outlet_flows.groupby('object').value.mean().idxmax()
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
                                    'flooding').groupby('object').apply(_f)
        syn_area = synthetic_subs.impervious_area.sum()
        syn_tot = syn_flooding.sum() / syn_area

        real_flooding = extract_var(real_results,
                                    'flooding').groupby('object').apply(_f)
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
def outlet_nse_flow(synthetic_G: nx.Graph,
                  synthetic_results: pd.DataFrame,
                  real_G: nx.Graph,
                  real_results: pd.DataFrame,
                  real_subs: gpd.GeoDataFrame,
                  **kwargs) -> float:
    """Outlet NSE flow.

    Calculate the Nash-Sutcliffe efficiency (NSE) of flow over time, where flow
    is measured as the total flow of all arcs that drain to the 'dominant'
    outlet node. The dominant outlet node of the 'real' network is calculated by
    dominant_outlet, while the dominant outlet node of the 'synthetic' network
    is calculated by best_outlet_match.
    """
    # Identify synthetic and real arcs that flow into the best outlet node
    _, syn_outlet = best_outlet_match(synthetic_G, real_subs)
    syn_arc = [d['id'] for u,v,d in synthetic_G.edges(data=True)
                if v == syn_outlet]
    _, real_outlet = dominant_outlet(real_G, real_results)
    real_arc = [d['id'] for u,v,d in real_G.edges(data=True)
                if v == real_outlet]
    
    return align_calc_nse(synthetic_results, 
                         real_results, 
                         'flow', 
                         syn_arc, 
                         real_arc)


@metrics.register
def outlet_nse_flooding(synthetic_G: nx.Graph,
                  synthetic_results: pd.DataFrame,
                  real_G: nx.Graph,
                  real_results: pd.DataFrame,
                  real_subs: gpd.GeoDataFrame,
                  **kwargs) -> float:
    """Outlet NSE flooding.
    
    Calculate the Nash-Sutcliffe efficiency (NSE) of flooding over time, where
    flooding is the total volume of flooded water across all nodes that drain
    to the 'dominant' outlet node. The dominant outlet node of the 'real' 
    network is calculated by dominant_outlet, while the dominant outlet node of
    the 'synthetic' network is calculated by best_outlet_match.
    """
    # Identify synthetic and real outlet arcs
    sg_syn, _ = best_outlet_match(synthetic_G, real_subs)
    sg_real, _ = dominant_outlet(real_G, real_results)
    
    return align_calc_nse(synthetic_results, 
                         real_results, 
                         'flooding', 
                         list(sg_syn.nodes),
                         list(sg_real.nodes))
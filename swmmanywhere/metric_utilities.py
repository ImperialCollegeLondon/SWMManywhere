# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
from inspect import signature
from typing import Callable

import geopandas as gpd
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
    # Extract data
    syn_data = extract_var(synthetic_results, variable)
    syn_data = syn_data.loc[syn_data.object.isin(syn_ids)]
    syn_data = syn_data.groupby('date').value.sum().reset_index()

    real_data = extract_var(real_results, variable)
    real_data = real_data.loc[real_data.object.isin(real_ids)]
    real_data = real_data.groupby('date').value.sum().reset_index()

    # Align data
    df = pd.merge(syn_data, 
                  real_data, 
                  on='date', 
                  suffixes=('_syn', '_real'), 
                  how='outer').sort_values(by='date')

    # Interpolate to time in real data
    df['value_syn'] = df.set_index('date').value_syn.interpolate().values
    df = df.dropna(subset=['value_real'])

    # Calculate NSE
    return nse(df.value_real, df.value_syn)

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
               predicate="intersects")
    )

    # Select the most common outlet
    outlet = nodes_joined.outlet.value_counts().idxmax()

    # Subselect the matching graph
    outlet_nodes = [n for n, d in synthetic_G.nodes(data=True) 
                    if d['outlet'] == outlet]
    return synthetic_G.subgraph(outlet_nodes), outlet

def dominant_outlet(G: nx.Graph,
                    results: pd.DataFrame) -> tuple[nx.Graph,int]:
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
    sg = G.subgraph(nx.ancestors(G, max_outlet) | {max_outlet})
    return sg, max_outlet

def nc_compare(G1, G2, funcname, **kw):
    """Compare two graphs using netcomp."""
    A1,A2 = [nx.adjacency_matrix(G) for G in (G1,G2)]
    return getattr(netcomp, funcname)(A1,A2,**kw)

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
def kstest_betweenness( 
                 synthetic_G: nx.Graph,
                 real_G: nx.Graph,
                 **kwargs) -> float:
        """Run the evaluated metric."""
        syn_betweenness = nx.betweenness_centrality(synthetic_G)
        real_betweenness = nx.betweenness_centrality(real_G)

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
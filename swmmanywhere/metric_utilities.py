# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
from abc import ABC, abstractmethod
from typing import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


class BaseMetric(ABC):
    """Base metric class."""
    @abstractmethod
    def __call__(self, 
                 *args,
                 **kwargs) -> float:
        """Run the evaluated metric."""
        return 0

class MetricRegistry(dict): 
    """Registry object.""" 
    
    def register(self, cls):
        """Register a metric."""
        if cls.__name__ in self:
            raise ValueError(f"{cls.__name__} already in the metric registry!")

        self[cls.__name__] = cls()
        return cls

    def __getattr__(self, name):
        """Get a metric from the graphfcn dict."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{name} NOT in the metric registry!")
        

metrics = MetricRegistry()

def register_metric(cls) -> Callable:
    """Register a metric.

    Args:
        cls (Callable): A class that inherits from BaseMetric

    Returns:
        cls (Callable): The same class
    """
    metrics.register(cls)
    return cls

def extract_var(df: pd.DataFrame,
                     var: str) -> pd.DataFrame:
    """Extract var from a dataframe."""
    df_ = df.loc[df.variable == var]
    df_['duration'] = (df_.date - \
                        df_.date.min()).dt.total_seconds()
    return df_

@register_metric
class bias_flood_depth(BaseMetric):
    """Bias flood depth."""
    def __call__(self, 
                 synthetic_results: pd.DataFrame,
                 real_results: pd.DataFrame,
                 synthetic_subs: gpd.GeoDataFrame,
                 real_subs: gpd.GeoDataFrame,
                 *args,
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

@register_metric
class kstest_betweenness(BaseMetric):
    """KS two sided of betweenness distribution."""
    def __call__(self, 
                 synthetic_G: nx.Graph,
                 real_G: nx.Graph,
                 real_subs: gpd.GeoDataFrame,
                 real_results: pd.DataFrame,
                 *args,
                 **kwargs) -> float:
        """Run the evaluated metric."""
        # Identify synthetic outlet and subgraph
        sg_syn, _ = best_outlet_match(synthetic_G, real_subs)
        
        # Identify real outlet and subgraph
        sg_real, _ = dominant_outlet(real_G, real_results)

        syn_betweenness = nx.betweenness_centrality(sg_syn)
        real_betweenness = nx.betweenness_centrality(sg_real)

        #TODO does it make more sense to use statistic or pvalue?
        return stats.ks_2samp(list(syn_betweenness.values()),
                              list(real_betweenness.values())).statistic

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
    nodes_gdf = gpd.GeoDataFrame(nodes_df, 
                                 geometry=gpd.points_from_xy(nodes_df.x, 
                                                             nodes_df.y),
                                 crs = synthetic_G.graph['crs'])
    nodes_joined = nodes_gdf.sjoin(real_subs, 
                                   how="right", 
                                   predicate="intersects")

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
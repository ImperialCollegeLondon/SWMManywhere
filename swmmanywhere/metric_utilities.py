# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
from inspect import signature
from typing import Callable

import geopandas as gpd
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
            if param not in allowable_params.keys():
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
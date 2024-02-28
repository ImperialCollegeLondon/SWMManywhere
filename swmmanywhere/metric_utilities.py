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
                 *args,
                 **kwargs) -> float:
        """Run the evaluated metric."""
        syn_betweenness = nx.betweenness_centrality(synthetic_G)
        real_betweenness = nx.betweenness_centrality(real_G)

        #TODO does it make more sense to use statistic or pvalue?
        return stats.ks_2samp(list(syn_betweenness.values()),
                              list(real_betweenness.values())).statistic
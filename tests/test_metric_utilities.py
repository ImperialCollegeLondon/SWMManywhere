from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely

from swmmanywhere import metric_utilities as mu
from swmmanywhere.graph_utilities import load_graph


def test_bias_flood_depth():
    """Test the bias_flood_depth metric."""
    # Create synthetic and real data
    synthetic_results = pd.DataFrame({
        'object': ['obj1', 'obj1','obj2','obj2'],
        'value': [10, 20, 5, 2],
        'variable': 'flooding',
        'date' : pd.to_datetime(['2021-01-01 00:00:00','2021-01-01 00:05:00',
                                 '2021-01-01 00:00:00','2021-01-01 00:05:00'])
    })
    real_results = pd.DataFrame({
        'object': ['obj1', 'obj1','obj2','obj2'],
        'value': [15, 25, 10, 20],
        'variable': 'flooding',
        'date' : pd.to_datetime(['2021-01-01 00:00:00','2021-01-01 00:05:00',
                                 '2021-01-01 00:00:00','2021-01-01 00:05:00'])
    })
    synthetic_subs = pd.DataFrame({
        'impervious_area': [100, 200],
    })
    real_subs = pd.DataFrame({
        'impervious_area': [150, 250],
    })

    # Run the metric
    val = mu.metrics.bias_flood_depth(synthetic_results = synthetic_results, 
                              real_results = real_results,
                              synthetic_subs = synthetic_subs,
                              real_subs = real_subs)
    assert val == -0.29523809523809524

def test_best_outlet_match():
    """Test the best_outlet_match and ks_betweenness."""
    G = load_graph(Path(__file__).parent / 'test_data' / 'graph_topo_derived.json')
    subs = [shapely.Polygon([(700262, 5709928),
                            (700262, 5709883),
                            (700351, 5709883),
                            (700351, 5709906),
                            (700306, 5709906),
                            (700306, 5709928),
                            (700262, 5709928)]),
            shapely.Polygon([(700306, 5709928),
                            (700284, 5709928),
                            (700284, 5709950),
                            (700374, 5709950),
                            (700374, 5709906),
                            (700351, 5709906),
                            (700306, 5709906),
                            (700306, 5709928)]),
            shapely.Polygon([(700351, 5709883),
                            (700351, 5709906),
                            (700374, 5709906),
                            (700374, 5709883),
                            (700396, 5709883),
                            (700396, 5709816),
                            (700329, 5709816),
                            (700329, 5709838),
                            (700329, 5709883),
                            (700351, 5709883)])]

    subs = gpd.GeoDataFrame(data = {'id' : [107733,
                                            1696030874,
                                            6277683849]
                                            },
                                    geometry = subs,
                                    crs = G.graph['crs'])
    
    sg, outlet = mu.best_outlet_match(synthetic_G = G, 
                                     real_subs = subs)
    outlets = nx.get_node_attributes(sg, 'outlet')
    assert len(set(outlets.values())) == 1
    assert outlet == 12354833

    results = pd.DataFrame([{'object' : 4253560,
                             'variable' : 'flow',
                             'value' : 10},
                            {'object' : '',
                             'variable' : 'flow',
                             'value' : 5}])

    val = mu.metrics.kstest_betweenness(synthetic_G = G, 
                                 real_G = G, 
                                 real_subs = subs,
                                 real_results = results)
    
    assert val == 0.0

    # Move the outlet up one node
    G_ = G.copy()
    nx.set_node_attributes(G_,
                           list(G_.in_edges(outlet))[0][0],
                           'outlet')
    G_.remove_node(outlet)

    val = mu.metrics.kstest_betweenness(synthetic_G = G_,
                                real_G = G,
                                real_subs = subs,
                                real_results = results)
    assert val == 0.4

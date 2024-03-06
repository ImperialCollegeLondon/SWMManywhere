from pathlib import Path

import numpy as np
import pandas as pd

from swmmanywhere.graph_utilities import load_graph
from swmmanywhere.metric_utilities import metrics as sm


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
    val = sm.bias_flood_depth(synthetic_results = synthetic_results, 
                              real_results = real_results,
                              synthetic_subs = synthetic_subs,
                              real_subs = real_subs)
    assert np.isclose(val, -0.29523809523809524)

def test_kstest_betweenness():
    """Test the kstest_betweenness metric."""
    G = load_graph(Path(__file__).parent / 'test_data' / 'graph_topo_derived.json')
    val = sm.kstest_betweenness(synthetic_G = G, real_G = G)
    assert val == 0.0

    G_ = G.copy()
    G_.remove_node(list(G.nodes)[0])
    val = sm.kstest_betweenness(synthetic_G = G_, real_G = G)
    assert np.isclose(val, 0.286231884057971)
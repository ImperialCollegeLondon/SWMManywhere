from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import shapely

from swmmanywhere import metric_utilities as mu
from swmmanywhere.graph_utilities import load_graph
from swmmanywhere.parameters import MetricEvaluation


def assert_close(a: float, b: float, rtol: float = 1e-3) -> None:
    """Assert that two floats are close."""
    assert np.isclose(a, b, rtol=rtol).all()


@pytest.fixture
def subs():
    """Get a GeoDataFrame of subcatchments."""
    subs = [
        shapely.Polygon(
            [
                (700262, 5709928),
                (700262, 5709883),
                (700351, 5709883),
                (700351, 5709906),
                (700306, 5709906),
                (700306, 5709928),
                (700262, 5709928),
            ]
        ),
        shapely.Polygon(
            [
                (700306, 5709928),
                (700284, 5709928),
                (700284, 5709950),
                (700374, 5709950),
                (700374, 5709906),
                (700351, 5709906),
                (700306, 5709906),
                (700306, 5709928),
            ]
        ),
        shapely.Polygon(
            [
                (700351, 5709883),
                (700351, 5709906),
                (700374, 5709906),
                (700374, 5709883),
                (700396, 5709883),
                (700396, 5709816),
                (700329, 5709816),
                (700329, 5709838),
                (700329, 5709883),
                (700351, 5709883),
            ]
        ),
    ]

    subs = gpd.GeoDataFrame(
        data={"id": [107733, 1696030874, 6277683849]}, geometry=subs, crs="EPSG:32630"
    )
    return subs


@pytest.fixture
def results():
    """Get a DataFrame of results."""
    results = pd.DataFrame(
        [
            {
                "id": 4253560,
                "variable": "flow",
                "value": 10,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 4253560,
                "variable": "flow",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 1696030874,
                "variable": "flooding",
                "value": 4.5,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 770549936,
                "variable": "flooding",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 107736,
                "variable": "flooding",
                "value": 10,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 107733,
                "variable": "flooding",
                "value": 1,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 107737,
                "variable": "flooding",
                "value": 2,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 1696030874,
                "variable": "flooding",
                "value": 0,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 770549936,
                "variable": "flooding",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 107736,
                "variable": "flooding",
                "value": 15,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 107733,
                "variable": "flooding",
                "value": 2,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 107737,
                "variable": "flooding",
                "value": 2,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
        ]
    )
    return results


def test_bias_flood_depth():
    """Test the bias_flood_depth metric."""
    # Create synthetic and real data
    synthetic_results = pd.DataFrame(
        {
            "id": ["obj1", "obj1", "obj2", "obj2"],
            "value": [10, 20, 5, 2],
            "variable": "flooding",
            "date": pd.to_datetime(
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:05:00",
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:05:00",
                ]
            ),
        }
    )
    real_results = pd.DataFrame(
        {
            "id": ["obj1", "obj1", "obj2", "obj2"],
            "value": [15, 25, 10, 20],
            "variable": "flooding",
            "date": pd.to_datetime(
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:05:00",
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:05:00",
                ]
            ),
        }
    )
    synthetic_subs = pd.DataFrame(
        {
            "impervious_area": [100, 200],
        }
    )
    real_subs = pd.DataFrame(
        {
            "impervious_area": [150, 250],
        }
    )

    # Run the metric
    val = mu.metrics.bias_flood_depth(
        synthetic_results=synthetic_results,
        real_results=real_results,
        synthetic_subs=synthetic_subs,
        real_subs=real_subs,
    )
    assert_close(val, -0.2952)


def test_kstest_betweenness():
    """Test the kstest_betweenness metric."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    val = mu.metrics.kstest_betweenness(synthetic_G=G, real_G=G)
    assert val == 0.0

    G_ = G.copy()
    G_.remove_node(list(G.nodes)[0])
    val = mu.metrics.kstest_betweenness(synthetic_G=G_, real_G=G)
    assert_close(val, 0.2862)


def test_kstest_edge_betweenness():
    """Test the kstest_betweenness metric."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    val = mu.metrics.kstest_edge_betweenness(synthetic_G=G, real_G=G)
    assert val == 0.0

    G_ = G.copy()
    G_.remove_node(list(G.nodes)[0])
    val = mu.metrics.kstest_edge_betweenness(synthetic_G=G_, real_G=G)
    assert_close(val, 0.38995)


def test_best_outfall_match(subs):
    """Test the best_outfall_match and ks_betweenness."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")

    sg, outfall = mu.best_outfall_match(synthetic_G=G, real_subs=subs)
    outfalls = nx.get_node_attributes(sg, "outfall")
    assert len(set(outfalls.values())) == 1
    assert outfall == 12354833


def test_nse():
    """Test the nse metric."""
    val = mu.nse(y=np.array([1, 2, 3, 4, 5]), yhat=np.array([1, 2, 3, 4, 5]))
    assert val == 1.0

    val = mu.nse(y=np.array([1, 2, 3, 4, 5]), yhat=np.array([3, 3, 3, 3, 3]))
    assert val == 0.0


def test_kge():
    """Test the kge metric."""
    val = mu.kge(y=np.array([1, 2, 3, 4, 5]), yhat=np.array([1, 2, 3, 4, 5]))
    assert_close(val, 1.0)

    val = mu.kge(y=np.array([1, 2, 3, 4, 5]), yhat=np.array([3, 3, 3, 3, 3]))
    assert_close(val, (1 - 2**0.5))


def test_inf():
    """Test metrics handling of invalid coefficients."""
    val = mu.kge(y=np.array([3, 3, 3, 3, 3]), yhat=np.array([1, 2, 3, 4, 5]))
    assert val == np.inf

    val = mu.nse(y=np.array([3, 3, 3, 3, 3]), yhat=np.array([1, 2, 3, 4, 5]))
    assert val == np.inf

    val = mu.relerror(y=np.array([-3, -3, 0, 3, 3]), yhat=np.array([1, 2, 3, 4, 5]))
    assert val == np.inf


def test_relerror_different_length():
    """Test the relerror metric with different length arrays."""
    val = mu.relerror(y=np.array([1, 2, 3, 4, 5, 6]), yhat=np.array([1]))
    assert_close(val, (1 - 3.5) / 3.5)


def test_outfall_nse_flow(subs):
    """Test the outfall_nse_flow metric."""
    # Load data
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")

    # Mock results
    results = pd.DataFrame(
        [
            {
                "id": 4253560,
                "variable": "flow",
                "value": 10,
                "date": pd.to_datetime("2021-01-01").date(),
            },
            {
                "id": "",
                "variable": "flow",
                "value": 5,
                "date": pd.to_datetime("2021-01-01").date(),
            },
            {
                "id": 4253560,
                "variable": "flow",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": "",
                "variable": "flow",
                "value": 2,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
        ]
    )

    # Calculate NSE (perfect results)
    val = mu.metrics.outfall_nse_flow(
        synthetic_G=G,
        synthetic_subs=None,
        synthetic_results=results,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 1.0

    # Calculate NSE (mean results)
    results_ = results.copy()
    results_.loc[[0, 2], "value"] = 7.5
    val = mu.metrics.outfall_nse_flow(
        synthetic_G=G,
        synthetic_subs=None,
        synthetic_results=results_,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 0.0

    # Change the graph
    G_ = G.copy()
    new_outfall = list(G_.in_edges(12354833))[0][0]
    nx.set_node_attributes(G_, new_outfall, "outfall")
    G_.remove_node(12354833)
    results_.loc[results_.id == 4253560, "id"] = 725226531

    # Calculate NSE (mean results)
    val = mu.metrics.outfall_nse_flow(
        synthetic_G=G_,
        synthetic_subs=None,
        synthetic_results=results_,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 0.0

    # Test time interpolation
    results_.loc[2, "date"] = pd.to_datetime("2021-01-01 00:00:10")
    results_.loc[[0, 2], "value"] = [0, 30]
    val = mu.metrics.outfall_nse_flow(
        synthetic_G=G_,
        synthetic_subs=None,
        synthetic_results=results_,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == -15.0


def test_outfall_nse_flooding(subs):
    """Test the outfall_nse_flow metric."""
    # Load data
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")

    # Mock results
    results = pd.DataFrame(
        [
            {
                "id": 4253560,
                "variable": "flow",
                "value": 10,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 4253560,
                "variable": "flow",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 25472468,
                "variable": "flooding",
                "value": 4.5,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 770549936,
                "variable": "flooding",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 109753,
                "variable": "flooding",
                "value": 10,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 25472468,
                "variable": "flooding",
                "value": 0,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 770549936,
                "variable": "flooding",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
            {
                "id": 109753,
                "variable": "flooding",
                "value": 15,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
        ]
    )

    # Calculate NSE (perfect results)
    val = mu.metrics.outfall_nse_flooding(
        synthetic_G=G,
        synthetic_subs=None,
        synthetic_results=results,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 1.0

    # Calculate NSE (mean results)
    results_ = results.copy()
    results_.loc[results_.id.isin([770549936, 25472468]), "value"] = [14.5 / 4] * 4
    val = mu.metrics.outfall_nse_flooding(
        synthetic_G=G,
        synthetic_subs=None,
        synthetic_results=results_,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 0.0

    # Change the graph
    G_ = G.copy()
    new_outfall = list(G_.in_edges(12354833))[0][0]
    nx.set_node_attributes(
        G_,
        {x: new_outfall for x, d in G_.nodes(data=True) if d["outfall"] == 12354833},
        "outfall",
    )
    G_.remove_node(12354833)

    # Calculate NSE (mean results)
    val = mu.metrics.outfall_nse_flooding(
        synthetic_G=G_,
        synthetic_subs=None,
        synthetic_results=results_,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 0.0

    # Test time interpolation
    results_.loc[results_.date == pd.to_datetime("2021-01-01 00:00:05"), "date"] = (
        pd.to_datetime("2021-01-01 00:00:10")
    )

    val = mu.metrics.outfall_nse_flooding(
        synthetic_G=G_,
        synthetic_results=results_,
        synthetic_subs=None,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 0.0


def test_design_params(subs):
    """Test the design param related metrics."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    nx.set_edge_attributes(G, 0.15, "diameter")

    # Mock results (only needed for dominant outfall)
    results = pd.DataFrame(
        [
            {
                "id": 4253560,
                "variable": "flow",
                "value": 10,
                "date": pd.to_datetime("2021-01-01 00:00:00"),
            },
            {
                "id": 4253560,
                "variable": "flow",
                "value": 5,
                "date": pd.to_datetime("2021-01-01 00:00:05"),
            },
        ]
    )

    # Target results
    design_results = {
        "outfall_kstest_diameters": 0.0625,
        "outfall_relerror_diameter": 0.0625,
        "outfall_relerror_length": -0.15088965,
        "outfall_relerror_nmanholes": -0.05,
        "outfall_relerror_npipes": -0.15789473,
    }

    # Iterate for G = G, i.e., perfect results
    metrics = mu.iterate_metrics(
        synthetic_G=G,
        synthetic_subs=None,
        synthetic_results=None,
        real_G=G,
        real_subs=subs,
        real_results=results,
        metric_list=design_results.keys(),
        metric_evaluation=MetricEvaluation(),
    )
    for metric, val in metrics.items():
        assert metric in design_results
        assert np.isclose(val, 0)

    # edit the graph for target results
    G_ = G.copy()
    G_.remove_node(list(G.nodes)[0])
    G_.edges[list(G_.edges)[0]]["diameter"] = 0.3

    metrics = mu.iterate_metrics(
        synthetic_G=G_,
        synthetic_subs=None,
        synthetic_results=None,
        real_G=G,
        real_subs=subs,
        real_results=results,
        metric_list=design_results.keys(),
        metric_evaluation=MetricEvaluation(),
    )

    for metric, val in metrics.items():
        assert metric in design_results
        assert np.isclose(val, design_results[metric]), metric


def test_netcomp_iterate():
    """Test the netcomp metrics and iterate_metrics."""
    netcomp_results = {
        "nc_deltacon0": 0.00129408,
        "nc_laplacian_dist": 155.428234,
        "nc_laplacian_norm_dist": 0.752901,
        "nc_adjacency_dist": 226.22945,
        "nc_resistance_distance": 8.098548,
        "nc_vertex_edge_distance": 0.132075,
    }
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    metrics = mu.iterate_metrics(
        synthetic_G=G,
        synthetic_subs=None,
        synthetic_results=None,
        real_G=G,
        real_subs=None,
        real_results=None,
        metric_list=netcomp_results.keys(),
        metric_evaluation=MetricEvaluation(),
    )
    for metric, val in metrics.items():
        assert metric in netcomp_results
        assert np.isclose(val, 0)

    G_ = load_graph(Path(__file__).parent / "test_data" / "street_graph.json")
    metrics = mu.iterate_metrics(
        synthetic_G=G_,
        synthetic_subs=None,
        synthetic_results=None,
        real_G=G,
        real_subs=None,
        real_results=None,
        metric_list=netcomp_results.keys(),
        metric_evaluation=MetricEvaluation(),
    )
    for metric, val in metrics.items():
        assert metric in netcomp_results
        assert np.isclose(val, netcomp_results[metric])


def test_subcatchment_nse_flooding(subs, results):
    """Test the outfall_nse_flow metric."""
    # Load data
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")

    # Calculate NSE (perfect results)
    val = mu.metrics.subcatchment_nse_flooding(
        synthetic_G=G,
        synthetic_subs=None,
        real_G=G,
        synthetic_results=results,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 1.0

    # Calculate NSE (remapped node)

    G_ = G.copy()
    # Create a mapping from the old name to the new name
    mapping = {1696030874: "new_name", 107737: "new_name2"}

    # Rename the node
    G_ = nx.relabel_nodes(G_, mapping)

    results_ = results.copy()
    results_.id = results_.id.replace(mapping)

    val = mu.metrics.subcatchment_nse_flooding(
        synthetic_G=G_,
        synthetic_subs=None,
        synthetic_results=results_,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=None,
    )
    assert val == 1.0

    # Test gridded
    val = mu.metrics.grid_nse_flooding(
        synthetic_G=G_,
        synthetic_subs=None,
        synthetic_results=results_,
        real_G=G,
        real_results=results,
        real_subs=subs,
        metric_evaluation=MetricEvaluation(),
    )
    assert val == 1.0


def test_create_grid():
    """Test the create_grid function."""
    grid = mu.create_grid((0, 0, 1, 1), 1 / 3 - 0.001)
    assert grid.shape[0] == 16
    assert set(grid.columns) == {"sub_id", "geometry"}


def test_restirctions():
    """Test the restriction register by generating an invalid metric."""
    # Invalid because length can't be calculated at grid scale
    with pytest.raises(ValueError):
        mu.metric_factory("grid_relerror_length")

    # Invalid because nmanholes can't be evaluated with nse
    with pytest.raises(ValueError):
        mu.metric_factory("outfall_nse_nmanholes")


def test_nodes_to_subs(subs):
    """Test the nodes_to_subs function."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    nodes = mu.nodes_to_subs(G, subs)
    nodes = set([(r.id, r.sub_id) for _, r in nodes.iterrows()])
    nodes_ = {
        (107733, 107733),
        (107736, 1696030874),
        (107737, 107733),
        (32925453, 6277683849),
        (770549936, 1696030874),
        (6277683849, 6277683849),
    }
    assert nodes == nodes_


def test_align_by_shape_and_median_coef(subs, results):
    """Test the align_by_shape and median_coef_by_group function."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    # Align the grid
    aligned = mu.align_by_shape("flooding", results, results, subs, G, G)
    assert aligned.shape[0] == 4
    assert (aligned.value_real == aligned.value_syn).all()

    # Test interp
    results_ = results.copy()
    results_ = pd.concat(
        [
            results_,
            results_.replace(
                {
                    "date": {
                        pd.to_datetime("2021-01-01 00:00:00"): pd.to_datetime(
                            "2021-01-01 00:00:03"
                        ),
                        pd.to_datetime("2021-01-01 00:00:05"): pd.to_datetime(
                            "2021-01-01 00:00:02"
                        ),
                    }
                }
            ),
        ]
    )

    aligned = mu.align_by_shape("flooding", results, results_, subs, G, G)
    assert aligned.shape[0] == 8
    assert (
        aligned.loc[aligned.date.isin(results.date)].value_real
        == aligned.loc[aligned.date.isin(results.date)].value_syn
    ).all()
    assert_close(
        aligned.loc[[2, 3, 4, 5], "value_syn"], [3.3333, 16.6666, 3.6666, 18.3333]
    )

    # Test median coef
    meds = mu.median_coef_by_group(aligned, "sub_id", mu.nse)
    assert_close(meds, 0.1111)

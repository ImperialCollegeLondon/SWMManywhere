# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely import geometry as sgeom

from swmmanywhere import parameters
from swmmanywhere.filepaths import FilePaths
from swmmanywhere.graph_utilities import (
    filter_streets,
    iterate_graphfcns,
    load_graph,
    save_graph,
    validate_graphfcn_list,
)
from swmmanywhere.graph_utilities import graphfcns as gu
from swmmanywhere.logging import logger


@pytest.fixture
def street_network():
    """Load a street network."""
    bbox = (-0.11643, 51.50309, -0.11169, 51.50549)
    G = load_graph(Path(__file__).parent / "test_data" / "street_graph.json")
    return G, bbox


def test_save_load(street_network):
    """Test the save_graph and load_graph functions."""
    G, _ = street_network
    # Load a street network
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the graph
        save_graph(G, Path(temp_dir) / "test_graph.json")
        # Load the graph
        G_new = load_graph(Path(temp_dir) / "test_graph.json")
        # Check if the loaded graph is the same as the original graph
        assert nx.is_isomorphic(G, G_new)


def test_assign_id(street_network):
    """Test the assign_id function."""
    G, _ = street_network
    G = gu.assign_id(G)
    for u, v, data in G.edges(data=True):
        assert "id" in data.keys()
        assert isinstance(data["id"], str)


def test_double_directed(street_network):
    """Test the double_directed function."""
    G, _ = street_network
    G = gu.assign_id(G)
    G = gu.double_directed(G)
    for u, v in G.edges():
        assert (v, u) in G.edges


def test_calculate_streetcover(street_network):
    """Test the calculate_streetcover function."""
    G, _ = street_network
    params = parameters.SubcatchmentDerivation()
    with tempfile.TemporaryDirectory() as temp_dir:
        addresses = FilePaths(
            base_dir=Path(temp_dir),
            bbox_bounds=[0, 1, 0, 1],
            project_name="",
            extension="json",
            streetcover=Path(temp_dir) / "streetcover.geojson",
        )
        _ = gu.calculate_streetcover(G, params, addresses)
        # TODO test that G hasn't changed? or is that a waste of time?
        assert addresses.model_paths.streetcover.exists()
        gdf = gpd.read_file(addresses.model_paths.streetcover)
        assert len(gdf) == len(G.edges)
        assert gdf.geometry.area.sum() > 0

        # Test odd lanes formatting
        """Test the streetcover function with oddly formatted lanes."""
        nx.set_edge_attributes(G, "2;1", "lanes")
        _ = gu.calculate_streetcover(G, params, addresses)
        assert addresses.model_paths.streetcover.exists()
        gdf1 = gpd.read_file(addresses.model_paths.streetcover)

        nx.set_edge_attributes(G, 3, "lanes")
        _ = gu.calculate_streetcover(G, params, addresses)
        assert addresses.model_paths.streetcover.exists()
        gdf2 = gpd.read_file(addresses.model_paths.streetcover)

        assert gdf1.geometry.area.sum() == gdf2.geometry.area.sum()


def test_split_long_edges(street_network):
    """Test the split_long_edges function."""
    G, _ = street_network
    G = gu.assign_id(G)
    max_length = 40
    params = parameters.SubcatchmentDerivation(max_street_length=max_length)
    G = gu.split_long_edges(G, params)
    for u, v, data in G.edges(data=True):
        assert data["length"] <= (max_length * 2)


def test_derive_subcatchments(street_network):
    """Test the derive_subcatchments function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        addresses = FilePaths(
            base_dir=Path(temp_dir),
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            elevation=Path(__file__).parent / "test_data" / "elevation.tif",
            building=temp_path / "building.geojson",
            streetcover=temp_path / "building.geojson",
            subcatchments=temp_path / "subcatchments.geojson",
        )
        params = parameters.SubcatchmentDerivation()
        G, _ = street_network

        # mock up buildings
        eg_bldg = sgeom.Polygon(
            [
                (700291.346, 5709928.922),
                (700331.206, 5709927.815),
                (700321.610, 5709896.444),
                (700293.192, 5709900.503),
                (700291.346, 5709928.922),
            ]
        )
        gdf = gpd.GeoDataFrame(geometry=[eg_bldg], crs=G.graph["crs"])
        gdf.to_file(addresses.bbox_paths.building, driver="GeoJSON")

        G = gu.calculate_contributing_area(G, params, addresses)
        for u, v, data in G.edges(data=True):
            assert "contributing_area" in data.keys()
            assert isinstance(data["contributing_area"], float)

        for u, data in G.nodes(data=True):
            assert "contributing_area" in data.keys()
            assert isinstance(data["contributing_area"], float)


def test_set_elevation_and_slope(street_network):
    """Test the set_elevation, set_surface_slope, chahinian_slope function."""
    G, _ = street_network
    with tempfile.TemporaryDirectory() as temp_dir:
        addresses = FilePaths(
            base_dir=Path(temp_dir),
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            elevation=Path(__file__).parent / "test_data" / "elevation.tif",
        )
        G = gu.set_elevation(G, addresses)
        for id_, data in G.nodes(data=True):
            assert "surface_elevation" in data.keys()
            assert math.isfinite(data["surface_elevation"])
            assert data["surface_elevation"] > 0

        G = gu.set_surface_slope(G)
        for u, v, data in G.edges(data=True):
            assert "surface_slope" in data.keys()
            assert math.isfinite(data["surface_slope"])

        G = gu.set_chahinian_slope(G)
        for u, v, data in G.edges(data=True):
            assert "chahinian_slope" in data.keys()
            assert math.isfinite(data["chahinian_slope"])

        slope_vals = {-2: 1, 0.3: 0, 0.4: 0, 12: 1}
        for slope, expected in slope_vals.items():
            first_edge = list(G.edges)[0]
            G.edges[first_edge]["surface_slope"] = slope / 100
            G = gu.set_chahinian_slope(G)
            assert G.edges[first_edge]["chahinian_slope"] == expected


def test_chahinian_angle(street_network):
    """Test the chahinian_angle function."""
    G, _ = street_network
    G = gu.set_chahinian_angle(G)
    for u, v, data in G.edges(data=True):
        assert "chahinian_angle" in data.keys()
        assert math.isfinite(data["chahinian_angle"])


def test_calculate_weights(street_network):
    """Test the calculate_weights function."""
    G, _ = street_network
    params = parameters.TopologyDerivation()
    for weight in params.weights:
        for ix, (u, v, data) in enumerate(G.edges(data=True)):
            data[weight] = ix

    G = gu.calculate_weights(G, params)
    for u, v, data in G.edges(data=True):
        assert "weight" in data.keys()
        assert math.isfinite(data["weight"])


def test_calculate_weights_novar(street_network):
    """Test the calculate_weights function with no variance."""
    G, _ = street_network
    params = parameters.TopologyDerivation()
    for weight in params.weights:
        for ix, (u, v, data) in enumerate(G.edges(data=True)):
            data[weight] = 1.5

    G = gu.calculate_weights(G, params)
    for u, v, data in G.edges(data=True):
        assert "weight" in data.keys()
        assert math.isfinite(data["weight"])


def test_identify_outfalls_no_river(street_network):
    """Test the identify_outfalls in the no river case."""
    G, _ = street_network
    G = gu.assign_id(G)
    G = gu.double_directed(G)
    with tempfile.TemporaryDirectory() as temp_dir:
        addresses = FilePaths(
            base_dir=Path(temp_dir),
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            elevation=Path(__file__).parent / "test_data" / "elevation.tif",
        )
        G = gu.set_elevation(G, addresses)
        for ix, (u, v, d) in enumerate(G.edges(data=True)):
            d["edge_type"] = "street"
            d["weight"] = ix
        params = parameters.OutfallDerivation()
        G = gu.identify_outfalls(G, params)
        outfalls = [
            (u, v, d) for u, v, d in G.edges(data=True) if d["edge_type"] == "outfall"
        ]
        assert len(outfalls) == 1


def test_identify_outfalls_sg(street_network):
    """Test the identify_outfalls with subgraphs."""
    G, _ = street_network

    G = gu.assign_id(G)
    G = gu.double_directed(G)
    elev_fid = Path(__file__).parent / "test_data" / "elevation.tif"
    with tempfile.TemporaryDirectory() as temp_dir:
        addresses = FilePaths(
            base_dir=Path(temp_dir),
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            elevation=elev_fid,
        )
        G = gu.set_elevation(G, addresses)
        for ix, (u, v, d) in enumerate(G.edges(data=True)):
            d["edge_type"] = "street"
            d["weight"] = ix

        params = parameters.OutfallDerivation(
            river_buffer_distance=200, outfall_length=10, method="withtopo"
        )
        dummy_river1 = sgeom.LineString(
            [(699913.878, 5709769.851), (699932.546, 5709882.575)]
        )
        dummy_river2 = sgeom.LineString(
            [(699932.546, 5709882.575), (700011.524, 5710060.636)]
        )
        dummy_river3 = sgeom.LineString(
            [(700011.524, 5710060.636), (700103.427, 5710169.052)]
        )

        G.add_edge(
            "river1",
            "river2",
            **{
                "length": 10,
                "edge_type": "river",
                "id": "river1-to-river2",
                "geometry": dummy_river1,
            },
        )
        G.add_edge(
            "river2",
            "river3",
            **{
                "length": 10,
                "edge_type": "river",
                "id": "river2-to-river3",
                "geometry": dummy_river2,
            },
        )

        G.add_edge(
            "river3",
            "river4",
            **{
                "length": 10,
                "edge_type": "river",
                "id": "river3-to-river4",
                "geometry": dummy_river3,
            },
        )

        G.nodes["river1"]["x"] = 699913.878
        G.nodes["river1"]["y"] = 5709769.851
        G.nodes["river2"]["x"] = 699932.546
        G.nodes["river2"]["y"] = 5709882.575
        G.nodes["river3"]["x"] = 700011.524
        G.nodes["river3"]["y"] = 5710060.636
        G.nodes["river4"]["x"] = 700103.427
        G.nodes["river4"]["y"] = 5710169.052

        # Cut into subgraphs
        G.remove_edge(12354833, 25472373)
        G.remove_edge(25472373, 12354833)
        G.remove_edge(109753, 25472854)
        G.remove_edge(25472854, 109753)

        # Test outfall derivation
        G_ = G.copy()
        G_ = gu.identify_outfalls(G_, params)

        # Two subgraphs = two routes to waste
        outfalls = [
            (u, v, d)
            for u, v, d in G_.edges(data=True)
            if d["edge_type"] == "waste-outfall"
        ]
        assert len(outfalls) == 2

        # With buffer distance 300, the subgraph near the river will have an outfall
        # between the nearest street node to each river node (there are 3 potential
        # links in 150m). The subgraph further from the river is too far to be linked
        # to the river nodes and so will have a dummy river node as an outfall. 3+1=5
        outfalls = [
            (u, v, d) for u, v, d in G_.edges(data=True) if d["edge_type"] == "outfall"
        ]
        assert len(outfalls) == 3


def test_identify_outfalls_and_derive_topology(street_network):
    """Test the identify_outfalls and derive_topology functions."""
    G, _ = street_network
    G = gu.assign_id(G)
    G = gu.double_directed(G)
    for ix, (u, v, d) in enumerate(G.edges(data=True)):
        d["edge_type"] = "street"
        d["weight"] = ix

    params = parameters.OutfallDerivation(
        river_buffer_distance=200, outfall_length=10, method="separate"
    )
    dummy_river1 = sgeom.LineString(
        [(699913.878, 5709769.851), (699932.546, 5709882.575)]
    )
    dummy_river2 = sgeom.LineString(
        [(699932.546, 5709882.575), (700011.524, 5710060.636)]
    )
    dummy_river3 = sgeom.LineString(
        [(700011.524, 5710060.636), (700103.427, 5710169.052)]
    )

    G.add_edge(
        "river1",
        "river2",
        **{
            "length": 10,
            "edge_type": "river",
            "id": "river1-to-river2",
            "geometry": dummy_river1,
        },
    )
    G.add_edge(
        "river2",
        "river3",
        **{
            "length": 10,
            "edge_type": "river",
            "id": "river2-to-river3",
            "geometry": dummy_river2,
        },
    )

    G.add_edge(
        "river3",
        "river4",
        **{
            "length": 10,
            "edge_type": "river",
            "id": "river3-to-river4",
            "geometry": dummy_river3,
        },
    )

    G.nodes["river1"]["x"] = 699913.878
    G.nodes["river1"]["y"] = 5709769.851
    G.nodes["river2"]["x"] = 699932.546
    G.nodes["river2"]["y"] = 5709882.575
    G.nodes["river3"]["x"] = 700011.524
    G.nodes["river3"]["y"] = 5710060.636
    G.nodes["river4"]["x"] = 700103.427
    G.nodes["river4"]["y"] = 5710169.052

    # Test outfall derivation
    G_ = G.copy()
    G_ = gu.identify_outfalls(G_, params)

    outfalls = [
        (u, v, d) for u, v, d in G_.edges(data=True) if d["edge_type"] == "outfall"
    ]
    assert len(outfalls) == 2

    # Test topo derivation
    G_ = gu.derive_topology(G_, params)
    assert len(G_.edges) == 22
    assert len(set([d["outfall"] for u, d in G_.nodes(data=True)])) == 2
    for u, d in G_.nodes(data=True):
        assert "x" in d.keys()
        assert "y" in d.keys()

    # Test outfall derivation parameters
    G_ = G.copy()
    params.outfall_length = 600
    G_ = gu.identify_outfalls(G_, params)
    outfalls = [
        (u, v, d) for u, v, d in G_.edges(data=True) if d["edge_type"] == "outfall"
    ]
    assert len(outfalls) == 1


def test_identify_outfalls_and_derive_topology_withtopo(street_network):
    """Test the identify_outfalls and derive_topology functions."""
    G, _ = street_network
    G = gu.assign_id(G)
    G = gu.double_directed(G)
    for ix, (u, v, d) in enumerate(G.edges(data=True)):
        d["edge_type"] = "street"
        d["weight"] = ix

    params = parameters.OutfallDerivation(
        river_buffer_distance=250, outfall_length=10, method="withtopo"
    )
    dummy_river1 = sgeom.LineString(
        [(699913.878, 5709769.851), (699932.546, 5709882.575)]
    )
    dummy_river2 = sgeom.LineString(
        [(699932.546, 5709882.575), (700011.524, 5710060.636)]
    )
    dummy_river3 = sgeom.LineString(
        [(700011.524, 5710060.636), (700103.427, 5710169.052)]
    )

    G.add_edge(
        "river1",
        "river2",
        **{
            "length": 10,
            "edge_type": "river",
            "id": "river1-to-river2",
            "geometry": dummy_river1,
        },
    )
    G.add_edge(
        "river2",
        "river3",
        **{
            "length": 10,
            "edge_type": "river",
            "id": "river2-to-river3",
            "geometry": dummy_river2,
        },
    )

    G.add_edge(
        "river3",
        "river4",
        **{
            "length": 10,
            "edge_type": "river",
            "id": "river3-to-river4",
            "geometry": dummy_river3,
        },
    )

    G.nodes["river1"]["x"] = 699913.878
    G.nodes["river1"]["y"] = 5709769.851
    G.nodes["river2"]["x"] = 699932.546
    G.nodes["river2"]["y"] = 5709882.575
    G.nodes["river3"]["x"] = 700011.524
    G.nodes["river3"]["y"] = 5710060.636
    G.nodes["river4"]["x"] = 700103.427
    G.nodes["river4"]["y"] = 5710169.052

    # Test outfall derivation
    G_ = G.copy()
    G_ = gu.identify_outfalls(G_, params)

    outfalls = [
        (u, v, d)
        for u, v, d in G_.edges(data=True)
        if d["edge_type"] == "waste-outfall"
    ]
    assert len(outfalls) == 1

    outfalls = [
        (u, v, d) for u, v, d in G_.edges(data=True) if d["edge_type"] == "outfall"
    ]
    assert len(outfalls) == 4

    # Test topo derivation
    G_ = gu.derive_topology(G_, params)
    assert len(G_.edges) == 20
    assert len(set([d["outfall"] for u, d in G_.nodes(data=True)])) == 4

    # Test outfall derivation parameters
    G_ = G.copy()
    params.outfall_length = 600
    G_ = gu.identify_outfalls(G_, params)
    G_ = gu.derive_topology(G_, params)
    assert len(set([d["outfall"] for u, d in G_.nodes(data=True)])) == 1
    for u, d in G_.nodes(data=True):
        assert "x" in d.keys()
        assert "y" in d.keys()


def test_pipe_by_pipe():
    """Test the pipe_by_pipe function."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    for ix, (u, d) in enumerate(G.nodes(data=True)):
        d["surface_elevation"] = ix
        d["contributing_area"] = ix

    params = parameters.HydraulicDesign()

    G = gu.pipe_by_pipe(G, params)
    for u, v, d in G.edges(data=True):
        assert "diameter" in d.keys()
        assert d["diameter"] in params.diameters

    for u, d in G.nodes(data=True):
        assert "chamber_floor_elevation" in d.keys()
        assert math.isfinite(d["chamber_floor_elevation"])


def get_edge_types(G):
    """Get the edge types in the graph."""
    edge_types = set()
    for u, v, d in G.edges(data=True):
        if isinstance(d["highway"], list):
            edge_types.union(d["highway"])
        else:
            edge_types.add(d["highway"])
    return edge_types


def test_remove_non_pipe_allowable_links():
    """Test the remove_non_pipe_allowable_links function."""
    G = load_graph(Path(__file__).parent / "test_data" / "street_graph.json")
    # Ensure some invalid paths
    topology_params = parameters.TopologyDerivation(omit_edges=["primary", "bridge"])

    # Test that an edge has a non-None 'bridge' entry
    assert len(set([d.get("bridge", None) for u, v, d in G.edges(data=True)])) > 1

    # Test that an edge has a 'primary' entry under highway
    assert "primary" in get_edge_types(G)

    G_ = gu.remove_non_pipe_allowable_links(G, topology_params)
    assert "primary" not in get_edge_types(G_)
    assert len(set([d.get("bridge", None) for u, v, d in G_.edges(data=True)])) == 1


def test_iterate_graphfcns():
    """Test the iterate_graphfcns function."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")
    params = parameters.get_full_parameters()
    params["topology_derivation"].omit_edges = ["primary", "bridge"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        addresses = FilePaths(
            base_dir=temp_path,
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            model=temp_path,
        )

        G = iterate_graphfcns(
            G, ["assign_id", "remove_non_pipe_allowable_links"], params, addresses
        )
        for u, v, d in G.edges(data=True):
            assert "id" in d.keys()
        assert "primary" not in get_edge_types(G)
        assert len(set([d.get("bridge", None) for u, v, d in G.edges(data=True)])) == 1


def _remove_edges(G: nx.Graph, **kw):
    """Remove all edges from the graph.

    Args:
        G (nx.Graph): The graph to remove edges from.
        kw: Additional keyword arguments. (which are ignored)

    Returns:
        nx.Graph: The graph with no edges.
    """
    G.remove_edges_from(list(G.edges))
    return G


def test_iterate_graphfcns_noedges():
    """Test the iterate_graphfcns function for a graph with no edges."""
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        addresses = FilePaths(
            base_dir=temp_path,
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            model=temp_path,
        )
        os.environ["SWMMANYWHERE_VERBOSE"] = "true"
        original_function = gu["remove_non_pipe_allowable_links"]
        gu["remove_non_pipe_allowable_links"] = _remove_edges
        G = iterate_graphfcns(
            G, ["assign_id", "remove_non_pipe_allowable_links"], {}, addresses
        )
        gu["remove_non_pipe_allowable_links"] = original_function
        os.environ["SWMMANYWHERE_VERBOSE"] = "false"
        assert (addresses.model_paths.model / "assign_id_graph.json").exists()
        assert not (
            addresses.model_paths.model / "remove_non_pipe_allowable_links_graph.json"
        ).exists()


def test_fix_geometries():
    """Test the fix_geometries function."""
    # Create a graph with edge geometry not matching node coordinates
    G = load_graph(Path(__file__).parent / "test_data" / "graph_topo_derived.json")

    # Test doesn't work if this isn't true
    assert G.get_edge_data(107733, 25472373, 0)["geometry"].coords[0] != (
        G.nodes[107733]["x"],
        G.nodes[107733]["y"],
    )

    # Run the function
    G_fixed = gu.fix_geometries(G)

    # Check that the edge geometry now matches the node coordinates
    assert G_fixed.get_edge_data(107733, 25472373, 0)["geometry"].coords[0] == (
        G_fixed.nodes[107733]["x"],
        G_fixed.nodes[107733]["y"],
    )


def almost_equal(a, b, tol=1e-6):
    """Check if two numbers are almost equal."""
    return abs(a - b) < tol


def test_merge_street_nodes(street_network):
    """Test the merge_street_nodes function."""
    G, _ = street_network
    subcatchment_derivation = parameters.SubcatchmentDerivation(node_merge_distance=20)
    G_ = gu.merge_street_nodes(G, subcatchment_derivation)
    assert not set([107736, 266325461, 2623975694, 32925453]).intersection(G_.nodes)
    assert almost_equal(G_.nodes[25510321]["x"], 700445.0112082)


def test_clip_to_catchments(street_network):
    """Test the clip_to_catchments function."""
    G, _ = street_network

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["SWMMANYWHERE_VERBOSE"] = "true"
        temp_path = Path(temp_dir)
        addresses = FilePaths(
            base_dir=temp_path,
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            nodes=temp_path / "nodes.geojson",
            elevation=Path(__file__).parent / "test_data" / "elevation.tif",
        )

        # Test default clipping
        subcatchment_derivation = parameters.SubcatchmentDerivation()
        G_ = gu.clip_to_catchments(
            G, addresses=addresses, subcatchment_derivation=subcatchment_derivation
        )
        assert len(G_.edges) == 7

        # Test default clipping streamorder
        subcatchment_derivation = parameters.SubcatchmentDerivation()
        subcatchment_derivation.subbasin_streamorder = 4
        G_ = gu.clip_to_catchments(
            G, addresses=addresses, subcatchment_derivation=subcatchment_derivation
        )
        assert len(G_.edges) == 2

        # Test clipping
        subcatchment_derivation = parameters.SubcatchmentDerivation(
            subbasin_streamorder=3,
            subbasin_membership=0.9,
            subbasin_clip_method="community",
        )
        G_ = gu.clip_to_catchments(
            G, addresses=addresses, subcatchment_derivation=subcatchment_derivation
        )
        assert len(G_.edges) == 31

        # Test clipping with different params
        subcatchment_derivation = parameters.SubcatchmentDerivation(
            subbasin_streamorder=4,
            subbasin_membership=0.3,
            subbasin_clip_method="community",
        )
        G_ = gu.clip_to_catchments(
            G, addresses=addresses, subcatchment_derivation=subcatchment_derivation
        )
        assert len(G_.edges) == 38

        # Test no cuts
        subcatchment_derivation = parameters.SubcatchmentDerivation(
            subbasin_streamorder=4,
            subbasin_membership=0,
            subbasin_clip_method="community",
        )
        G_ = gu.clip_to_catchments(
            G, addresses=addresses, subcatchment_derivation=subcatchment_derivation
        )
        assert len(G_.edges) == 39

        # Cut between every community not entirely within the same basin
        subcatchment_derivation = parameters.SubcatchmentDerivation(
            subbasin_streamorder=4,
            subbasin_membership=1,
            subbasin_clip_method="community",
        )
        G_ = gu.clip_to_catchments(
            G, addresses=addresses, subcatchment_derivation=subcatchment_derivation
        )
        assert len(G_.edges) == 28

        # Check streamorder adjustment
        with tempfile.NamedTemporaryFile(
            suffix=".log", mode="w+b", delete=False
        ) as temp_file:
            fid = Path(temp_file.name)
            os.environ["SWMMANYWHERE_VERBOSE"] = "true"
            logger.add(fid)
            subcatchment_derivation = parameters.SubcatchmentDerivation(
                subbasin_streamorder=5, subbasin_membership=0.9
            )
            G_ = gu.clip_to_catchments(
                G, addresses=addresses, subcatchment_derivation=subcatchment_derivation
            )
            ftext = str(temp_file.read())
            assert """No subbasins found""" in ftext
            assert """WARNING""" in ftext
            logger.remove()
            os.environ["SWMMANYWHERE_VERBOSE"] = "false"
        assert (addresses.model_paths.nodes.parent / "subbasins.geojson").exists()


def test_filter_streets():
    """Test the _filter_streets function."""
    # Create a sample graph
    G = nx.Graph()
    G.add_edges_from(
        [
            (1, 2, {"edge_type": "street"}),
            (2, 3, {"edge_type": "street"}),
            (3, 4, {"edge_type": "outfall"}),
            (4, 5, {"edge_type": "river"}),
        ]
    )

    # Test case 1: Filter streets
    G_streets = filter_streets(G)
    assert set(G_streets.nodes) == {1, 2, 3}
    assert set(G_streets.edges) == {(1, 2), (2, 3)}

    # Test case 2: Empty graph
    G_empty = nx.Graph()
    G_empty_streets = filter_streets(G_empty)
    assert len(G_empty_streets.nodes) == 0
    assert len(G_empty_streets.edges) == 0

    # Test case 3: All non-street edges
    G_non_streets = nx.Graph()
    G_non_streets.add_edges_from(
        [(1, 2, {"edge_type": "non-street"}), (2, 3, {"edge_type": "non-street"})]
    )
    G_non_streets_filtered = filter_streets(G_non_streets)
    assert len(G_non_streets_filtered.nodes) == 0
    assert len(G_non_streets_filtered.edges) == 0


def test_validate_graphfcn_list(street_network):
    """Test the validate_graphfcn_list function."""
    # Test case 1: Valid list
    validate_graphfcn_list(["assign_id", "double_directed"])

    # Test case 2: Invalid list
    with pytest.raises(ValueError) as exc_info:
        validate_graphfcn_list(["assign_id", "not_a_function"])
    assert "not_a_function" in str(exc_info.value)

    # Test case 3: Valid order
    G, _ = street_network
    validate_graphfcn_list(["assign_id", "double_directed"], G)

    # Test case 4: Invalid order
    with pytest.raises(ValueError) as exc_info:
        validate_graphfcn_list(["assign_id", "calculate_weights"], G)
    assert "calculate_weights requires edge attributes" in str(exc_info.value)
    assert "chahinian_angle" in str(exc_info.value)

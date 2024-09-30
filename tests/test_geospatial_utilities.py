"""Misc line to trigger workflow."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import rasterio as rst
from scipy.interpolate import RegularGridInterpolator
from shapely import geometry as sgeom

from swmmanywhere import geospatial_utilities as go
from swmmanywhere import graph_utilities as ge
from swmmanywhere.logging import set_verbose
from swmmanywhere.misc.debug_derive_rc import derive_rc_alt


@pytest.fixture
def street_network():
    """Load a street network."""
    G = ge.load_graph(Path(__file__).parent / "test_data" / "street_graph.json")
    return G


def almost_equal(a, b, tol=1e-6):
    """Check if two numbers are almost equal."""
    if hasattr(a, "shape"):
        return ((a - b) < tol).all().all()
    return abs(a - b) < tol


def test_interp_with_nans():
    """Test the interp_interp_with_nans function."""
    # Define a simple grid and values
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    values = np.linspace(0, 1, 25)
    values_grid = values.reshape(5, 5)

    # Define an interpolator
    interp = RegularGridInterpolator((x, y), values_grid)

    # Test the function at a point inside the grid
    yx = (0.875, 0.875)
    result = go.interp_with_nans(yx, interp, grid, values)
    assert result == 0.875

    # Test the function on a nan point
    values_grid[1][1] = np.nan
    yx = (0.251, 0.25)
    result = go.interp_with_nans(yx, interp, grid, values)
    assert result == values_grid[1][2]


@patch("rasterio.open")
def test_interpolate_points_on_raster(mock_rst_open):
    """Test the interpolate_points_on_raster function."""
    # Mock the raster file
    mock_src = MagicMock()
    mock_src.read.return_value = np.array([[1, 2], [3, 4]])
    mock_src.bounds = MagicMock()
    mock_src.bounds.left = 0
    mock_src.bounds.right = 1
    mock_src.bounds.bottom = 0
    mock_src.bounds.top = 1
    mock_src.width = 2
    mock_src.height = 2
    mock_src.nodata = None
    mock_rst_open.return_value.__enter__.return_value = mock_src

    # Define the x and y coordinates
    x = [0.25, 0.75]
    y = [0.25, 0.75]

    # Call the function
    result = go.interpolate_points_on_raster(x, y, Path("fake_path"))

    # [2.75, 2.25] feels unintuitive but it's because rasters measure from the top
    assert result == [2.75, 2.25]


def test_get_utm():
    """Test the get_utm_epsg function."""
    # Test a northern hemisphere point
    crs = go.get_utm_epsg(-1.0, 51.0)
    assert crs == "EPSG:32630"

    # Test a southern hemisphere point
    crs = go.get_utm_epsg(-1.0, -51.0)
    assert crs == "EPSG:32730"


def create_raster(fid):
    """Define a function to create a mock raster file."""
    data = np.ones((100, 100))
    transform = rst.transform.from_origin(0, 0, 0.1, 0.1)
    with rst.open(
        fid,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as src:
        src.write(data, 1)


def test_reproject_raster():
    """Test the reproject_raster function."""
    # Create a mock raster file
    fid = Path("test.tif")
    try:
        create_raster(fid)

        # Define the input parameters
        target_crs = "EPSG:32630"
        new_fid = Path("test_reprojected.tif")

        # Call the function
        go.reproject_raster(target_crs, fid)

        # Check if the reprojected file exists
        assert new_fid.exists()

        # Check if the reprojected file has the correct CRS
        with rst.open(new_fid) as src:
            assert src.crs.to_string() == target_crs
    finally:
        # Regardless of test outcome, delete the temp file
        fid.unlink(missing_ok=True)
        new_fid.unlink(missing_ok=True)


def test_get_transformer():
    """Test the get_transformer function."""
    # Test a northern hemisphere point
    transformer = go.get_transformer("EPSG:4326", "EPSG:32630")

    initial_point = (-0.1276, 51.5074)
    expected_point = (699330.1106898375, 5710164.30300683)
    new_point = transformer.transform(*initial_point)
    assert almost_equal(new_point[0], expected_point[0])
    assert almost_equal(new_point[1], expected_point[1])


def test_reproject_graph():
    """Test the reproject_graph function."""
    # Create a mock graph
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2)
    G.add_node(3, x=1, y=2)
    G.add_edge(2, 3, geometry=sgeom.LineString([(1, 1), (1, 2)]))

    # Define the input parameters
    source_crs = "EPSG:4326"
    target_crs = "EPSG:32630"

    # Call the function
    G_new = go.reproject_graph(G, source_crs, target_crs)

    # Test node coordinates
    assert almost_equal(G_new.nodes[1]["x"], 833978.5569194595)
    assert almost_equal(G_new.nodes[1]["y"], 0)
    assert almost_equal(G_new.nodes[2]["x"], 945396.6839773951)
    assert almost_equal(G_new.nodes[2]["y"], 110801.83254625657)
    assert almost_equal(G_new.nodes[3]["x"], 945193.8596723974)
    assert almost_equal(G_new.nodes[3]["y"], 221604.0105092727)

    # Test edge geometry
    assert almost_equal(list(G_new[1][2]["geometry"].coords)[0][0], 833978.5569194595)
    assert almost_equal(list(G_new[2][3]["geometry"].coords)[0][0], 945396.6839773951)


def test_nearest_node_buffer():
    """Test the nearest_node_buffer function."""
    # Create mock dictionaries of points
    points1 = {"a": sgeom.Point(0, 0), "b": sgeom.Point(1, 1)}
    points2 = {"c": sgeom.Point(0.5, 0.5), "d": sgeom.Point(2, 2)}

    # Define the input threshold
    threshold = 1.0

    # Call the function
    matching = go.nearest_node_buffer(points1, points2, threshold)

    # Check if the function returns the correct matching nodes
    assert matching == {"a": "c", "b": "c"}


def test_burn_shape_in_raster():
    """Test the burn_shape_in_raster function."""
    # Create a mock geometry
    geoms = [
        sgeom.LineString([(0, 0), (1, 1)]),
        sgeom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ]

    # Define the input parameters
    depth = 1.0
    raster_fid = Path("input.tif")
    new_raster_fid = Path("output.tif")
    try:
        create_raster(raster_fid)

        # Call the function
        go.burn_shape_in_raster(geoms, depth, raster_fid, new_raster_fid)

        with rst.open(raster_fid) as src:
            data_ = src.read(1)

        # Open the new GeoTIFF file and check if it has been correctly modified
        with rst.open(new_raster_fid) as src:
            data = src.read(1)
            assert (data != data_).any()
    finally:
        # Regardless of test outcome, delete the temp file
        raster_fid.unlink(missing_ok=True)
        new_raster_fid.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "method,area,slope,width",
    [("pyflwdir", 2498, 0.1187, 28.202), ("whitebox", 2998, 0.1102, 30.894)],
)
@pytest.mark.parametrize("verbose", [True, False])
def test_derive_subcatchments(street_network, method, area, slope, width, verbose):
    """Test the derive_subcatchments function."""
    set_verbose(verbose)

    elev_fid = Path(__file__).parent / "test_data" / "elevation.tif"

    polys = go.derive_subcatchments(street_network, elev_fid, method)
    assert "slope" in polys.columns
    assert "area" in polys.columns
    assert "geometry" in polys.columns
    assert "id" in polys.columns
    assert polys.shape[0] > 0
    assert polys.dropna().shape == polys.shape
    assert polys.crs == street_network.graph["crs"]

    assert almost_equal(polys.set_index("id").loc[2623975694, "area"], area, tol=1)
    assert almost_equal(
        polys.set_index("id").loc[2623975694, "slope"], slope, tol=0.001
    )
    assert almost_equal(
        polys.set_index("id").loc[2623975694, "width"], width, tol=0.001
    )


def test_derive_rc(street_network):
    """Test the derive_rc function."""
    crs = street_network.graph["crs"]
    eg_bldg = sgeom.Polygon(
        [
            (700291, 5709928),
            (700331, 5709927),
            (700321, 5709896),
            (700293, 5709900),
            (700291, 5709928),
        ]
    )
    buildings = gpd.GeoDataFrame(geometry=[eg_bldg], crs=crs)
    subs = [
        sgeom.Polygon(
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
        sgeom.Polygon(
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
        sgeom.Polygon(
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

    streetcover = [
        d["geometry"].buffer(5) for u, v, d in street_network.edges(data=True)
    ]
    streetcover = gpd.GeoDataFrame(geometry=streetcover, crs=crs)

    subs = gpd.GeoDataFrame(
        data={"id": [107733, 1696030874, 6277683849]}, geometry=subs, crs=crs
    )
    subs["area"] = subs.geometry.area

    # Test no RC
    subs_rc = go.derive_rc(subs, buildings, buildings).set_index("id")
    assert subs_rc.loc[6277683849, "impervious_area"] == 0
    assert subs_rc.loc[107733, "impervious_area"] > 0

    # Test alt method
    subs_rc_alt = derive_rc_alt(subs, buildings, buildings).set_index("id")
    assert almost_equal(
        subs_rc[["area", "impervious_area", "rc"]],
        subs_rc_alt[["area", "impervious_area", "rc"]],
    )

    # Test some RC
    subs_rc = go.derive_rc(subs, buildings, streetcover).set_index("id")
    assert almost_equal(subs_rc.loc[6277683849, "impervious_area"], 1092.452579)
    assert almost_equal(subs_rc.loc[6277683849, "rc"], 21.770677)
    assert subs_rc.rc.max() <= 100

    # Test alt method
    subs_rc_alt = derive_rc_alt(subs, buildings, streetcover).set_index("id")
    assert almost_equal(
        subs_rc[["area", "impervious_area", "rc"]],
        subs_rc_alt[["area", "impervious_area", "rc"]],
    )

    # Test intersecting buildings and streets
    buildings = buildings.overlay(streetcover.dissolve(), how="union")
    subs_rc = go.derive_rc(subs, buildings, streetcover).set_index("id")
    assert almost_equal(subs_rc.loc[6277683849, "impervious_area"], 1092.452579)
    assert almost_equal(subs_rc.loc[6277683849, "rc"], 21.770677)
    assert subs_rc.rc.max() <= 100

    # Test alt method
    subs_rc_alt = derive_rc_alt(subs, buildings, streetcover).set_index("id")
    assert almost_equal(
        subs_rc[["area", "impervious_area", "rc"]],
        subs_rc_alt[["area", "impervious_area", "rc"]],
    )


def test_calculate_angle():
    """Test the calculate_angle function."""
    # Test with points forming a right angle
    point1 = (0, 0)
    point2 = (1, 0)
    point3 = (1, 1)
    assert go.calculate_angle(point1, point2, point3) == 90

    # Test with points forming a straight line
    point1 = (0, 0)
    point2 = (1, 0)
    point3 = (2, 0)
    assert go.calculate_angle(point1, point2, point3) == 180

    # Test with points forming an angle of 45 degrees
    point1 = (0, 0)
    point2 = (1, 0)
    point3 = (0, 1)
    assert almost_equal(go.calculate_angle(point1, point2, point3), 45)

    # Test with points forming an angle of 0 degrees
    point1 = (0, 0)
    point2 = (1, 0)
    point3 = (0, 0)
    assert go.calculate_angle(point1, point2, point3) == 0


def test_remove_intersections():
    """Test the remove_intersections function."""
    square1 = {"id": "s1", "geometry": sgeom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])}
    square2 = {
        "id": "s2",
        "geometry": sgeom.Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
    }
    square3 = {
        "id": "s3",
        "geometry": sgeom.Polygon([(0.25, 0.25), (0.5, 0.25), (0.5, 0.5), (0.25, 0.5)]),
    }
    square4 = {
        "id": "s4",
        "geometry": sgeom.Polygon([(0.75, 0.75), (1, 0.75), (1, 1), (0.75, 1)]),
    }
    polys = gpd.GeoDataFrame([square1, square2, square3, square4])

    target1 = {
        "id": "s1",
        "geometry": sgeom.Polygon(
            [
                (1.0, 0.0),
                (0.5, 0.0),
                (0.5, 0.25),
                (0.5, 0.5),
                (0.25, 0.5),
                (0.0, 0.5),
                (0.0, 1.0),
                (0.75, 1.0),
                (0.75, 0.75),
                (1.0, 0.75),
                (1.0, 0.0),
            ]
        ),
    }
    target2 = {
        "id": "s2",
        "geometry": sgeom.Polygon(
            [
                (0.5, 0),
                (0, 0),
                (0, 0.5),
                (0.25, 0.5),
                (0.25, 0.25),
                (0.5, 0.25),
                (0.5, 0),
            ]
        ),
    }
    targets = gpd.GeoDataFrame([target1, target2, square3, square4])

    polys_ = go.remove_intersections(polys)

    polys_["area"] = polys_.geometry.area
    targets["area"] = targets.geometry.area
    assert polys_.set_index("id")[["area"]].equals(targets.set_index("id")[["area"]])


def test_graph_to_geojson(street_network):
    """Test the graph_to_geojson function."""
    crs = street_network.graph["crs"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        go.graph_to_geojson(
            street_network,
            temp_path / "graph_nodes.geojson",
            temp_path / "graph_edges.geojson",
            crs,
        )
        gdf = gpd.read_file(temp_path / "graph_nodes.geojson")
        assert gdf.crs == crs
        assert gdf.shape[0] == len(street_network.nodes)

        gdf = gpd.read_file(temp_path / "graph_edges.geojson")
        assert gdf.shape[0] == len(street_network.edges)


def test_merge_points(street_network):
    """Test the merge_points function."""
    mapping = go.merge_points(
        [(d["x"], d["y"]) for u, d in street_network.nodes(data=True)], 20
    )
    assert set(mapping.keys()) == set([2, 3, 5, 15, 16, 18, 22])
    assert set([x["maps_to"] for x in mapping.values()]) == set([2, 5, 15])
    assert mapping[15]["maps_to"] == 15
    assert mapping[18]["maps_to"] == 15
    assert almost_equal(mapping[18]["coordinate"][0], 700445.0112082)

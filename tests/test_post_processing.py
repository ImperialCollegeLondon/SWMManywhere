from __future__ import annotations

import difflib
import filecmp
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyswmm
from shapely import geometry as sgeom

from swmmanywhere import post_processing as stt
from swmmanywhere.filepaths import FilePaths

fid = (
    Path(__file__).parent.parent
    / "src"
    / "swmmanywhere"
    / "defs"
    / "basic_drainage_all_bits.inp"
)


def test_overwrite_section():
    """Test the overwrite_section function.

    All this tests is that the data is written to the file.
    """
    # Copy the example file to a temporary file

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fname = Path(temp_dir) / "temp.inp"
        shutil.copy(fid, temp_fname)
        data = [
            ["1", "1", "1", "1.166", "100", "500", "0.5", "0", "empty"],
            ["2", "1", "1", "1.1", "100", "500", "0.5", "0", "empty"],
            ["subca_3", "1", "1", "2", "100", "400", "0.5", "0", "empty"],
        ]

        section = "[SUBCATCHMENTS]"
        stt.overwrite_section(data, section, temp_fname)
        with temp_fname.open("r") as file:
            content = file.read()
        assert "subca_3" in content


def test_change_flow_routing():
    """Test the change_flow_routing function."""
    # Copy the example file to a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / "temp.inp"
        shutil.copy(fid, temp_fid)
        new_routing = "STEADY"
        stt.change_flow_routing(new_routing, temp_fid)
        with temp_fid.open("r") as file:
            content = file.read()
        assert "STEADY" in content


def test_data_input_dict_to_inp():
    """Test the data_input_dict_to_inp function.

    All this tests is that the data is written to a new file.
    """
    data_dict = {
        "SUBCATCHMENTS": [
            ["1", "1", "1", "1.166", "100", "500", "0.5", "0", "empty"],
            ["2", "1", "1", "1.1", "100", "500", "0.5", "0", "empty"],
            ["subca_3", "1", "1", "2", "100", "400", "0.5", "0", "empty"],
        ]
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / "temp.inp"
        shutil.copy(fid, temp_fid)
        stt.data_dict_to_inp(data_dict, fid, temp_fid)
        with temp_fid.open("r") as file:
            content = file.read()
        assert "subca_3" in content


def test_explode_polygon():
    """Test the explode_polygon function."""
    df = pd.Series(
        {
            "subcatchment": "1",
            "geometry": sgeom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        }
    )
    result = stt.explode_polygon(df)
    assert result.shape[0] == 5
    assert result.loc[3, "y"] == 1


def generate_data_dict():
    """Generate a data dict for testing."""
    nodes = gpd.GeoDataFrame(
        {
            "id": ["node1", "node2"],
            "x": [0, 1],
            "y": [0, 1],
            "max_depth": [1, 1],
            "chamber_floor_elevation": [1, 1],
            "surface_elevation": [2, 2],
            "manhole_area": [0.5, 0.5],
        }
    )
    outfalls = gpd.GeoDataFrame(
        {
            "id": ["node2_outfall"],
            "chamber_floor_elevation": [-4],
            "surface_elevation": [-4],
            "x": [-49],
            "y": [-49],
        }
    )
    conduits = gpd.GeoDataFrame(
        {
            "id": ["node1-node2", "node2-node2_outfall"],
            "u": ["node1", "node2"],
            "v": ["node2", "node2_outfall"],
            "length": [1, (50**2 + 50**2) ** 0.5],
            "roughness": [0.01, 0.01],
            "shape_swmm": ["CIRCULAR", "CIRCULAR"],
            "diameter": [1, 15],
            "capacity": [1e10, 1e10],
        }
    )
    subs = gpd.GeoDataFrame(
        {
            "subcatchment": ["node1-sub"],
            "rain_gage": ["1"],
            "id": ["node1"],
            "area": [1],
            "rc": [1],
            "width": [1],
            "slope": [0.001],
            "geometry": [sgeom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        }
    )
    rain_fid = "storm.dat"
    event = {"name": "1", "unit": "mm", "interval": "00:05", "fid": rain_fid}
    symbol = {"x": 0, "y": 0, "name": "1"}

    return {
        "nodes": nodes,
        "outfalls": outfalls,
        "conduits": conduits,
        "subs": subs,
        "event": event,
        "symbol": symbol,
    }


def test_synthetic_write():
    """Test the synthetic_write function."""
    data_dict = generate_data_dict()
    with tempfile.TemporaryDirectory() as model_dir:
        model_dir = Path(model_dir)
        addresses = FilePaths(
            base_dir=model_dir,
            bbox_bounds=[0, 1, 0, 1],
            project_name="test",
            extension="json",
            precipitation="storm.dat",
        )
        # Write the model with synthetic_write
        nodes = gpd.GeoDataFrame(data_dict["nodes"])
        nodes.geometry = gpd.points_from_xy(nodes.x, nodes.y)
        nodes.to_file(addresses.model_paths.nodes)
        nodes = nodes.set_index("id")
        edges = gpd.GeoDataFrame(pd.DataFrame(data_dict["conduits"]).iloc[[0]])
        edges.geometry = [
            sgeom.LineString([nodes.loc[u, "geometry"], nodes.loc[v, "geometry"]])
            for u, v in zip(edges.u, edges.v)
        ]
        edges.to_file(addresses.model_paths.edges)
        subs = data_dict["subs"].copy()
        subs["subcatchment"] = ["node1"]
        subs.to_file(addresses.model_paths.subcatchments)
        stt.synthetic_write(addresses)

        # Write the model with data_dict_to_inp
        comparison_file = addresses.model_paths.model / "model_base.inp"
        template_fid = (
            Path(__file__).parent.parent
            / "src"
            / "swmmanywhere"
            / "defs"
            / "basic_drainage_all_bits.inp"
        )

        # Manually convert to ha to make comparable to synthetic_write
        data_dict["subs"]["area"] /= 10000

        # Write the model with data_dict_to_inp
        stt.data_dict_to_inp(
            stt.format_to_swmm_dict(**data_dict), template_fid, comparison_file
        )

        # Compare
        new_input_file = addresses.model_paths.inp
        are_files_identical = filecmp.cmp(
            new_input_file, comparison_file, shallow=False
        )
        if not are_files_identical:
            with new_input_file.open("r") as file1, comparison_file.open("r") as file2:
                diff = difflib.unified_diff(
                    file1.readlines(),
                    file2.readlines(),
                    fromfile=str(new_input_file),
                    tofile=str(comparison_file),
                )
            print("".join(diff))
        assert are_files_identical, "The files are not identical"

        # Test that it doesn't break there are more outfalls than links
        nodes.loc["new_node"] = nodes.iloc[0].copy()
        nodes.reset_index().to_file(addresses.model_paths.nodes)

        # Check that there will be more outfalls than edges
        assert sum(~nodes.index.astype(str).isin(edges.u.astype(str))) > edges.shape[0]

        # Run to check
        stt.synthetic_write(addresses)


def test_format_to_swmm_dict():
    """Test the format_format_to_swmm_dict function.

    Writes a formatted dict to a model and checks that it runs without
    error.
    """
    data_dict = generate_data_dict()
    data_dict = stt.format_to_swmm_dict(**data_dict)

    rain_fid = (
        Path(__file__).parent.parent / "src" / "swmmanywhere" / "defs" / "storm.dat"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        shutil.copy(rain_fid, temp_dir / "storm.dat")
        temp_fid = temp_dir / "temp.inp"
        stt.data_dict_to_inp(data_dict, fid, temp_fid)
        with pyswmm.Simulation(str(temp_fid)) as sim:
            for ind, step in enumerate(sim):
                pass

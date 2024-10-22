"""Tests for the main module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import jsonschema
import pytest
import yaml

from swmmanywhere import swmmanywhere
from swmmanywhere.graph_utilities import graphfcns
from swmmanywhere.metric_utilities import metrics
from swmmanywhere.utilities import plot_basic, plot_map


def test_run():
    """Test the run function."""
    demo_dir = Path(__file__).parent.parent / "src" / "swmmanywhere" / "defs"
    model = demo_dir / "basic_drainage_all_bits.inp"
    storevars = ["flooding", "flow", "runoff", "depth"]
    results = swmmanywhere.run(model, reporting_iters=50, storevars=storevars)
    assert set(results.variable.unique()) == set(storevars)

    # Ensure more reporting iterations results in more results
    results_ = swmmanywhere.run(model, reporting_iters=25, storevars=storevars)
    assert results_.shape[0] > results.shape[0]

    # Ensure a shorter duration results in fewer results
    results_ = swmmanywhere.run(model, duration=10000, storevars=storevars)
    assert results_.shape[0] < results.shape[0]

    model.with_suffix(".out").unlink()
    model.with_suffix(".rpt").unlink()


def test_swmmanywhere():
    """Test the swmmanywhere function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load the config
        test_data_dir = Path(__file__).parent / "test_data"
        defs_dir = Path(__file__).parent.parent / "src" / "swmmanywhere" / "defs"
        with (defs_dir / "demo_config.yml").open("r") as f:
            config = yaml.safe_load(f)

        # Set some test values
        base_dir = Path(temp_dir)
        config["base_dir"] = str(base_dir)
        config["bbox"] = [0.05677, 51.55656, 0.07193, 51.56726]
        config["address_overrides"] = {
            "building": str(test_data_dir / "building.geoparquet"),
        }
        config["parameter_overrides"] = {
            "subcatchment_derivation": {"subbasin_streamorder": 5}
        }
        config["run_settings"]["duration"] = 1000
        config["model_number"] = 0

        # Fill the real dict with unused paths to avoid filevalidation errors
        config["real"]["subcatchments"] = str(defs_dir / "storm.dat")
        config["real"]["inp"] = str(defs_dir / "storm.dat")
        config["real"]["graph"] = str(defs_dir / "storm.dat")

        # Write the config
        with open(base_dir / "test_config.yml", "w") as f:
            yaml.dump(config, f)

        # Load and test validation of the config
        config = swmmanywhere.load_config(base_dir / "test_config.yml")

        # Set the test config to just use the generated data
        model_dir = base_dir / "demo" / "bbox_1" / "model_0"
        config["real"]["subcatchments"] = model_dir / "subcatchments.geoparquet"
        config["real"]["inp"] = model_dir / "model_0.inp"
        config["real"]["graph"] = model_dir / "graph.parquet"

        # Run swmmanywhere
        os.environ["SWMMANYWHERE_VERBOSE"] = "true"
        inp, metrics = swmmanywhere.swmmanywhere(config)

        # Check metrics were calculated
        assert metrics is not None
        for key, val in metrics.items():
            if not val:
                continue
            assert isinstance(val, float)

        assert set(metrics.keys()) == set(config["metric_list"])

        # Check results were saved
        assert (inp.parent / f'{config["graphfcn_list"][-1]}_graph.json').exists()
        assert inp.exists()
        assert (inp.parent / "results.parquet").exists()
        assert (config["real"]["inp"].parent / "real_results.parquet").exists()

        # Check the map functions
        plot_basic(inp.parent)
        plot_map(inp.parent)


def test_load_config_file_validation():
    """Test the file validation of the config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        defs_dir = Path(__file__).parent.parent / "src" / "swmmanywhere" / "defs"
        base_dir = Path(temp_dir)

        # Test file not found
        with pytest.raises(FileNotFoundError) as exc_info:
            swmmanywhere.load_config(base_dir / "test_config.yml")
            assert "test_config.yml" in str(exc_info.value)

        with (defs_dir / "demo_config.yml").open("r") as f:
            config = yaml.safe_load(f)

        # Correct and avoid filevalidation errors
        config["real"] = None

        # Fill with unused paths to avoid filevalidation errors
        config["base_dir"] = str(defs_dir / "storm.dat")

        with open(base_dir / "test_config.yml", "w") as f:
            yaml.dump(config, f)

        config = swmmanywhere.load_config(base_dir / "test_config.yml")
        assert isinstance(config, dict)


def test_load_config_schema_validation():
    """Test the schema validation of the config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        defs_dir = Path(__file__).parent.parent / "src" / "swmmanywhere" / "defs"
        base_dir = Path(temp_dir)

        # Load the config
        with (defs_dir / "demo_config.yml").open("r") as f:
            config = yaml.safe_load(f)

        # Make an edit not to schema
        config["base_dir"] = 1

        with open(base_dir / "test_config.yml", "w") as f:
            yaml.dump(config, f)

        # Test schema validation
        with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
            swmmanywhere.load_config(base_dir / "test_config.yml")
            assert "null" in str(exc_info.value)


def test_save_config():
    """Test the save_config function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        defs_dir = Path(__file__).parent.parent / "src" / "swmmanywhere" / "defs"

        with (defs_dir / "demo_config.yml").open("r") as f:
            config = yaml.safe_load(f)

        # Correct and avoid filevalidation errors
        config["real"] = None

        # Fill with unused paths to avoid filevalidation errors
        config["base_dir"] = str(defs_dir / "storm.dat")

        swmmanywhere.save_config(config, temp_dir / "test.yml")

        # Reload to check OK
        config = swmmanywhere.load_config(temp_dir / "test.yml")


@pytest.mark.downloads
def test_minimal_req():
    """Test SWMManywhere with minimal info."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            "base_dir": Path(temp_dir),
            "project": "my_test",
            "bbox": [1.52740, 42.50524, 1.54273, 42.51259],
        }

        swmmanywhere.swmmanywhere(config)


def test_custom_metric():
    """Test adding a custom metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load the config
        gf_module = str(Path(__file__).parent / "test_data" / "custom_metrics.py")

        config = swmmanywhere.load_config(validation=False)
        config["custom_metric_modules"] = [str(gf_module)]
        config["metric_list"].append("new_metric")

        # Set some test values
        config_address = Path(temp_dir) / "test_config.yml"
        config["base_dir"] = temp_dir
        config["bbox"] = [0, 1, 0, 1]
        del config["real"]

        # Write the config
        swmmanywhere.save_config(config, config_address)

        # Import of `custom_metric` by CI testing environment adds `new_metric`
        if "new_metric" in metrics:
            del metrics["new_metric"]

        # Load and test validation of the config
        config = swmmanywhere.load_config(config_address)

        # Check metric was added
        assert "new_metric" in metrics

        # Remove the custom metric for other tests
        del metrics["new_metric"]


def test_custom_graphfcn():
    """Test adding a custom graphfcn."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load the config
        gf_module = str(Path(__file__).parent / "test_data" / "custom_graphfcns.py")

        config = swmmanywhere.load_config(validation=False)
        config["custom_graphfcn_modules"] = [str(gf_module)]
        config["graphfcn_list"].append("new_graphfcn")

        # Set some test values
        config_address = Path(temp_dir) / "test_config.yml"
        config["base_dir"] = temp_dir
        config["bbox"] = [0, 1, 0, 1]
        del config["real"]

        # Write the config
        swmmanywhere.save_config(config, config_address)

        # Import of `custom_graphfcns` by CI testing environment adds `new_graphfcn`
        if "new_graphfcn" in graphfcns:
            del graphfcns["new_graphfcn"]

        # Load and test validation of the config
        config = swmmanywhere.load_config(config_address)

        # Check graphfcn was added
        assert "new_graphfcn" in graphfcns

        # Remove the custom graphfcn for other tests
        del graphfcns["new_graphfcn"]

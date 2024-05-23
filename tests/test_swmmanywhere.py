"""Tests for the main module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import jsonschema
import pytest
import yaml

from swmmanywhere import __version__, swmmanywhere


def test_version():
    """Check that the version is acceptable."""
    assert __version__ == "0.1.0"


def test_run():
    """Test the run function."""
    demo_dir = Path(__file__).parent.parent / 'swmmanywhere' / 'defs'
    model =  demo_dir / 'basic_drainage_all_bits.inp'
    storevars = ['flooding','flow','runoff','depth']
    results = swmmanywhere.run(model,
                               reporting_iters = 50,
                               storevars = storevars)
    assert set(results.variable.unique()) == set(storevars)
    
    # Ensure more reporting iterations results in more results
    results_ = swmmanywhere.run(model,
                               reporting_iters = 25,
                               storevars = storevars)
    assert results_.shape[0] > results.shape[0]

    # Ensure a shorter duration results in fewer results
    results_ = swmmanywhere.run(model,
                               duration = 10000,
                               storevars = storevars)
    assert results_.shape[0] < results.shape[0]

    model.with_suffix('.out').unlink()
    model.with_suffix('.rpt').unlink()

def test_swmmanywhere():
    """Test the swmmanywhere function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load the config
        test_data_dir = Path(__file__).parent / 'test_data'
        defs_dir = Path(__file__).parent.parent / 'swmmanywhere' / 'defs'
        with (test_data_dir / 'demo_config.yml').open('r') as f:
            config = yaml.safe_load(f)

        # Set some test values
        base_dir = Path(temp_dir)
        config['base_dir'] = str(base_dir)
        config['bbox'] = [0.05428,51.55847,0.07193,51.56726]
        config['address_overrides'] = {
            'building': str(test_data_dir / 'building.geoparquet'),
            'precipitation': str(defs_dir / 'storm.dat')
            }
        
        config['run_settings']['duration'] = 1000
        api_keys = {'nasadem_key' : 'b206e65629ac0e53d599e43438560d28'}
        with open(base_dir / 'api_keys.yml', 'w') as f:
            yaml.dump(api_keys, f)
        config['api_keys'] = str(base_dir / 'api_keys.yml')
        
        # Fill the real dict with unused paths to avoid filevalidation errors
        config['real']['subcatchments'] = str(defs_dir / 'storm.dat')
        config['real']['inp'] = str(defs_dir / 'storm.dat')
        config['real']['graph'] = str(defs_dir / 'storm.dat')

        # Write the config
        with open(base_dir / 'test_config.yml', 'w') as f:
            yaml.dump(config, f)
        
        # Load and test validation of the config
        config = swmmanywhere.load_config(base_dir / 'test_config.yml')

        # Set the test config to just use the generated data
        model_dir = base_dir / 'demo' / 'bbox_1' / 'model_1'
        config['real']['subcatchments'] = model_dir / 'subcatchments.geoparquet'
        config['real']['inp'] = model_dir / 'model_1.inp'
        config['real']['graph'] = model_dir / 'graph.parquet'

        # Run swmmanywhere
        os.environ["SWMMANYWHERE_VERBOSE"] = "true"
        inp, metrics = swmmanywhere.swmmanywhere(config)

        # Check metrics were calculated
        assert metrics is not None
        for key, val in metrics.items():
            if not val:
                continue
            assert isinstance(val, float)
        
        assert set(metrics.keys()) == set(config['metric_list'])

        # Check results were saved
        assert (inp.parent / f'{config["graphfcn_list"][-1]}_graph.json').exists()
        assert inp.exists()
        assert (inp.parent / 'results.parquet').exists()
        assert (config['real']['inp'].parent / 'real_results.parquet').exists()

def test_load_config_file_validation():
    """Test the file validation of the config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = Path(__file__).parent / 'test_data'
        defs_dir = Path(__file__).parent.parent / 'swmmanywhere' / 'defs'
        base_dir = Path(temp_dir)
        
        # Test file not found
        with pytest.raises(FileNotFoundError) as exc_info:
            swmmanywhere.load_config(base_dir / 'test_config.yml')
            assert "test_config.yml" in str(exc_info.value)

        with (test_data_dir / 'demo_config.yml').open('r') as f:
            config = yaml.safe_load(f)
        
        # Correct and avoid filevalidation errors
        config['real'] = None
        
        # Fill with unused paths to avoid filevalidation errors
        config['base_dir'] = str(defs_dir / 'storm.dat')
        config['api_keys'] = str(defs_dir / 'storm.dat')
        
        with open(base_dir / 'test_config.yml', 'w') as f:
            yaml.dump(config, f)
        
        config = swmmanywhere.load_config(base_dir / 'test_config.yml')
        assert isinstance(config, dict)

def test_load_config_schema_validation():
    """Test the schema validation of the config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = Path(__file__).parent / 'test_data'
        base_dir = Path(temp_dir)

        # Load the config
        with (test_data_dir / 'demo_config.yml').open('r') as f:
            config = yaml.safe_load(f)
        
        # Make an edit not to schema
        config['base_dir'] = 1
        
        with open(base_dir / 'test_config.yml', 'w') as f:
            yaml.dump(config, f)

        # Test schema validation
        with pytest.raises(jsonschema.exceptions.ValidationError) as exc_info:
            swmmanywhere.load_config(base_dir / 'test_config.yml')
            assert "null" in str(exc_info.value)

def test_check_parameters_to_sample():
    """Test the check_parameters_to_sample validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_dir = Path(__file__).parent / 'test_data'
        defs_dir = Path(__file__).parent.parent / 'swmmanywhere' / 'defs'
        base_dir = Path(temp_dir)

        # Load the config
        with (test_data_dir / 'demo_config.yml').open('r') as f:
            config = yaml.safe_load(f)
        
        # Correct and avoid filevalidation errors
        config['real'] = None
        
        # Fill with unused paths to avoid filevalidation errors
        config['base_dir'] = str(defs_dir / 'storm.dat')
        config['api_keys'] = str(defs_dir / 'storm.dat')

        # Make an edit that should fail
        config['parameters_to_sample'] = ['not_a_parameter']
        
        with open(base_dir / 'test_config.yml', 'w') as f:
            yaml.dump(config, f)

        # Test parameter validation
        with pytest.raises(ValueError) as exc_info:
            swmmanywhere.load_config(base_dir / 'test_config.yml')
        assert "not_a_parameter" in str(exc_info.value)

        # Test parameter_overrides invalid category
        config['parameter_overrides'] = {'fake_category' : {'fake_parameter' : 0}}
        with pytest.raises(ValueError) as exc_info:
            swmmanywhere.check_parameter_overrides(config)
        assert "fake_category not a category" in str(exc_info.value)

        # Test parameter_overrides invalid parameter
        config['parameter_overrides'] = {'hydraulic_design' : {'fake_parameter' : 0}}
        with pytest.raises(ValueError) as exc_info:
            swmmanywhere.check_parameter_overrides(config)
        assert "fake_parameter not found" in str(exc_info.value)
        
        # Test parameter_overrides valid
        config['parameter_overrides'] = {'hydraulic_design' : {'min_v' : 1.0}}
        _ = swmmanywhere.check_parameter_overrides(config)
            

def test_save_config():
    """Test the save_config function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        test_data_dir = Path(__file__).parent / 'test_data'
        defs_dir = Path(__file__).parent.parent / 'swmmanywhere' / 'defs'

        with (test_data_dir / 'demo_config.yml').open('r') as f:
            config = yaml.safe_load(f)
        
        # Correct and avoid filevalidation errors
        config['real'] = None
        
        # Fill with unused paths to avoid filevalidation errors
        config['base_dir'] = str(defs_dir / 'storm.dat')
        config['api_keys'] = str(defs_dir / 'storm.dat')

        swmmanywhere.save_config(config, temp_dir / 'test.yml')

        # Reload to check OK
        config = swmmanywhere.load_config(temp_dir / 'test.yml')

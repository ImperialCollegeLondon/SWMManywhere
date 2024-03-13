"""Tests for the main module."""
import tempfile
from pathlib import Path

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
    with tempfile.TemporaryDirectory() as base_dir:
        test_data_dir = Path(__file__).parent / 'test_data'
        defs_dir = Path(__file__).parent.parent / 'swmmanywhere' / 'defs'
        config = swmmanywhere.load_config(test_data_dir / 'demo_config.yml')
        base_dir = Path(base_dir)
        config['base_dir'] = str(base_dir)
        config['bbox'] = (0.05428,51.55847,0.07193,51.56726)
        config['address_overrides'] = {
            'building': test_data_dir / 'building.geojson',
            'precipitation': defs_dir / 'storm.dat'
            }
        model_dir = base_dir / 'demo' / 'bbox_1' / 'model_1'
        config['real']['subcatchments'] = model_dir / 'subcatchments.geoparquet'
        config['real']['inp'] = model_dir / 'model_1.inp'
        config['real']['graph'] = model_dir / 'graph.parquet'
        config['run_settings']['duration'] = 1000
        api_keys = {'nasadem_key' : 'b206e65629ac0e53d599e43438560d28'}
        with open(base_dir / 'api_keys.yml', 'w') as f:
            yaml.dump(api_keys, f)
        config['api_keys'] = base_dir / 'api_keys.yml'
        swmmanywhere.swmmanywhere(config)
        
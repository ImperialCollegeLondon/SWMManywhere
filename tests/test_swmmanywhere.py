"""Tests for the main module."""
from pathlib import Path

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
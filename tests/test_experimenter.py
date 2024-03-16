"""Tests for the main experimenter."""

import numpy as np

from swmmanywhere import parameters
from swmmanywhere.paper import experimenter

def assert_close(a: float, b: float, rtol: float = 1e-3) -> None:
    """Assert that two floats are close."""
    assert np.isclose(a, b, rtol=rtol).all()

def test_formulate_salib_problem():
    """Test the formulate_salib_problem function."""
    problem = experimenter.formulate_salib_problem([{'min_v' : [0.5,1.5]},
                                                    'max_v'])
    assert problem['num_vars'] == 2
    max_v = parameters.HydraulicDesign().model_json_schema()['properties']['max_v']
    assert problem['names'] == ['min_v','max_v']
    assert problem['bounds'] == [[0.5,1.5], [max_v['minimum'],max_v['maximum']]]

def test_generate_samples():
    """Test the generate_samples function."""
    samples = experimenter.generate_samples(N = 2,
                            parameters_to_select = ['min_v',
                                                    'max_v',
                                                    'chahinian_slope_scaling'],
                            seed = 1,
                            groups = False)
    assert len(samples) == 48
    assert set([x['param'] for x in samples]) == {'min_v','max_v','chahinian_slope_scaling'}
    assert_close(samples[0]['value'], 0.31093)

    samples = experimenter.generate_samples(N = 2,
                            parameters_to_select = ['min_v',
                                                    'max_v',
                                                    'chahinian_slope_scaling'],
                            seed = 1,
                            groups = True)
    assert len(samples) == 36
    
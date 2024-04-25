from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from swmmanywhere.paper import plotting


def test_create_behavioural_indices():
    """Test the create_behavioural_indices function."""
    # Create a DataFrame with some dummy data
    data = {
        'nse1': [-1, 0, 0.5, 1, -2],
        'nse2': [-1, 0, 0.5, 1, -2],
        'kge1': [-2, -0.4, 0, 0.5, 1],
        'kge2': [-0.41, -0.5, 0, 0.5, 1],
        'bias1': [-0.2, -0.09, 0, 0.1, 0.2],
        'bias2': [-0.2, -0.1, 0, 0.1, 0.2],
    }
    df = pd.DataFrame(data)

    # Call create_behavioral_indices with the dummy DataFrame
    strict_indices, less_strict_indices = plotting.create_behavioral_indices(df)

    # Check that the returned series have the expected values
    assert (strict_indices == [False, True, True, True, False]).all()
    assert (less_strict_indices == [False, True, True, False, False]).all()

def test_plot_objectives():
    """Test the plot_objectives function."""
    # Create a DataFrame with some dummy data
    data = {
        'param1': [1, 2, 3, 4, 5],
        'param2': [2, 3, 4, 5, 6],
        'obj1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'obj2': [0.2, 0.3, 0.4, 0.5, 0.6],
    }
    df = pd.DataFrame(data)

    # Define the parameters and objectives
    parameters = ['param1', 'param2']
    objectives = ['obj1', 'obj2']

    # Create dummy behavioral indices
    strict_indices = pd.Series([True, True, False, False, True])
    less_strict_indices = pd.Series([True, False, True, False, True])
    behavioral_indices = (strict_indices, less_strict_indices)

    with TemporaryDirectory() as temp_dir:
        # Call plot_objectives with the dummy DataFrame
        plotting.plot_objectives(df, 
                                 parameters, 
                                 objectives, 
                                 behavioral_indices, 
                                 Path(temp_dir))

        assert (Path(temp_dir) / 'param1.png').exists()
        assert (Path(temp_dir) / 'param2.png').exists()

def test_plot_sensitivity_indices():
    """Test the plot_sensitivity_indices function."""
    # Create a dictionary with some dummy data
    class Data:
        # Mock up SALib output
        params = ['param_1', 'param_2', 'param_3', 'param_4', 'param_5']
        total = pd.DataFrame({'ST': [0.1, 0.2, 0.3, 0.4, 0.5]},
                             index=params)
        first = pd.DataFrame({'S1': [0.05, 0.1, 0.15, 0.2, 0.25]},
                             index=params)
        second = pd.DataFrame({'S2': [0.05, 0.1, 0.15, 0.2, 0.25]},
                              index=params)
        
        def to_df(self):
            return (self.total, self.first, self.second)
    data = Data()
    r_ = {'obj1': data, 'obj2': data}

    # Define the objectives
    objectives = ['obj1','obj2']
    with TemporaryDirectory() as temp_dir:
        fid = Path(temp_dir) / 'indices.png'
        # Call plot_sensitivity_indices with the dummy dictionary
        plotting.plot_sensitivity_indices(r_, objectives, fid)

        assert fid.exists()

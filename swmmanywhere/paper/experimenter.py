# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
import os
import sys
from pathlib import Path

import pandas as pd
from SALib.sample import sobol

from swmmanywhere import swmmanywhere
from swmmanywhere.parameters import FilePaths, get_full_parameters_flat

# Set the number of threads to 1 to avoid conflicts with parallel processing
# for pysheds (at least I think that is what is happening)
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def formulate_salib_problem(parameters_to_select: list[str | dict] = []):
    """Formulate a SALib problem for a sensitivity analysis.

    Args:
        parameters_to_select (list, optional): List of parameters to include in 
            the analysis. Defaults to [].

    Returns:
        dict: A dictionary containing the problem formulation.
    """
    # Get all parameters schema
    parameters = get_full_parameters_flat()

    problem = {'names': [], 'bounds': [], 'groups': [], 'dists': [], 
               'num_vars' : len(parameters_to_select)}

    for parameter in parameters_to_select:
        if isinstance(parameter, dict):
            bounds = list(parameter.values())[0]
            parameter = list(parameter.keys())[0]
        else:
            bounds = [parameters[parameter]['minimum'],
                      parameters[parameter]['maximum']]
        
        problem['names'].append(parameter)
        problem['bounds'].append(bounds)
        problem['dists'].append(parameters[parameter].get('dist', 'unif'))
        problem['groups'].append(parameters[parameter]['category'])
    return problem

def generate_samples(N = None,
                     parameters_to_select = None,
                     seed = 1,
                     groups = False):
    """Generate samples for a sensitivity analysis.

    Args:
        N (int, optional): Number of samples to generate. Defaults to None.
        parameters_to_select (list, optional): List of parameters to include in 
            the analysis. Defaults to None.
        seed (int, optional): Random seed. Defaults to 1.
        groups (bool, optional): Whether to include the group names in the
            sampling (significantly changes how many samples are taken). 
            Defaults to False.

    Returns:
        list: A list of dictionaries containing the parameter values.
    """
    problem = formulate_salib_problem(parameters_to_select)
    
    if N is None:
        N = 2 ** (problem['num_vars'] - 1) 
    if not groups:
        problem_ = problem.copy()
        del problem_['groups']
    else:
        problem_ = problem.copy()
    param_values = sobol.sample(problem_, 
                                N, 
                                calc_second_order=True,
                                seed = seed)
    # attach names:
    X = []
    for ix, params in enumerate(param_values):
        for x,y,z in zip(problem['groups'],
                         problem['names'],
                         params):
            X.append({'param' : y,
                    'value' : z,
                    'iter' : ix,
                    'group' : x})
    return X

def process_parameters(jobid, nproc, config_base):
    """Generate and run parameter samples for the sensitivity analysis.

    This function generates parameter samples and runs the swmmanywhere model
    for each sample. It is designed to be run in parallel as a jobarray.

    Args:
        jobid (int): The job id.
        nproc (int): The number of processors to use.
        config_base (dict): The base configuration dictionary.

    Returns:
        dict: A list of dictionaries containing the results.
    """
    # Generate samples
    X = generate_samples(parameters_to_select=config_base['parameters_to_sample'],
                         N=2**config_base['sample_magnitude'])
    
    X = pd.DataFrame(X)
    gb = X.groupby('iter')
    
    flooding_results = {}
    if nproc is None:
        nproc = len(X)

    # Iterate over the samples, running the model when the jobid matches the
    # processor number
    for ix, params_ in gb:
        if ix % nproc != jobid:
            continue
        config = config_base.copy()

        # Update the parameters
        for _, row in params_.iterrows():
            config['parameter_overrides'][row['param']] = row['value']
        flooding_results[ix] = ix

        # Run the model
        addresses, metrics = swmmanywhere.swmmanywhere(config)

        # Save the results
        flooding_results[ix] = {'iter': ix, 
                                **metrics, 
                                **params_.set_index('param').value.to_dict()}
    return flooding_results, addresses

def save_results(jobid: int, results: list[dict], addresses: FilePaths) -> None:
    """Save the results of the sensitivity analysis.

    A results directory is created in the addresses.bbox directory, and the
    results are saved to a csv file there, labelled by jobid.

    Args:
        jobid (int): The job id.
        results (list[dict]): A list of dictionaries containing the results.
        addresses (FilePaths): The file paths.
    """
    results_fid = addresses.bbox / 'results'
    results_fid.mkdir(parents=True, exist_ok=True)
    fid_flooding = results_fid / f'{jobid}_metrics.csv'
    pd.DataFrame(results).T.to_csv(fid_flooding)

def parse_arguments():
    """Parse the command line arguments.

    Returns:
        tuple: A tuple containing the job id, number of processors, and the
            configuration file path.
    """
    if len(sys.argv) > 1:
        jobid = int(sys.argv[1])
        nproc = int(sys.argv[2])
        config_path = Path(sys.argv[3])
    else:
        jobid = 1
        nproc = None
        config_path = Path(__file__).parent.parent.parent / 'tests' /\
              'test_data' / 'demo_config_sa.yml'
    return jobid, nproc, config_path

if __name__ == '__main__':
    jobid, nproc, config_path = parse_arguments()
    config_base = swmmanywhere.load_config(config_path)
    if config_base['parameter_overrides'] is None:
        config_base['parameter_overrides'] = {}
    flooding_results, addresses = process_parameters(jobid, nproc, config_base)
    save_results(jobid, flooding_results, addresses)
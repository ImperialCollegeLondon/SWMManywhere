# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
import sys

from SALib.sample import sobol

from swmmanywhere.parameters import get_full_parameters


def formulate_salib_problem(parameters_to_select = None):
    """Formulate a SALib problem for a sensitivity analysis.

    Args:
        parameters_to_select (list, optional): List of parameters to include in 
            the analysis. Defaults to None.

    Returns:
        dict: A dictionary containing the problem formulation.
    """
    parameters = get_full_parameters()
    problem = {'names' : [],
                'bounds': [],
                'groups' : [],
                'dists' : []}
    for category, pars in parameters.items():
        for key, par in pars.schema()['properties'].items():
            keep = False
            if parameters_to_select is not None:
                if key in parameters_to_select:
                    keep = True
            else:
                keep = True
            if keep:
                if 'dist' in par.keys():
                    dist = par['dist']
                else:
                    dist = 'unif'
                problem['bounds'].append([par['minimum'], 
                                          par['maximum']])
                problem['names'].append(key)
                problem['dists'].append(dist)
                problem['groups'].append(category)
    problem['num_vars'] = len(problem['names'])
    return problem

def generate_samples(N = None,
                     parameters_to_select = None):
    """Generate samples for a sensitivity analysis.

    Args:
        N (int, optional): Number of samples to generate. Defaults to None.
        parameters_to_select (list, optional): List of parameters to include in 
            the analysis. Defaults to None.

    Returns:
        list: A list of dictionaries containing the parameter values.
    """
    problem = formulate_salib_problem(parameters_to_select)
    
    if N is None:
        N = 2 ** (problem['num_vars'] - 1) 
    
    param_values = sobol.sample(problem, 
                                N, 
                                calc_second_order=True)
    # attach names:
    X = []
    for ix, params in param_values:
        X.append({x : y for x,y in zip(problem['names'],
                                       params)})
    return X

            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        jobid = int(sys.argv[1])
        nproc = int(sys.argv[2])
    else:
        jobid = None
        nproc = None
    
    parameters_to_select = ['river_buffer_distance',
                            'outlet_length',
                            'surface_slope_scaling',
                            'elevation_scaling',
                            'length_scaling',
                            'chahinan_angle_scaling',
                            'contributing_area_scaling',
                            'surface_slope_exponent',
                            'elevation_exponent',
                            'length_exponent',
                            'chahinan_angle_exponent',
                            'contributing_area_exponent'
                            ]
    X = generate_samples(parameters_to_select = parameters_to_select)
    for ix, params in enumerate(X):
        if ix % nproc == jobid:
            pass
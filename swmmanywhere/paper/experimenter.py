# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyswmm
from SALib.sample import sobol

from swmmanywhere import preprocessing
from swmmanywhere.geospatial_utilities import graph_to_geojson
from swmmanywhere.graph_utilities import graphfcns as gu
from swmmanywhere.graph_utilities import load_graph
from swmmanywhere.parameters import get_full_parameters
from swmmanywhere.post_processing import synthetic_write


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
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        jobid = int(sys.argv[1])
        nproc = int(sys.argv[2])
        project = sys.argv[3]
    else:
        jobid = 1
        nproc = None
        project = 'demo'
    bbox = (0.04020,51.55759,0.09825591114207548,51.62050)
    parameters_to_select = ['min_v',
                            'max_v',
                            'max_fr',
                            'precipitation',
                            'outlet_length',
                            'chahinian_slope_scaling',
                            'length_scaling',
                            'contributing_area_scaling',
                            'chahinian_slope_exponent',
                            'length_exponent',
                            'contributing_area_exponent',
                            'lane_width',
                            'max_street_length'
                            ]
    X = generate_samples(parameters_to_select = parameters_to_select,
                         N = 2**10)
    X = pd.DataFrame(X)
    gb = X.groupby('iter')
    base_dir = Path(r'/rds/general/user/bdobson/ephemeral/swmmanywhere')
    # base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')
    # project = 'demo'

    function_list = ['assign_id',
                    'format_osmnx_lanes',
                    'double_directed',
                    'split_long_edges',
                    'calculate_contributing_area',
                    'set_elevation',
                    'set_surface_slope',
                    'set_chahinian_slope',
                    'set_chahinan_angle',
                    'calculate_weights',
                    'identify_outlets',
                    'derive_topology',
                    'pipe_by_pipe']
    flooding_results = {}
    if nproc is None:
        nproc = len(X)
    for ix, params_ in gb:
        if ix % nproc == jobid:
            flooding_results[ix] = ix
            addresses = preprocessing.create_project_structure(bbox, 
                                                               project, 
                                                               base_dir,
                                                               model_number = ix)
            params = get_full_parameters()
            params['topology_derivation'].weights = ['chahinian_slope',
                                                     'length',
                                                     'contributing_area']
            for key, row in params_.iterrows():
                setattr(params[row['group']], row['param'], row['value'])
            addresses.model.mkdir(parents = True, exist_ok = True)
            G = load_graph(addresses.bbox / 'base_graph.json')
            for fcn in function_list:
                print(f'starting {fcn} for job {ix} on {jobid}')
                G = getattr(gu, fcn)(G, 
                                        addresses = addresses, 
                                        **params)
            graph_to_geojson(G, 
                             addresses.model / f'graph_{ix}.geojson',
                             G.graph['crs'])
            
            nodes_gdf = gpd.read_file(addresses.model / f'graph_{ix}_nodes.geojson')
            nodes_gdf = nodes_gdf.rename(columns = {'elevation' : 'surface_elevation'})
            #TODO dropna because there are some duplicate edges...
            edges = gpd.read_file(addresses.model / 
                                  f'graph_{ix}_edges.geojson').dropna(subset=['diameter'])
            subs_gdf = gpd.read_file(addresses.bbox / 'subcatchments.parquet')
            #TODO river nodes have catchments associated too..
            subs_gdf = subs_gdf.loc[subs_gdf.id.isin(nodes_gdf.id)]
            synthetic_write(addresses,nodes_gdf,edges,subs_gdf)

            rain_fid = addresses.defs / 'storm.dat'
            
            shutil.copy(rain_fid, addresses.model / 'storm.dat')
            reporting_iters = 50
            with pyswmm.Simulation(str(addresses.model /
                                        f'{addresses.model.name}.inp')) as sim:
                sim.start()
                nodes = list(pyswmm.Nodes(sim))
                links = list(pyswmm.Links(sim))
                subs = list(pyswmm.Subcatchments(sim))
                results = []
                t_ = sim.current_time
                dt = 86400
                ind = 0
                while ((sim.current_time - t_).total_seconds() <= dt) & \
                    (sim.current_time < sim.end_time) &\
                        (not sim._terminate_request):
                    # Iterate the main model timestep
                    time = sim._model.swmm_step()
                    
                    # Break condition
                    if time < 0:
                        sim._terminate_request = True
                        break
                    # Store results in a list of dictionaries
                    if ind % reporting_iters == 0:
                        for link in links:
                            results.append({'date' : sim.current_time,
                                                'value' : link.flow,
                                                'variable' : 'flow',
                                                'object' : link._linkid})
                        for node in nodes:
                            results.append({'date' : sim.current_time,
                                                'value' : node.depth,
                                                'variable' : 'depth',
                                                'object' : node._nodeid})
                            
                            results.append({'date' : sim.current_time,
                                                'value' : node.flooding,
                                                'variable' : 'flood',
                                                'object' : node._nodeid})
                        for sub in subs:
                            results.append({'date' : sim.current_time,
                                                'value' : sub.runoff,
                                                'variable' : 'runoff',
                                                'object' : sub._subcatchmentid})
                    ind+=1
            results = pd.DataFrame(results)
            results.to_parquet(addresses.model / f'results_{ix}.gzip')
            area = subs_gdf.area.sum()
            flooding = results.loc[results.variable == 'flood']
            flooding['duration'] = (flooding.date - \
                                    flooding.date.min()).dt.total_seconds()
            
            def _f(x):
                return np.trapz(x.value,x.duration)

            total_flooding = flooding.groupby('object').apply(_f)
            
            total_flooding = total_flooding.sum()
            #Litres per m2
            total_flooding = total_flooding / subs_gdf.impervious_area.sum()
            
            #Simulated offline
            if project == 'demo':
                baseline_flooding = 31269000 / 2162462.1
            elif project == 'bellinge':
                baseline_flooding = 37324 / 843095
            
            maxflow = results.loc[results.variable == 'flow'].value.max()

            pbias = (total_flooding - baseline_flooding) / baseline_flooding
            flooding_results[ix] = {'pbias' : pbias,
                                    'maxflow' : maxflow,
                                    'iter' : ix,
                                    **params_.set_index('param').value.to_dict()}
    results_fid = addresses.bbox / 'results'
    results_fid.mkdir(parents = True, exist_ok = True)
    fid_flooding = results_fid / f'{jobid}_flooding.csv'
    pd.DataFrame(flooding_results).T.to_csv(fid_flooding)

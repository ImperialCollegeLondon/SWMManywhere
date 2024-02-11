# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created 2023-12-20.

@author: Barnaby Dobson
"""
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
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
    for ix, params in enumerate(param_values):
        for x,y,z in zip(problem['groups'],
                         problem['names'],
                         params):
            X.append({'group' : x,
                    'param' : y,
                    'value' : z,
                    'iter' : ix})
    return X

            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        jobid = int(sys.argv[1])
        nproc = int(sys.argv[2])
    else:
        jobid = 1
        nproc = None
    bbox = (0.04020,51.55759,0.09825591114207548,51.62050)
    parameters_to_select = ['river_buffer_distance',
                            'outlet_length',
                            'surface_slope_scaling',
                            'elevation_scaling',
                            'length_scaling',
                            'contributing_area_scaling',
                            'surface_slope_exponent',
                            'elevation_exponent',
                            'length_exponent',
                            'contributing_area_exponent'
                            ]
    X = generate_samples(parameters_to_select = parameters_to_select)
    X = pd.DataFrame(X)
    gb = X.groupby('iter')
    base_dir = Path(r'/rds/general/user/bdobson/ephemeral/swmmanywhere')
    # base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')
    project = 'demo'

    function_list = ['set_elevation',
                     'set_surface_slope',
                     'set_chahinan_angle',
                     'calculate_weights',
                     'identify_outlets',
                     'derive_topology',
                     'pipe_by_pipe']
    flooding_results = {}
    if nproc is None:
        nproc = len(X)
    for ix, params in enumerate(X):
        if ix % nproc == jobid:
            addresses = preprocessing.create_project_structure(bbox, 
                                                               project, 
                                                               base_dir)
            params = get_full_parameters()
            params['topology_derivation'].weights = ['surface_slope',
                                                     'length',
                                                     'contributing_area']
            for key, row in gb.get_group(ix).iterrows():
                setattr(params[row['group']], row['param'], row['value'])
            addresses.model.mkdir(parents = True, exist_ok = True)
            G = load_graph(addresses.bbox / 'graph_sequence2.json')
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
            subs_gdf['area'] *= 0.0001 # convert to ha
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
            flooding = sum(results.loc[results.variable == 'flood',
                                        'value'] > 0.0001) / nodes_gdf.shape[0]
            baseline_flooding = 0.48364788935311914 # timesteps / len_nodes
            pbias = (flooding - baseline_flooding) / baseline_flooding
            flooding_results[ix] = {'pbias' : pbias, 
                                    'iter' : ix,
                                    **gb.get_group(ix).set_index(['group','param']).value.to_dict()}
    results_fid = addresses.bbox / 'results'
    results_fid.mkdir(parents = True, exist_ok = True)
    fid_flooding = results_fid / f'{jobid}_flooding.json'
    with open(fid_flooding, 'w') as f:
        json.dump(flooding_results, f)

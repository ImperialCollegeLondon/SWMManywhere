# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyswmm

import swmmanywhere
from swmmanywhere import preprocessing
from swmmanywhere.geospatial_utilities import get_utm_epsg, graph_to_geojson
from swmmanywhere.graph_utilities import graphfcns as gu
from swmmanywhere.graph_utilities import save_graph
from swmmanywhere.post_processing import synthetic_write

if __name__ == '__main__':
    api_keys = {'cds_username' : '270593',
                'cds_api_key' : '1f0479b3-403f-4bd5-96b7-a2f494b149f0',
                'nasadem_key' : 'b206e65629ac0e53d599e43438560d28'}
    bbox = (0.04020,51.55759,0.09825591114207548,51.62050)
    project = 'demo'

    bbox = (10.23131,55.30225,  10.38378,55.38579)
    project = 'bellinge'

    base_dir = Path(r'/rds/general/user/bdobson/ephemeral/swmmanywhere')
    #base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')
    
    addresses = preprocessing.create_project_structure(bbox, 
                                                                    project, 
                                                                    base_dir)

    preprocessing.run_downloads(bbox, addresses, api_keys)

    G = preprocessing.create_starting_graph(addresses)
    save_graph(G, addresses.bbox / 'base_graph.json')


    crs = get_utm_epsg(bbox[0], bbox[1])
    graph_to_geojson(G, addresses.bbox / 'base_graph.geojson', crs)


    sequence1 = ["assign_id",
                "format_osmnx_lanes",
                "double_directed",
                "split_long_edges"]
    parameters = swmmanywhere.parameters.get_full_parameters()
    parameters['outlet_derivation'].outlet_length = 60
    parameters['subcatchment_derivation'].max_street_length = 200
    for fcn in sequence1:
        G = getattr(gu, fcn)(G, 
                            addresses = addresses, 
                            **parameters)
    save_graph(G, addresses.bbox / 'graph_sequence1.json')
    graph_to_geojson(G, addresses.bbox / 'graph_sequence1.geojson', crs)

    for fcn in ['calculate_contributing_area']:
        G = getattr(gu, fcn)(G, 
                            addresses = addresses, 
                            **parameters)
    save_graph(G, addresses.bbox / 'graph_sequence2.json')
    graph_to_geojson(G, addresses.bbox / 'graph_sequence2.geojson', crs)

    sequence3 = ["set_elevation",
                "set_surface_slope",
                "set_chahinian_slope",
                "set_chahinan_angle",
                "calculate_weights"]

    for fcn in sequence3:
        G = getattr(gu, fcn)(G, 
                            addresses = addresses, 
                            **parameters)
    save_graph(G, addresses.bbox / 'graph_sequence3.json')
    graph_to_geojson(G, addresses.bbox / 'graph_sequence3.geojson', crs)

    for fcn in ['identify_outlets']:
            G = getattr(gu, fcn)(G, 
                            addresses = addresses, 
                            **parameters)
    save_graph(G, addresses.bbox / 'graph_sequence4.json')
    graph_to_geojson(G, addresses.bbox / 'graph_sequence4.geojson', crs)

    for fcn in ['derive_topology']:
            G = getattr(gu, fcn)(G, 
                            addresses = addresses, 
                            **parameters)
    save_graph(G, addresses.bbox / 'graph_sequence5.json')
    graph_to_geojson(G, addresses.bbox / 'graph_sequence5.geojson', crs)

    for fcn in ['pipe_by_pipe']:
            G = getattr(gu, fcn)(G, 
                            addresses = addresses, 
                            **parameters)
    save_graph(G, addresses.bbox / 'graph_sequence6.json')
    graph_to_geojson(G, addresses.model / 'graph_sequence6.geojson', crs)

    nodes = gpd.read_file(addresses.model / 'graph_sequence6_nodes.geojson')
    nodes = nodes.rename(columns = {'elevation' : 'surface_elevation'})
    #TODO dropna because there are some duplicate edges...
    edges = gpd.read_file(addresses.model / 
                          'graph_sequence6_edges.geojson').dropna(subset=['diameter'])
    subs = gpd.read_file(addresses.model / 'subcatchments.parquet')
    #TODO river nodes have catchments associated too..
    subs = subs.loc[subs.id.isin(nodes.id)]

    area = subs.impervious_area.sum()
    synthetic_write(addresses,nodes,edges,subs)

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
            ind += 1
        results = pd.DataFrame(results)

        results = pd.DataFrame(results)
        results.to_parquet(addresses.model / 'results.gzip')
        
        flooding = results.loc[results.variable == 'flood']
        flooding['duration'] = (flooding.date - \
                                flooding.date.min()).dt.total_seconds()
        
        def _f(x):
            return np.trapz(x.value,x.duration)

        total_flooding = flooding.groupby('object').apply(_f)
        
        total_flooding = total_flooding.sum()
        #Litres per m2
        total_flooding = total_flooding / area
        print(total_flooding)

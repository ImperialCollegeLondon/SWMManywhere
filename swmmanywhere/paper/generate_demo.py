# mypy: ignore-errors
# -*- coding: utf-8 -*-
"""Created on 2024-01-26.

@author: Barney
"""
from pathlib import Path

import swmmanywhere
from swmmanywhere import preprocessing
from swmmanywhere.geospatial_utilities import get_utm_epsg, graph_to_geojson
from swmmanywhere.graph_utilities import graphfcns as gu
from swmmanywhere.graph_utilities import save_graph

if __name__ == '__main__':
    api_keys = {'cds_username' : '270593',
                'cds_api_key' : '1f0479b3-403f-4bd5-96b7-a2f494b149f0',
                'nasadem_key' : 'b206e65629ac0e53d599e43438560d28'}
    bbox = (0.04020,51.55759,0.09825591114207548,51.62050)

    project = 'demo'
    base_dir = Path(r'/rds/general/user/bdobson/ephemeral/swmmanywhere')

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
    graph_to_geojson(G, addresses.bbox / 'graph_sequence6.geojson', crs)
"""A script to subselect a SWMM model based on a query arc."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx

from swmmanywhere import parameters
from swmmanywhere.geospatial_utilities import graph_to_geojson
from swmmanywhere.graph_utilities import load_graph, save_graph
from swmmanywhere.post_processing import synthetic_write


def main():
    """Subselect a SWMM model based on a query arc."""
    project = 'bellinge'
    # Define addresses
    base_dir = Path(r'C:\Users\bdobson\Documents\data\swmmanywhere')
    graph = nx.MultiDiGraph(
        load_graph(base_dir / project / 'real' / 'graph.json'))
    subcatchments = gpd.read_file(
        base_dir / project / 'real' / 'subcatchments.geojson')

    # Tidy graph
    for u,v,d in graph.edges(data=True):
        if d['diameter'] <= 0:
            d['diameter'] = 3
        if d['length'] == 0:
            d['length'] = 1
        d['edge_type'] = 'street'

    nx.set_edge_attributes(graph, 0, 'contributing_area')
    nx.set_node_attributes(graph, 0, 'contributing_area')
    for idx, row in subcatchments.iterrows():
        for u,v,d in graph.edges(row['Outlet'],data=True):
            d['contributing_area'] = row['impervious_area']
        nx.set_node_attributes(graph, 
                               {row['Outlet']:row['impervious_area']}, 
                               'contributing_area')

    new_dir = base_dir / f'{project}_formatted' / 'real'
    addresses = parameters.FilePaths(base_dir = new_dir,
                        project_name = None,
                        bbox_number = None,
                        model_number = None,
                        extension = 'json')
    
    addresses.edges = new_dir / 'edges.geojson'
    addresses.nodes = new_dir / 'nodes.geojson'
    addresses.subcatchments = new_dir / 'subcatchments.geojson'
    addresses.inp = new_dir / 'model.inp'
    addresses.precipitation = Path(
        r'C:\Users\bdobson\Documents\GitHub\SWMManywhere\swmmanywhere\defs\storm.dat')
    iw_models = Path(r'C:\Users\bdobson\Documents\data\infoworks_models')
    if project == 'cranbrook':
        # For some reason the graph nodes don't have all the necessary info
        nodes = gpd.read_file(iw_models / "cranbrook" / "Nodes.shp")
        nodes = nodes.rename(columns = {'chamber_fl':'chamber_floor_elevation' , 
                                        'ground_lev':'surface_elevation'})
        nx.set_node_attributes(graph, 
                            nodes.set_index('node_id').chamber_floor_elevation.to_dict(), 
                            'chamber_floor_elevation')
        nx.set_node_attributes(graph, 
                            nodes.set_index('node_id').surface_elevation.to_dict(), 
                            'surface_elevation')
    elif project == 'bellinge':
        nodes = gpd.read_file(
            iw_models / "bellinge" / "swmmio_conversion" / "nodes.geojson"
            )
        nodes = nodes.rename(columns = {'InvertElev' : 'chamber_floor_elevation',
                                        'Name' : 'node_id'})
        nodes['surface_elevation'] = nodes['chamber_floor_elevation'] +\
              nodes['MaxDepth']
        nx.set_node_attributes(graph, 
                            nodes.set_index('node_id').chamber_floor_elevation.to_dict(), 
                            'chamber_floor_elevation')
        nx.set_node_attributes(graph, 
                            nodes.set_index('node_id').surface_elevation.to_dict(), 
                            'surface_elevation')
    
    # And subcatchments aren't formatted
    subcatchments = subcatchments.rename(columns = {'id':'misc_id',
                                                    'Outlet':'id',
                                                    'PercImperv':'rc',
                                                    'Width':'width',
                                                    'PercSlope':'slope'})
    subcatchments = subcatchments.drop_duplicates('id')
    subcatchments['area'] = subcatchments.geometry.area

    # Define where to cut the model
    if project == 'cranbrook':
        query_arc = 'node_1439.1'
    elif project == 'bellinge':
        query_arc = 'F74F370_F74F360_l1'

    # Get the nodes
    (us_node, ds_node) = [(u,v) for u,v,d in graph.edges(data=True) 
                          if d['id'] == query_arc][0]

    anc = nx.ancestors(graph,us_node)
    anc = anc.union([us_node, ds_node])

    # Remove elements not in anc
    subcatchments = subcatchments[subcatchments['id'].isin(anc)]
    new_graph = graph.subgraph(anc).copy()

    # Convert attribute names
    for u,v,d in new_graph.edges(data=True):
        d['u'] = d.pop('InletNode')
        d['v'] = d.pop('OutletNode')

    # Write the new model
    new_dir.mkdir(exist_ok=True, parents = True)
    subcatchments.to_file(new_dir / 'subcatchments.geojson')
    graph_to_geojson(new_graph,
                    addresses.nodes,
                    addresses.edges,
                    new_graph.graph['crs'])
    synthetic_write(addresses)
    save_graph(new_graph, new_dir / 'graph.json')

    # Provide info
    print(f'new bbox: {subcatchments.to_crs(4326).total_bounds}')
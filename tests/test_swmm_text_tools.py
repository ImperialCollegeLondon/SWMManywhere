import difflib
import filecmp
import os
import shutil
import tempfile

import geopandas as gpd
import pandas as pd
import pyswmm
from shapely import geometry as sgeom

from swmmanywhere import swmm_text_tools as stt


def test_overwrite_section():
    """Test the overwrite_section function.
    
    All this tests is that the data is written to the file.
    """
    # Copy the example file to a temporary file
    fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                        'swmmanywhere',
                        'defs',
                        'basic_drainage_all_bits.inp')
    temp_fid = 'temp.inp'
    shutil.copy(fid, temp_fid)
    try:
        data = [["1","1","1","1.166","100","500","0.5","0","empty"],
                ["2","1","1","1.1","100","500","0.5","0","empty"],
                ["subca_3","1","1","2","100","400","0.5","0","empty"]]
        
        section = '[SUBCATCHMENTS]'
        stt.overwrite_section(data, section, temp_fid)
        with open(temp_fid, 'r') as file:
            content = file.read()
        assert 'subca_3' in content
    finally:
        os.remove(temp_fid)

def test_change_flow_routing():
    """Test the change_flow_routing function."""
    # Copy the example file to a temporary file
    fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                        'swmmanywhere',
                        'defs',
                        'basic_drainage_all_bits.inp')
    temp_fid = 'temp.inp'
    shutil.copy(fid, temp_fid)
    try:
        new_routing = 'FAKE_ROUTING'
        stt.change_flow_routing(new_routing, temp_fid)
        with open(temp_fid, 'r') as file:
            content = file.read()
        assert 'FAKE_ROUTING' in content
    finally:
        os.remove(temp_fid)

def test_data_input_dict_to_inp():
    """Test the data_input_dict_to_inp function.
    
    All this tests is that the data is written to a new file.
    """
    data_dict = {'SUBCATCHMENTS': 
                 [["1","1","1","1.166","100","500","0.5","0","empty"],
                  ["2","1","1","1.1","100","500","0.5","0","empty"],
                  ["subca_3","1","1","2","100","400","0.5","0","empty"]]
                  }
    
    fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                        'swmmanywhere',
                        'defs',
                        'basic_drainage_all_bits.inp')
    temp_fid = 'temp.inp'
    shutil.copy(fid, temp_fid)
    try:
        stt.data_dict_to_inp(data_dict,
                             fid,
                             temp_fid)
        with open(temp_fid, 'r') as file:
            content = file.read()
        assert 'subca_3' in content
    finally:
        os.remove(temp_fid)

def test_explode_polygon():
    """Test the explode_polygon function."""
    df = pd.Series({'subcatchment' : '1',
                    'geometry' : sgeom.Polygon([(0,0), (1,0), 
                                                (1,1), (0,1)])})
    result = stt.explode_polygon(df)
    assert result.shape[0] == 5
    assert result.loc[3,'y'] == 1

def generate_data_dict():
    """Generate a data dict for testing."""
    nodes = gpd.GeoDataFrame({'id' : ['node1', 'node2'],
                                'x' : [0, 1],
                                'y' : [0, 1],
                                'max_depth' : [1, 1],
                                'chamber_floor_elevation' : [1, 1],
                                'surface_elevation' : [2, 2],
                                'manhole_area' : [0.5,0.5]
                                })
    outfalls = gpd.GeoDataFrame({'id' : ['node2_outfall'],
                                    'chamber_floor_elevation' : [0],
                                    'surface_elevation' : [0],
                                    'x' : [0],
                                    'y' : [0]})
    conduits = gpd.GeoDataFrame({'id' : ['node1-node2','node2-node2_outfall'],
                                    'u' : ['node1','node2'],
                                    'v' : ['node2','node2_outfall'],
                                    'length' : [1,1],
                                    'roughness' : [0.01,0.01],
                                    'shape_swmm' : ['CIRCULAR','CIRCULAR'],
                                    'diameter' : [1,15],
                                    'capacity' : [1E10,1E10]
                                    })
    subs = gpd.GeoDataFrame({'subcatchment' : ['node1-sub'],
                                    'rain_gage' : ['1'],
                                    'id' : ['node1'],
                                    'area' : [1],
                                    'rc' : [1],
                                    'width' : [1],
                                    'slope' : [0.001],
                                    'geometry' : [sgeom.Polygon([(0,0), (1,0), 
                                                                (1,1), (0,1)])]})
    rain_fid = 'storm.dat'
    event = {'name' : '1',
                    'unit' : 'mm',
                    'interval' : '01:00',
                    'fid' : rain_fid}
    symbol = {'x' : 0,
                'y' : 0,
                'name' : '1'}
    
    return {'nodes' : nodes,
            'outfalls' : outfalls,
            'conduits' : conduits,
            'subs' : subs,
            'event' : event,
            'symbol' : symbol}

def test_synthetic_write():
    """Test the synthetic_write function."""
    data_dict = generate_data_dict()    
    model_dir = tempfile.mkdtemp('_1')
    try:
        # Write the model with synthetic_write
        nodes = gpd.GeoDataFrame(data_dict['nodes'])
        nodes.geometry = gpd.points_from_xy(nodes.x, nodes.y)
        nodes.to_file(
            os.path.join(
                model_dir, 
                'pipe_by_pipe_nodes.geojson'))
        nodes = nodes.set_index('id')
        edges = gpd.GeoDataFrame(pd.DataFrame(data_dict['conduits']).iloc[[0]])
        edges.geometry = [sgeom.LineString([nodes.loc[u,'geometry'],
                                            nodes.loc[v,'geometry']]) 
                          for u,v in zip(edges.u, edges.v)]
        edges.to_file(
            os.path.join(
                model_dir,
                'pipe_by_pipe_edges.geojson'))
        subs = data_dict['subs'].copy()
        subs['subcatchment'] = ['node1']
        subs.to_file(os.path.join(model_dir, 
                                                'subcatchments.geojson'))
        stt.synthetic_write(model_dir)

        # Write the model with data_dict_to_inp
        comparison_file = os.path.join(model_dir, "model_base.inp")
        template_fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                        'swmmanywhere',
                        'defs',
                        'basic_drainage_all_bits.inp')
        stt.data_dict_to_inp(stt.format_to_swmm_dict(**data_dict),
                             template_fid,
                             comparison_file)
        
        # Compare
        new_input_file = os.path.join(model_dir, "model_1.inp")
        are_files_identical = filecmp.cmp(new_input_file,
                                           comparison_file, 
                                           shallow=False)
        if not are_files_identical:
            with open(new_input_file, 
                      'r') as file1, open(comparison_file, 
                                          'r') as file2:
                diff = difflib.unified_diff(
                    file1.readlines(),
                    file2.readlines(),
                    fromfile=new_input_file,
                    tofile=comparison_file,
                )
            print(''.join(diff))
        assert are_files_identical, "The files are not identical"
    finally:
        pass
        # shutil.rmtree(model_dir)

def test_format_to_swmm_dict():
    """Test the format_format_to_swmm_dict function.
    
    Writes a formatted dict to a model and checks that it runs without 
    error.
    """
    data_dict = generate_data_dict()
    data_dict = stt.format_to_swmm_dict(**data_dict)
    
    
    fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                        'swmmanywhere',
                        'defs',
                        'basic_drainage_all_bits.inp')
    rain_fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                        'swmmanywhere',
                        'defs',
                        'storm.dat')
    temp_dir = tempfile.mkdtemp()
    shutil.copy(rain_fid, os.path.join(temp_dir,'storm.dat'))
    temp_fid = os.path.join(temp_dir,'temp.inp')
    try:
        stt.data_dict_to_inp(data_dict,
                             fid,
                             temp_fid)
        with pyswmm.Simulation(temp_fid) as sim:
            for ind, step in enumerate(sim):
                pass
    finally:
        shutil.rmtree(temp_dir)
        
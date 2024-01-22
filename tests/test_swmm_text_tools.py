import os
import shutil

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

def test_format_format_to_swmm_dict():
    """Test the format_format_to_swmm_dict function.
    
    Writes a formatted dict to a model and checks that it runs without 
    error.
    """
    nodes = gpd.GeoDataFrame({'id' : ['node1', 'node2'],
                                'x' : [0, 1],
                                'y' : [0, 1],
                                'max_depth' : [1, 1],
                                'chamber_floor_elevation' : [1, 1],
                                'manhole_area' : [1,1]
                                })
    outfalls = gpd.GeoDataFrame({'id' : ['outfall3'],
                                    'chamber_floor_elevation' : [1],
                                    'x' : [0],
                                    'y' : [0]})
    conduits = gpd.GeoDataFrame({'id' : ['link1','link2'],
                                    'u' : ['node1','node2'],
                                    'v' : ['node2','outfall3'],
                                    'length' : [1,1],
                                    'roughness' : [1,1],
                                    'shape_swmm' : ['CIRCULAR','CIRCULAR'],
                                    'diameter' : [1,1],
                                    'capacity' : [0.1,0.1]
                                    })
    subs = gpd.GeoDataFrame({'subcatchment' : ['sub1'],
                                    'rain_gage' : ['1'],
                                    'id' : ['node1'],
                                    'area' : [1],
                                    'rc' : [1],
                                    'width' : [1],
                                    'slope' : [0.001],
                                    'geometry' : [sgeom.Polygon([(0,0), (1,0), 
                                                                (1,1), (0,1)])]})
    
    rain_fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                        'swmmanywhere',
                        'defs',
                        'storm.dat')
    event = {'name' : '1',
                    'unit' : 'mm',
                    'interval' : 1,
                    'fid' : rain_fid}
    symbol = {'x' : 0,
                'y' : 0,
                'name' : 'name'}
    data_dict = stt.format_to_swmm_dict(nodes,
                                        outfalls,
                                        conduits,
                                        subs,
                                        event,
                                        symbol)
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
        with pyswmm.Simulation(temp_fid) as sim:
            for ind, step in enumerate(sim):
                pass
    finally:
        os.remove(temp_fid.replace('.inp','.rpt'))
        os.remove(temp_fid)
        
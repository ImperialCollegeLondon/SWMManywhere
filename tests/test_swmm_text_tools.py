import os
import shutil

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
import os
import shutil

from swmmanywhere import swmm_text_tools as stt


def test_overwrite_section():
    """Test the overwrite_section function."""
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
        stt.overwrite_section(data, temp_fid, section)
        with open(temp_fid, 'r') as file:
            content = file.read()
        assert 'subca_3' in content
    finally:
        os.remove(temp_fid)
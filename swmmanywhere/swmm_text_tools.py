# -*- coding: utf-8 -*-
"""Created 2024-01-22.

@author: Barnaby Dobson
"""
import re

import numpy as np


def overwrite_section(data: np.ndarray,
                      section: str,
                      fid: str):
    """Overwrite a section of a SWMM .inp file with new data.

    Args:
        data (np.ndarray): Data array to be written to the SWMM .inp file.
        section (str): Section of the SWMM .inp file to be overwritten.
        fid (str): File path to the SWMM .inp file.
        
    Example:
        data = np.array([
                ['1', '1', '1', '1.166', '100', '500', '0.5', '0', 'empty'],
                ['2', '1', '1', '1.1', '100', '500', '0.5', '0', 'empty'],
                ['3', '1', '1', '2', '100', '400', '0.5', '0', 'empty']])
        fid = 'my_pre_existing_swmm_input_file.inp'
        section = '[SUBCATCHMENTS]'
        overwrite_section(data, section, fid)
    """
    # Read the existing SWMM .inp file
    with open(fid, 'r') as infile:
        lines = infile.readlines()
    
    # Create a flag to indicate whether we are within the target section
    within_target_section = False
    
    # Iterate through the lines and make modifications as needed
    with open(fid, 'w') as outfile:
        for ix, line in enumerate(lines):
        
            if line.strip() == section:
                within_target_section = True
                outfile.write(line)  # Write the start section header
                
                # Write headers
                i = 1
                while lines[ix + i][0] == ';':
                    outfile.write(lines[ix + i]) # Write column headers
                    i+=1
                    
                example_line = lines[ix + i]
                print('example_line {1}: {0}'
                      .format(example_line.replace('\n',''), section))
                print('note - this line must have at least as many column')
                print('entries as all other rows in this section\n')
                pattern = r'(\s+)'
    
                # Find all matches of the pattern in the input line
                matches = re.findall(pattern, example_line)
                
                # Calculate the space counts by taking the length of each match 
                space_counts = [len(x) + len(y) for x,y in
                                 zip(matches, example_line.split())]
                if len(space_counts) == 0:
                    if data.shape[0] == 0:
                        pass
                    else:
                        print('no template for data?')
                else:
                    space_counts[-1] -= 1
                    new_text = ''
                    if data is not None:
                        for i, row in enumerate(data):
                            if section != '[CONTROLS]':
                                formatted_row = [
                                    '{0:<{1}}'.format(x, 
                                                      max(y, len(str(x)) + 1)) 
                                    for x, y in zip(row, space_counts)
                                ]
                                new_line = '{0}\n'\
                                    .format(''.join(formatted_row))
                                new_text += new_line
                            else:
                                new_text = row
        
                    outfile.write(new_text)  # Write the new content
                    outfile.write('\n')
            elif re.search(r'\[.*?\]', line):
                within_target_section = False
                outfile.write(line)  # Write the end section header
            elif not within_target_section:
                outfile.write(line)  # Write lines outside the target section

def change_flow_routing(new_routing, inp_file):
    """Change the flow routing method in a SWMM .inp file.

    Args:
        new_routing (str): New flow routing method (KINWAVE, DYNWAVE, STEADY).
        inp_file (str): File path to the SWMM .inp file.
    """
    # Read the input file
    with open(inp_file, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the FLOW_ROUTING line
    for i, line in enumerate(lines):
        if line.strip().startswith('FLOW_ROUTING'):
            lines[i] = f'FLOW_ROUTING {new_routing}\n'
            break
    
    # Write the modified content back to the input file
    with open(inp_file, 'w') as f:
        f.writelines(lines)

def data_dict_to_inp(data_dict: dict[str, np.ndarray],
                     base_input_file: str, 
                     new_input_file: str, 
                     routing: str = "DYNWAVE"):
    """Write a SWMM .inp file from a dictionary of data arrays.

    Args:
        data_dict (dict[str, np.ndarray]): Dictionary of data arrays. Where
            each key is a SWMM section and each value is a numpy array of
            data to be written to that section. The existing section is 
            overwritten
        base_input_file (str): File path to the example/template .inp file.
        new_input_file (str): File path to the new SWMM .inp file.
        routing (str, optional): Flow routing method (KINWAVE, DYNWAVE,
            STEADY). Defaults to "DYNWAVE".
    """
    # Read the content from the existing input file
    with open(base_input_file, 'r') as existing_file:
        existing_content = existing_file.read()

    # Open the new input file for writing
    with open(new_input_file, 'w') as new_file:
        # Write the content from the existing input file to the new file
        new_file.write(existing_content)

    # Write the inp file
    for key, data in data_dict.items():
        print(key)
        start_section = '[{0}]'.format(key)
     
        overwrite_section(data, new_input_file, start_section)

    # Set the flow routing
    change_flow_routing(routing, new_input_file)
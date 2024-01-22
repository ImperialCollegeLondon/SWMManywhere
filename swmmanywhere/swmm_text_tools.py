# -*- coding: utf-8 -*-
"""Created 2024-01-22.

@author: Barnaby Dobson
"""
import re

import numpy as np


def overwrite_section(data: np.ndarray,
                      fid: str, 
                      section: str):
    """Overwrite a section of a SWMM .inp file with new data.

    Args:
        data (np.ndarray): Data array to be written to the SWMM .inp file.
        fid (str): File path to the SWMM .inp file.
        section (str): Section of the SWMM .inp file to be overwritten.
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
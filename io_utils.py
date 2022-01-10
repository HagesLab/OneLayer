# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 17:39:26 2022

@author: cfai2
"""
import numpy as np
import tables
def extract_values(string, delimiter):
    """Converts a string with deliimiters into a list of float values"""
	# E.g. "100,200,300" with "," delimiter becomes [100,200,300]
    values = string.split(delimiter)
    values = np.array(values, dtype=float)
    return values

def get_split_and_clean_line(line: str):
    """Split line by colon symbol ':' and
    remove preceding and trailing spaces."""
    split_line = line.split(':')
    split_line = [i.strip() for i in split_line]
    return split_line

def check_valid_filename(file_name):
    """Screens file_name for prohibited characters"""
    prohibited_characters = [".","<",">","/","\\","*","?",":","\"","|"]
	# return !any(char in file_name for char in prohibited_characters)
    if any(char in file_name for char in prohibited_characters):
        return False

    return True
        
def u_read(filename, t0=None, t1=None, l=None, r=None, single_tstep=False, 
           need_extra_node=False, force_1D=True):
    """Read a subset of a 2D array (from time t0 to t1 and position l to r) stored in an .h5 file"""
    if (t0 is not None) and single_tstep:
        t1 = t0 + 1
        
    if need_extra_node:
        r = r + 1
        
    with tables.open_file(filename, mode='r') as ifstream:
        data = np.array(ifstream.root.data[t0:t1, l:r])
        
        if force_1D and np.ndim(data) == 2 and len(data) == 1:
            data = data.flatten()
        return data
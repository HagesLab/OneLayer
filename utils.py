# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:53:38 2021

@author: cfai2
"""
import numpy as np
import tables
def extract_values(string, delimiter):
    # Converts a string with deliimiters into a list of float values
	# E.g. "100,200,300" with "," delimiter becomes [100,200,300]
    values = []
    substring = string
    
    while (not substring.find(delimiter) == -1):
        next_delimiter = substring.find(delimiter)
        values.append(float(substring[0:next_delimiter]))
        substring = substring[next_delimiter + 1:]

    values.append(float(substring))

    return values

def check_valid_filename(file_name):
    prohibited_characters = [".","<",">","/","\\","*","?",":","\"","|"]
	# return !any(char in file_name for char in prohibited_characters)
    if any(char in file_name for char in prohibited_characters):
        return False

    return True
        
def u_read(filename, t0=None, t1=None, l=None, r=None, single_tstep=False, need_extra_node=False):
    if not (t0 is None) and single_tstep:
        t1 = t0 + 1
        
    if need_extra_node:
        r = r + 1
        
    with tables.open_file(filename, mode='r') as ifstream:
        data = np.array(ifstream.root.data[t0:t1, l:r])
        
        if np.ndim(data) == 2 and len(data) == 1:
            data = data.flatten()
        return data
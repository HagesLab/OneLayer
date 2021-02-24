# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:53:38 2021

@author: cfai2
"""
import numpy as np
import tables

def to_index(x,dx, absUpperBound, is_edge=False):
    # Warning: this and toCoord() always round x down to the nearest node (or edge if is_edge=True)!
    absLowerBound = dx / 2 if not is_edge else 0
    if (x < absLowerBound):
        return 0

    if (x > absUpperBound):
        return int(absUpperBound / dx)

    return int((x - absLowerBound) / dx)

def to_pos(i,dx, is_edge=False):
    absLowerBound = dx / 2 if not is_edge else 0
    return (absLowerBound + i * dx)

def to_array(value, m, is_edge):
    if not isinstance(value, np.ndarray):
        if is_edge:
            return np.ones(m+1) * value
        else:
            return np.ones(m) * value
        
    else:
        return value
    
def get_all_combinations(value_dict):
    combinations = []
    param_names = list(value_dict.keys())
        
    iterable_param_indexes = {}
    iterable_param_lengths = {}
    for param in param_names:
        iterable_param_indexes[param] = 0
        iterable_param_lengths[param] = value_dict[param].__len__()
    
    pivot_index = param_names.__len__() - 1

    current_params = dict(value_dict)
    # Create a list of all combinations of parameter values
    while(pivot_index >= 0):

        # Generate the next parameter set using lists of indices stored in the helper structures
        for iterable_param in param_names:
            current_params[iterable_param] = value_dict[iterable_param][iterable_param_indexes[iterable_param]]

        combinations.append(dict(current_params))

        # Determine the next iterable parameter using a "reverse search" amd update indices from right to left
        # For example, given Param_A = [1,2,3], Param_B = [4,5,6], Param_C = [7,8]:
        # The order {A, B, C} this algorithm will run is: 
        # {1,4,7}, 
        # {1,4,8}, 
        # {1,5,7}, 
        # {1,5,8}, 
        # {1,6,7}, 
        # {1,6,8},
        # ...
        # {3,6,7},
        # {3,6,8}
        pivot_index = param_names.__len__() - 1
        while (pivot_index >= 0 and iterable_param_indexes[param_names[pivot_index]] == iterable_param_lengths[param_names[pivot_index]] - 1):
            pivot_index -= 1

        iterable_param_indexes[param_names[pivot_index]] += 1

        for i in range(pivot_index + 1, param_names.__len__()):
            iterable_param_indexes[param_names[i]] = 0
            
    return combinations
    
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
    
def autoscale(val_array=None, min_val=None, max_val=None):
    # Help a matplotlib plot determine whether a log or linear scale should be used
    # when plotting val_array
    if max_val is not None and min_val is not None:
        pass
    elif val_array is not None:
        max_val = np.amax(val_array)
        min_val = np.amin(val_array)
        
    else:
        return 'linear'
    
    if not (min_val == 0) and np.abs(max_val / min_val) > 1e1:
        return 'symlog'
    elif (min_val < 0) and np.abs(min_val / max_val) > 1e1:
        return 'symlog'
    else:
        return 'linear'
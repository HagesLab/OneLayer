# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 17:39:26 2022

@author: cfai2
"""
import numpy as np
import tables
import datetime
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

def export_ICfile(newFileName, nb, flags, layers, allow_write_LGC=False):
    with open(newFileName, "w+") as ofstream:
        
        # We don't really need to note down the time of creation, but it could be useful for interaction with other programs.
        ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
        ofstream.write("System_class: {}\n".format(nb.module.system_ID))
        ofstream.write("f$ System Flags:\n")
        
        for flag in nb.module.flags_dict:
            ofstream.write("{}: {}\n".format(flag, flags[flag]))
              
        for layer_name, layer in layers.items():
            ofstream.write("L$: {}\n".format(layer_name))
            ofstream.write("p$ Space Grid:\n")
            if isinstance(layer, dict):
                ofstream.write("Total_length: {}\n".format(layer["Total_length"]))
                ofstream.write("Node_width: {}\n".format(layer["Node_width"]))
            else:
                ofstream.write("Total_length: {}\n".format(layer.total_length))
                ofstream.write("Node_width: {}\n".format(layer.dx))
        
            ofstream.write("p$ System Parameters:\n")
        
            # Saves occur as-is: any missing parameters are saved with whatever default value module gives them
            for param in layer.params:
                if param == "Total_length" or param == "Node_width": continue
                param_values = layer.params[param].value
                if isinstance(param_values, np.ndarray):
                    # Write the array in a more convenient format
                    ofstream.write("{}: {:.8e}".format(param, param_values[0]))
                    for value in param_values[1:]:
                        ofstream.write("\t{:.8e}".format(value))
                        
                    ofstream.write('\n')
                else:
                    # The param value is just a single constant
                    ofstream.write("{}: {}\n".format(param, param_values))
              
            if allow_write_LGC:
                if nb.module.system_ID in nb.LGC_eligible_modules and nb.using_LGC[layer_name]:
                    ofstream.write("p$ Laser Parameters\n")
                    for laser_param in nb.LGC_values[layer_name]:
                        ofstream.write("{}: {}\n".format(laser_param,
                                                         nb.LGC_values[layer_name][laser_param]))
    
                    ofstream.write("p$ Laser Options\n")
                    for laser_option in nb.LGC_options[layer_name]:
                        ofstream.write("{}: {}\n".format(laser_option, 
                                                         nb.LGC_options[layer_name][laser_option]))
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:03:19 2021

@author: cfai2
"""
import tkinter as tk
import numpy as np

class Param_Rule:
    """The Parameter Toolkit uses these to build Parameter()'s values"""
    def __init__(self, variable, type, l_bound, r_bound=-1, l_boundval=-1, r_boundval=-1):
        self.variable = variable # e.g. N, P, E-Field
        self.type = type
        self.l_bound = l_bound
        self.r_bound = r_bound
        self.l_boundval = l_boundval
        self.r_boundval = r_boundval
        return

    def get(self):
        """Pack values into a display string for GUI"""
        if (self.type == "POINT"):
            return((self.variable + ": " + self.type + " at x=" 
                    + '{:.4e}'.format(self.l_bound) + " with value: " 
                    + '{:.4e}'.format(self.l_boundval)))

        elif (self.type == "FILL"):
            return((self.variable + ": " + self.type + " from x=" 
                    + '{:.4e}'.format(self.l_bound) + " to " 
                    + '{:.4e}'.format(self.r_bound) + " with value: " 
                    + '{:.4e}'.format(self.l_boundval)))

        elif (self.type == "LINE"):
            return((self.variable + ": " + self.type + " from x=" 
                    + '{:.4e}'.format(self.l_bound) + " to " 
                    + '{:.4e}'.format(self.r_bound) + 
                    " with left value: " + '{:.4e}'.format(self.l_boundval) 
                    + " and right value: " + '{:.4e}'.format(self.r_boundval)))

        elif (self.type == "EXP"):
            return((self.variable + ": " + self.type + " from x=" 
                    + '{:.4e}'.format(self.l_bound) + " to " 
                    + '{:.4e}'.format(self.r_bound) + 
                    " with left value: " + '{:.4e}'.format(self.l_boundval) 
                    + " and right value: " + '{:.4e}'.format(self.r_boundval)))

        else:
            return("Error #101: Invalid initial condition")
        
class Flag:
    """This class exists to solve a little problem involving tkinter checkbuttons: we get the value of a checkbutton using its tk.IntVar() 
       but we interact with the checkbutton using the actual tk.CheckButton() element
       So wrap both of those together in a single object and call it a day
    """
    def __init__(self, master, display_name):
        self.tk_var = tk.IntVar()
        self.tk_element = tk.ttk.Checkbutton(master=master, text=display_name, 
                                             variable=self.tk_var, 
                                             onvalue=1, offvalue=0)
        return
    
    def value(self):
        return self.tk_var.get()

class Batchable:
    """Much like the flag class, the Batchable() serves to collect together various tk elements and values for the batch IC tool."""
    def __init__(self, tk_optionmenu, tk_entrybox, param_name):
        self.tk_optionmenu = tk_optionmenu
        self.tk_entrybox = tk_entrybox
        self.param_name = param_name
        return
    
class Data_Set:
    def __init__(self, data, grid_x, params_dict, type, filename):
        self.data = data
        self.grid_x = grid_x
        self.params_dict = dict(params_dict)
        self.type = type
        self.filename = filename
        return
    
    def tag(self, for_matplotlib=False):
        """Return an identifier for a dataset using its originating filename and data type"""
        # For some reason, Matplotlib legends don't like leading underscores
        if not for_matplotlib:
            return self.filename + "_" + self.type
        else:
            return (self.filename + "_" + self.type).strip('_')
        
class Raw_Data_Set(Data_Set):
    """Object containing all the metadata required to plot and integrate saved data sets"""
    def __init__(self, data, grid_x, node_x, total_time, dt, params_dict, type, filename, show_index):
        super().__init__(data, grid_x, params_dict, type, filename)
        self.node_x = node_x        # Array of x-coordinates corresponding to system nodes - needed to generate initial condition from data

        # node_x and grid_x will usually be identical, unless the data is a type (like E-field) that exists on edges
        # There's a little optimization that can be made here because grid_x will either be identical to node_x or not, but that makes the code harder to follow

        self.show_index = show_index # Time step number data belongs to
        self.total_time = total_time
        self.dt = dt
        self.num_tsteps = int(0.5 + total_time / dt)
        return

    def build(self):
        """Concatenate (x,y) pairs for export"""
        return np.vstack((self.grid_x, self.data))
    
class Integrated_Data_Set(Data_Set):
    def __init__(self, data, grid_x, total_time, dt, params_dict, type, filename):
        super().__init__(data, grid_x, params_dict, type, filename)
        self.total_time = total_time
        self.dt = dt
        return
    

class Data_Group:
    def __init__(self):
        self.type = "None"
        self.datasets = {}
        self.flags = None
        self.dt = -1
        self.total_t = -1
        return
    
    def get_maxval(self):
        return np.amax([np.amax(self.datasets[tag].data) for tag in self.datasets])
    
    def get_minval(self):
        return np.amin([np.amin(self.datasets[tag].data) for tag in self.datasets])
    
    def size(self):
        return len(self.datasets)
    
    def clear(self):
        self.datasets.clear()
        return

class Raw_Data_Group(Data_Group):
    def __init__(self):
        super().__init__()
        return

    def add(self, data, tag):
        
        if not self.datasets: 
            # Allow the first set in to set the dt and t restrictions
            self.dt = data.dt
            self.total_t = data.total_time
            self.type = data.type

        # Only allow datasets with identical time step size and total time
        if (self.dt == data.dt and self.total_t == data.total_time and self.type == data.type):
            self.datasets[tag] = data

        else:
            print("Cannot plot selected data sets: dt or total t mismatch")

        return

    def build(self, convert_out_dict):
        result = []
        for key in self.datasets:
            result.append(self.datasets[key].grid_x)
            result.append(self.datasets[key].data * convert_out_dict[self.type])
        return result

    def get_max_x(self):
        return np.amax([self.datasets[tag].grid_x[-1]
                        for tag in self.datasets])

    def get_maxtime(self):
        return np.amax([self.datasets[tag].total_time
                        for tag in self.datasets])

    def get_maxnumtsteps(self):
        return np.amax([self.datasets[tag].num_tsteps for tag in self.datasets])

class Integrated_Data_Group(Data_Group):
    def __init__(self):
        super().__init__()
        return
    
    def add(self, new_set):
        if not self.datasets: 
            # Allow the first set in to set the type restriction
            self.dt = new_set.dt
            self.total_t = new_set.total_time
            self.type = new_set.type

        # Only allow datasets with identical time step size and total time - this should always be the case after any integration; otherwise something has gone wrong
        if (self.type == new_set.type):
            self.datasets[new_set.tag()] = new_set

        return

class Scalable_Plot_State:
    def __init__(self, plot_obj=None):
        self.plot_obj = plot_obj
        self.xaxis_type = 'linear'
        self.yaxis_type = 'log'
        self.xlim = (-1,-1)
        self.ylim = (-1,-1)
        self.display_legend = 1
        self.do_freeze_axes = 0
        return

class Integration_Plot_State(Scalable_Plot_State):
    def __init__(self):
        super().__init__()
        self.mode = ""
        self.x_param = "None"   # This is usually "Time"
        self.global_gridx = None    # In some modes of operation every I_Set will have the same grid_x
        self.datagroup = Integrated_Data_Group()
        return

class Analysis_Plot_State(Scalable_Plot_State):
    # Object containing variables needed for each small plot on analysis tab
	# This is really a wrapper that enhances interactions between the Data_Group object and the embedded plot
    # There are currently four of these
    def __init__(self):
        super().__init__()
        self.time_index = 0
        self.data_filenames = []
        self.datagroup = Raw_Data_Group()
        return

    # def remove_duplicate_filenames(self):
    #     # Sets, unlike arrays, contain at most one copy of each item. Forcing an array into a set like this
    #     # is a fast way to scrub duplicate entries, which is needed because we don't want to waste time
    #     # plotting the same data set multiple times.
    #     # Unfortunately, sets are not indexable, so we convert back into arrays to regain index access ability.
            
    #     self.data_filenames = list(set(self.data_filenames))
    #     return

    def add_time_index(self, offset):
        self.time_index += offset
        if self.time_index < 0: 
            self.time_index = 0
        if self.time_index > self.datagroup.get_maxnumtsteps(): 
            self.time_index = self.datagroup.get_maxnumtsteps()
        return

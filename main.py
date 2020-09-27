#################################################
# Transient Electron Dynamics Simulator
# Model photoluminescent behavior in one-dimensional nanowire
# Last modified: Sep 12, 2020
# Author: Calvin Fai, Charles Hages
# Contact:
################################################# 

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
from scipy import integrate as intg

import tkinter.filedialog
import tkinter.scrolledtext as tkscrolledtext
import tkinter as tk
from tkinter import ttk # ttk is a sort of expansion pack to Tkinter, featuring additional elements and features.
import time
import datetime
import os
import tables
import itertools
from functools import partial # This lets us pass params to functions called by tkinter buttons
import finite

import pandas as pd # For bayesim compatibility
import bayesim.hdf5io as dd

np.seterr(divide='raise', over='warn', under='warn', invalid='raise')
class Param_Rule:
    # V2 Update
    # This class stores info to help Nanowire calculate Parameter values
    def __init__(self, variable, type, l_bound, r_bound=-1, l_boundval=-1, r_boundval=-1):
        self.variable = variable # e.g. N, P, E-Field
        self.type = type
        self.l_bound = l_bound
        self.r_bound = r_bound
        self.l_boundval = l_boundval
        self.r_boundval = r_boundval
        return

    def get(self):
        if (self.type == "POINT"):
            return((self.variable + ": " + self.type + " at x=" + '{:.4e}'.format(self.l_bound) + " with value: " + '{:.4e}'.format(self.l_boundval)))

        elif (self.type == "FILL"):
            return((self.variable + ": " + self.type + " from x=" + '{:.4e}'.format(self.l_bound) + " to " + '{:.4e}'.format(self.r_bound) + " with value: " + '{:.4e}'.format(self.l_boundval)))

        elif (self.type == "LINE"):
            return((self.variable + ": " + self.type + " from x=" + '{:.4e}'.format(self.l_bound) + " to " + '{:.4e}'.format(self.r_bound) + 
                    " with left value: " + '{:.4e}'.format(self.l_boundval) + " and right value: " + '{:.4e}'.format(self.r_boundval)))

        elif (self.type == "EXP"):
            return((self.variable + ": " + self.type + " from x=" + '{:.4e}'.format(self.l_bound) + " to " + '{:.4e}'.format(self.r_bound) + 
                    " with left value: " + '{:.4e}'.format(self.l_boundval) + " and right value: " + '{:.4e}'.format(self.r_boundval)))

        else:
            return("Error #101: Invalid initial condition")

    def is_edge(self):
        return (self.variable == "dEc" or self.variable == "chi")

class Parameter:
    # Helper class to store info about each of a Nanowire's parameters and initial distributions
    def __init__(self, is_edge, units):
        self.is_edge = is_edge
        # self.value can be a number (i.e. the parameter value is constant across the length of the nanowire)
        # or an array (i.e. the parameter value is spatially dependent)
        self.units = units
        self.value = 0
        self.param_rules = []
        return

class Nanowire:
    # A Nanowire object contains all information regarding the initial state of a nanowire
    def __init__(self):
        self.total_length = -1
        self.dx = -1
        self.grid_x_nodes = -1
        self.grid_x_edges = -1
        self.spacegrid_is_set = False
        self.param_dict = {"Mu_N":Parameter(is_edge=True, units="[cm^2 / V s]"), "Mu_P":Parameter(is_edge=True, units="[cm^2 / V s]"), 
                            "N0":Parameter(is_edge=False, units="[cm^-3]"), "P0":Parameter(is_edge=False, units="[cm^-3]"), 
                            "B":Parameter(is_edge=False, units="[cm^3 / s]"), "Tau_N":Parameter(is_edge=False, units="[ns]"), 
                            "Tau_P":Parameter(is_edge=False, units="[ns]"), "Sf":Parameter(is_edge=False, units="[cm / s]"), 
                            "Sb":Parameter(is_edge=False, units="[cm / s]"), "Temperature":Parameter(is_edge=True, units="[K]"), 
                            "Rel-Permitivity":Parameter(is_edge=True, units=""), "Ext_E-Field":Parameter(is_edge=True, units="[V/um]"),
                            "Theta":Parameter(is_edge=False, units="[cm^-1]"), "Alpha":Parameter(is_edge=False, units="[cm^-1]"), 
                            "Delta":Parameter(is_edge=False, units=""), "Frac-Emitted":Parameter(is_edge=False, units=""),
                            "init_deltaN":Parameter(is_edge=False, units="[cm^-3]"), "init_deltaP":Parameter(is_edge=False, units="[cm^-3]"), 
                            "init_E_field":Parameter(is_edge=True, units="[WIP]"), "Ec":Parameter(is_edge=True, units="[WIP]"),
                            "electron_affinity":Parameter(is_edge=True, units="[WIP]")}

        # This tells TEDs what flags there will be, but isn't used for storage until a simulation begins.
        # The reason why is that outside of simulations, we care more about whether a flag appears visually selected than its value
        self.flags_dict = {"ignore_alpha":0,
                           "symmetric_system":0}
        return

    def add_param_rule(self, param_name, new_rule):
        self.param_dict[param_name].param_rules.append(new_rule)
        self.update_param_toarray(param_name)
        return

    def swap_param_rules(self, param_name, i):
        self.param_dict[param_name].param_rules[i], self.param_dict[param_name].param_rules[i-1] = self.param_dict[param_name].param_rules[i-1], self.param_dict[param_name].param_rules[i]
        self.update_param_toarray(param_name)
        return

    def remove_param_rule(self, param_name, i):
        # TODO: Only one updated needed for multiple removals
        # Add shortcut for remove_all
        self.param_dict[param_name].param_rules.pop(i)
        self.update_param_toarray(param_name)
        return

    def update_param_toarray(self, param_name):
        # Recalculate a Parameter from its Param_Rules
        # This should be done every time the Param_Rules are changed
        param = self.param_dict[param_name]

        if param.is_edge:
            new_param_value = np.zeros(self.grid_x_edges.__len__())
        else:
            new_param_value = np.zeros(self.grid_x_nodes.__len__())

        for condition in param.param_rules:
            i = finite.toIndex(condition.l_bound, self.dx, self.total_length, param.is_edge)
            
            # If the left bound coordinate exceeds the width of the node toIndex() (which always rounds down) 
            # assigned, it should actually be mapped to the next node
            if (condition.l_bound - finite.toCoord(i, self.dx, param.is_edge) >= self.dx / 2): i += 1

            if (condition.type == "POINT"):
                new_param_value[i] = condition.l_boundval

            elif (condition.type == "FILL"):
                j = finite.toIndex(condition.r_bound, self.dx, self.total_length, param.is_edge)
                new_param_value[i:j+1] = condition.l_boundval

            elif (condition.type == "LINE"):
                slope = (condition.r_boundval - condition.l_boundval) / (condition.r_bound - condition.l_bound)
                j = finite.toIndex(condition.r_bound, self.dx, self.total_length, param.is_edge)

                ndx = np.linspace(0, self.dx * (j - i), j - i + 1)
                new_param_value[i:j+1] = condition.l_boundval + ndx * slope

            elif (condition.type == "EXP"):
                j = finite.toIndex(condition.r_bound, self.dx, self.total_length, param.is_edge)

                ndx = np.linspace(0, j - i, j - i + 1)
                try:
                    new_param_value[i:j+1] = condition.l_boundval * np.power(condition.r_boundval / condition.l_boundval, ndx / (j - i))
                except FloatingPointError:
                    print("Warning: Step size too large to resolve initial condition accurately")

        param.value = new_param_value
        return

    def DEBUG_print(self):
        #print("Behold the One Nanowire in its infinite glory:")
        mssg = ""
        if self.spacegrid_is_set:
            mssg += ("Space Grid is set\n")
            mssg += ("Nodes: {}\n".format(self.grid_x_nodes))
            mssg += ("Edges: {}\n".format(self.grid_x_edges))
        else:
            mssg += ("Grid is not set\n")

        for param in self.param_dict:
            mssg += ("{} {}: {}\n".format(param, self.param_dict[param].units, self.param_dict[param].value))

        return mssg
    
class Flag:
    # This class exists to solve a little problem involving tkinter checkbuttons: we get the value of a checkbutton using its tk.IntVar() 
    # but we interact with the checkbutton using the actual tk.CheckButton() element
    # So wrap both of those together in a single object and call it a day
    def __init__(self, tk_element, tk_var, value=0):
        self.tk_element = tk_element
        self.tk_var = tk_var
        self.value = value
        return

class Batchable:
    # Much like the flag class, the Batchable() serves to collect together various tk elements and values for the batch IC tool.
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
        # For some reason, Matplotlib legends don't like leading underscores
        if not for_matplotlib:
            return self.filename + "_" + self.type
        else:
            return (self.filename + "_" + self.type).strip('_')
        

class Raw_Data_Set(Data_Set):
    # Object containing all the metadata required to plot and integrate saved data sets
    def __init__(self, data, grid_x, node_x, edge_x, params_dict, type, filename, show_index):
        super().__init__(data, grid_x, params_dict, type, filename)
        self.node_x = node_x        # Array of x-coordinates corresponding to system nodes - needed to generate initial condition from data
        self.edge_x = edge_x        # node_x but for system node edges - also needed to regenerate ICs

        # node_x and grid_x will usually be identical, unless the data is a type (like E-field) that exists on edges
        # There's a little optimization that can be made here because grid_x will either be identical to node_x or edge_x, but that makes the code harder to follow

        self.show_index = show_index # Time step number data belongs to

        self.num_tsteps = int(0.5 + self.params_dict["Total-Time"] / self.params_dict["dt"])
        return

    def build(self):
        return np.vstack((self.grid_x, self.data))
    
class Integrated_Data_Set(Data_Set):
    def __init__(self, data, grid_x, params_dict, type, filename):
        super().__init__(data, grid_x, params_dict, type, filename)
        return

class Data_Group:
    def __init__(self):
        self.type = "None"
        self.datasets = {}
        return
    
    def get_maxval(self):
        return np.amax([np.amax(self.datasets[tag].data) for tag in self.datasets])
    
    def size(self):
        return len(self.datasets)
    
    def clear(self):
        self.datasets.clear()
        return

class Raw_Data_Group(Data_Group):
    def __init__(self):
        super().__init__()
        self.dt = -1
        self.total_t = -1
        return

    def add(self, data, tag):
        if (len(self.datasets) == 0): # Allow the first set in to set the dt and t restrictions
           self.dt = data.params_dict["dt"]
           self.total_t = data.params_dict["Total-Time"]
           self.type = data.type

        # Only allow datasets with identical time step size and total time
        if (self.dt == data.params_dict["dt"] and self.total_t == data.params_dict["Total-Time"] and self.type == data.type):
            self.datasets[tag] = data

        else:
            print("Cannot plot selected data sets: dt or total t mismatch")

        return

    def build(self):
        # FIXME: Needs to convert_out data
        #result = [item for item in self.datasets[key].build() for key in self.datasets]
        result = []
        for key in self.datasets:
            result.append(self.datasets[key].grid_x)
            result.append(self.datasets[key].data)
        return result

    def get_max_x(self):
        return np.amax([self.datasets[tag].params_dict["Total_length"] for tag in self.datasets])

    def get_maxtime(self):
        return np.amax([self.datasets[tag].params_dict["Total-Time"] for tag in self.datasets])

    def get_maxnumtsteps(self):
        return np.amax([self.datasets[tag].num_tsteps for tag in self.datasets])

class Integrated_Data_Group(Data_Group):
    def __init__(self):
        super().__init__()
        return
    
    def add(self, new_set):
        if (len(self.datasets) == 0): # Allow the first set in to set the type restriction
           self.type = new_set.type

        # Only allow datasets with identical time step size and total time - this should always be the case after any integration; otherwise something has gone wrong
        if (self.type == new_set.type):
            self.datasets[new_set.tag] = new_set

        return

class Scalable_Plot_State:
    def __init__(self, plot_obj=None):
        self.plot_obj = plot_obj
        self.xaxis_type = 'linear'
        self.yaxis_type = 'log'
        self.xlim = (-1,-1)
        self.ylim = (-1,-1)
        self.display_legend = 1
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
        if self.time_index < 0: self.time_index = 0
        if self.time_index > self.datagroup.get_maxnumtsteps(): 
            self.time_index = self.datagroup.get_maxnumtsteps()
        return

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

def tips(string, new_length):
    if string.__len__() > new_length * 2:
        return string[:new_length] + "..." + string[-new_length:]

    else:
        return string

class Notebook:
	# This is somewhat Java-like: everything about the GUI exists inside a class
    # A goal is to achieve total separation between this class (i.e. the GUI) and all mathematical operations, which makes this GUI reusable for different problems

    def __init__(self, title):
        ## Set up GUI, special variables needed to interact with certain tkinter objects, other "global" variables
        # Create the main Tkinter object
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', False)
        self.root.title(title)
        self.notebook = tk.ttk.Notebook(self.root)

        # Attempt to create a default config.txt file, if no config.txt can be found
        self.gen_default_config_file()

        # List of default directories for most I/O operations
        self.get_default_dirs()

        # Lists of conversions into and out of TEDs units (e.g. nm/s) from common units (e.g. cm/s)
        # Multiply the parameter values the user enters in common units by the corresponding coefficient in this dictionary to convert into TEDs units
        self.convert_in_dict = {"Mu_N": ((1e7) ** 2) / (1e9), "Mu_P": ((1e7) ** 2) / (1e9), # [cm^2 / V s] to [nm^2 / V ns]
                                "N0": ((1e-7) ** 3), "P0": ((1e-7) ** 3),                   # [cm^-3] to [nm^-3]
                                "Thickness": 1, "dx": 1,
                                "B": ((1e7) ** 3) / (1e9),                                  # [cm^3 / s] to [nm^3 / ns]
                                "Tau_N": 1, "Tau_P": 1,                                     # [ns]
                                "Sf": (1e7) / (1e9), "Sb": (1e7) / (1e9),                   # [cm / s] to [nm / ns]
                                "Temperature": 1, "Rel-Permitivity": 1, 
                                "Ext_E-Field": 1e-3,                                        # [V/um] to [V/nm]
                                "Theta": 1e-7, "Alpha": 1e-7,                               # [cm^-1] to [nm^-1]
                                "Delta": 1, "Frac-Emitted": 1,
                                "init_deltaN": ((1e-7) ** 3), "init_deltaP": ((1e-7) ** 3),
                                "init_E_field": 1, "Ec": 1, "electron_affinity": 1,
                                "N": ((1e-7) ** 3), "P": ((1e-7) ** 3),                     # [cm^-3] to [nm^-3]
                                "E_field": 1}
        
        # FIXME: Check these
        self.convert_in_dict["RR"] = self.convert_in_dict["B"] * self.convert_in_dict["N"] * self.convert_in_dict["P"]
        self.convert_in_dict["NRR"] = self.convert_in_dict["N"] * self.convert_in_dict["Tau_N"]
        self.convert_in_dict["PL"] = self.convert_in_dict["RR"] * self.convert_in_dict["Theta"]
        # Multiply the parameter values TEDs is using by the corresponding coefficient in this dictionary to convert back into common units
        self.convert_out_dict = {}
        for param in self.convert_in_dict:
            self.convert_out_dict[param] = self.convert_in_dict[param] ** -1

        # Tkinter checkboxes and radiobuttons require special variables to extract user input
        # IntVars or BooleanVars are sufficient for binary choices e.g. whether a checkbox is checked
        # while StringVars are more suitable for open-ended choices e.g. selecting one mode from a list
        self.check_ignore_recycle = tk.IntVar()
        self.check_symmetric = tk.IntVar()
        self.check_do_ss = tk.IntVar()
        self.check_reset_params = tk.IntVar()
        self.check_reset_inits = tk.IntVar()
        self.check_display_legend = tk.IntVar()
        self.check_autointegrate = tk.IntVar(value=1)

        self.check_calculate_init_material_expfactor = tk.IntVar()
        self.AIC_stim_mode = tk.StringVar()
        self.AIC_gen_power_mode = tk.StringVar()
        self.active_analysisplot_ID = tk.IntVar()
        self.active_integrationplot_ID = tk.IntVar()

        self.init_shape_selection = tk.StringVar()
        self.init_var_selection = tk.StringVar()
        self.HIC_viewer_selection = tk.StringVar()
        self.EIC_var_selection = tk.StringVar()
        self.display_selection = tk.StringVar()

        self.check_bay_params = {"Mu_N":tk.IntVar(), "Mu_P":tk.IntVar(), "N0":tk.IntVar(), "P0":tk.IntVar(),
                        "B":tk.IntVar(), "Tau_N":tk.IntVar(), "Tau_P":tk.IntVar(), "Sf":tk.IntVar(), \
                        "Sb":tk.IntVar(), "Temperature":tk.IntVar(), "Rel-Permitivity":tk.IntVar(), \
                        "Theta":tk.IntVar(), "Alpha":tk.IntVar(), "Delta":tk.IntVar(), "Frac-Emitted":tk.IntVar()}
        
        self.bay_mode = tk.StringVar(value="model")

        # Flags and containers for IC arrays
        self.nanowire = Nanowire()

        self.HIC_list = []
        self.HIC_listbox_currentparam = ""
        self.IC_file_list = None
        self.init_N = None
        self.init_P = None
        self.init_E_field = None
        self.init_Ec = None
        self.init_Chi = None
        self.IC_file_name = ""
        self.using_AIC = False

        self.sim_N = None
        self.sim_P = None
        self.sim_E_field = None

        self.carry_include_N = tk.IntVar()
        self.carry_include_P = tk.IntVar()
        self.carry_include_E_field = tk.IntVar()
        
        # Helpers, flags, and containers for analysis plots
        self.analysis_plots = [Analysis_Plot_State(), Analysis_Plot_State(), Analysis_Plot_State(), Analysis_Plot_State()]
        self.integration_plots = [Integration_Plot_State(), Integration_Plot_State()]
        self.data_var = tk.StringVar()
        self.fetch_PLmode = tk.StringVar()
        self.fetch_intg_mode = tk.StringVar()
        self.yaxis_type = tk.StringVar()
        self.xaxis_type = tk.StringVar()

        self.menu_bar = tk.Menu(self.notebook)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Manage Initial Condition Files", command=partial(tk.filedialog.askopenfilenames, title="This window does not open anything - Use this window to move or delete IC files", initialdir=self.default_dirs["Initial"]))
        self.file_menu.add_command(label="Manage Data Files", command=partial(tk.filedialog.askdirectory, title="This window does not open anything - Use this window to move or delete data files",initialdir=self.default_dirs["Data"]))
        self.file_menu.add_command(label="Manage Export Files", command=partial(tk.filedialog.askopenfilenames, title="This window does not open anything - Use this window to move or delete export files",initialdir=self.default_dirs["PL"]))
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.view_menu.add_command(label="Toggle Fullscreen", command=self.toggle_fullscreen)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)

        self.tool_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.tool_menu.add_command(label="Batch Op. Tool", command=self.do_batch_popup)
        self.menu_bar.add_cascade(label="Tools", menu=self.tool_menu)

        #self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        #self.help_menu.add_command(label="About", command=self.do_about_popup)
        #self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        # Check when popup menus open and close
        self.sys_summary_popup_isopen = False
        self.sys_param_shortcut_popup_isopen = False
        self.batch_popup_isopen = False
        self.resetIC_popup_isopen = False
        self.overwrite_popup_isopen = False
        self.integration_popup_isopen = False
        self.integration_getbounds_popup_isopen = False
        self.PL_xaxis_popup_isopen = False
        self.change_axis_popup_isopen = False
        self.plotter_popup_isopen = False
        self.IC_carry_popup_isopen = False
        self.bayesim_popup_isopen = False

        self.root.config(menu=self.menu_bar)

        s = ttk.Style()
        s.theme_use('classic')


        self.add_tab_inputs()
        self.add_tab_simulate()
        self.add_tab_analyze()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_selected)
        self.tab_inputs.bind("<<NotebookTabChanged>>", self.on_input_subtab_selected)

        print("Initialization complete")
        print("Detecting Initial Condition and Data Directories...")
        try:
            os.mkdir(self.default_dirs["Initial"])
            print("No Initial Condition Directory detected; automatically creating...")
        except FileExistsError:
            print("Initial Condition Directory detected")
        
        try:
            os.mkdir(self.default_dirs["Data"])
            print("No Data Directory detected; automatically creating...")
        except FileExistsError:
            print("Data Directory detected")

        try:
            os.mkdir(self.default_dirs["PL"])
            print("No PL Directory detected; automatically creating...")
        except FileExistsError:
            print("PL Directory detected")

        return

    def run(self):
        self.notebook.pack(expand=1, fill="both")
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()

        self.root.geometry('%dx%d+0+0' % (width,height))
        self.root.mainloop()
        return

    def toggle_fullscreen(self):
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
        return

    def gen_default_config_file(self):
        if not os.path.isfile("config.txt"):
            print("Could not find config.txt; generating default...")
            with open("config.txt", 'w+') as ofstream:
                ofstream.write("# Directories\n")
                ofstream.write("Initial: \"Initial\"\n")
                ofstream.write("Data: \"Data\"\n")
                ofstream.write("Analysis: \"Analysis\"\n")

        return
    
    def get_default_dirs(self):
        with open("config.txt", 'r') as ifstream:
            for line in ifstream:
                if (line == "" or '#' in line): continue

                elif (line[0:line.find(':')] == "Initial"):
                    default_initial = line[(line.find('\"')+1):].strip().strip('\"\n')

                elif (line[0:line.find(':')] == "Data"):
                    default_simdata = line[line.find('\"')+1:].strip().strip('\"\n')

                elif (line[0:line.find(':')] == "Analysis"):
                    default_analysis = line[line.find('\"')+1:].strip().strip('\"\n')

                else: continue

            self.default_dirs = {"Initial":default_initial, "Data":default_simdata, "PL":default_analysis}
        return

    ## Create GUI elements for each tab
	# Tkinter works a bit like a bulletin board - we declare an overall frame and pin things to it at specified locations
	# This includes other frames, which is evident in how the tab_inputs has three sub-tabs pinned to itself.
    def add_tab_inputs(self):
        self.tab_inputs = tk.ttk.Notebook(self.notebook)
        self.tab_analytical_init = tk.ttk.Frame(self.tab_inputs)
        self.tab_rules_init = tk.ttk.Frame(self.tab_inputs)
        self.tab_explicit_init = tk.ttk.Frame(self.tab_inputs)

        var_dropdown_list = [str(param + self.nanowire.param_dict[param].units) for param in self.nanowire.param_dict]
        HIC_method_dropdown_list = ["POINT", "FILL", "LINE", "EXP"]
        unitless_dropdown_list = [param for param in self.nanowire.param_dict]
        
        self.line_sep_style = tk.ttk.Style()
        self.line_sep_style.configure("Grey Bar.TSeparator", background='#000000', padding=160)

        self.header_style = tk.ttk.Style()
        self.header_style.configure("Header.TLabel", background='#D0FFFF',highlightbackground='#000000')

		# We use the grid location specifier for general placement and padx/pady for fine-tuning
		# The other two options are the pack specifier, which doesn't really provide enough versatility,
		# and absolute coordinates, which maximize versatility but are a pain to adjust manually.
        self.IO_frame = tk.ttk.Frame(self.tab_inputs)
        self.IO_frame.grid(row=0,column=0,columnspan=2, pady=(25,0))
        
        self.load_ICfile_button = tk.ttk.Button(self.IO_frame, text="Load", command=self.select_init_file)
        self.load_ICfile_button.grid(row=0,column=0)

        self.DEBUG_BUTTON = tk.ttk.Button(self.IO_frame, text="debug", command=self.DEBUG)
        self.DEBUG_BUTTON.grid(row=0,column=1)

        self.save_ICfile_button = tk.ttk.Button(self.IO_frame, text="Save", command=self.save_ICfile)
        self.save_ICfile_button.grid(row=0,column=2)

        self.reset_IC_button = tk.ttk.Button(self.IO_frame, text="Reset", command=self.reset_IC)
        self.reset_IC_button.grid(row=0, column=3)

        self.spacegrid_frame = tk.ttk.Frame(self.tab_inputs)
        self.spacegrid_frame.grid(row=1,column=0,columnspan=2)

        self.steps_head = tk.ttk.Label(self.spacegrid_frame, text="Space Grid", style="Header.TLabel")
        self.steps_head.grid(row=0,column=0,columnspan=2)

        self.thickness_label = tk.ttk.Label(self.spacegrid_frame, text="Thickness [nm]")
        self.thickness_label.grid(row=1,column=0)

        self.thickness_entry = tk.ttk.Entry(self.spacegrid_frame, width=9)
        self.thickness_entry.grid(row=1,column=1)

        self.dx_label = tk.ttk.Label(self.spacegrid_frame, text="Node width [nm]")
        self.dx_label.grid(row=2,column=0)

        self.dx_entry = tk.ttk.Entry(self.spacegrid_frame, width=9)
        self.dx_entry.grid(row=2,column=1)

        self.params_frame = tk.ttk.Frame(self.tab_inputs)
        self.params_frame.grid(row=2,column=0,columnspan=2, rowspan=4)

        self.system_params_head = tk.ttk.Label(self.params_frame, text="System Parameters",style="Header.TLabel")
        self.system_params_head.grid(row=0, column=0,columnspan=2)
        
        self.system_params_shortcut_button = tk.ttk.Button(self.params_frame, text="Short-cut Param Entry Tool", command=self.do_sys_param_shortcut_popup)
        self.system_params_shortcut_button.grid(row=1,column=0,columnspan=2)

        self.flags_frame = tk.ttk.Frame(self.tab_inputs)
        self.flags_frame.grid(row=6,column=0,columnspan=2)

        self.flags_head = tk.ttk.Label(self.flags_frame, text="Flags", style="Header.TLabel")
        self.flags_head.grid(row=0,column=0,columnspan=2)
        
        # TODO: Procedurally generated elements for flags
        self.ignore_recycle_checkbutton = tk.ttk.Checkbutton(self.flags_frame, text="Ignore photon recycle?", variable=self.check_ignore_recycle, onvalue=1, offvalue=0)
        self.ignore_recycle_checkbutton.grid(row=1,column=0)

        self.symmetry_checkbutton = tk.ttk.Checkbutton(self.flags_frame, text="Symmetric system?", variable=self.check_symmetric, onvalue=1, offvalue=0)
        self.symmetry_checkbutton.grid(row=2,column=0)

        self.ICtab_status = tk.Text(self.tab_inputs, width=20,height=4)
        self.ICtab_status.grid(row=7, column=0, columnspan=2)
        self.ICtab_status.configure(state='disabled')
        
        self.system_log_button = tk.ttk.Button(self.tab_inputs, text="Print System Summary", command=self.do_sys_summary_popup)
        self.system_log_button.grid(row=8,column=0,columnspan=2)

        self.line1_separator = tk.ttk.Separator(self.tab_inputs, orient="vertical", style="Grey Bar.TSeparator")
        self.line1_separator.grid(row=0,rowspan=30,column=2,pady=(24,0),sticky="ns")
     
        ## Analytical Initial Condition (AIC):

        # An empty GUI element is used to force the analytical IC elements into the correct position.
        # Note that self.tab_analytical_init is a sub-frame attached to the overall self.tab_inputs
        # Normally, the first element of a frame like self.tab_analytical_init would start at row=0, column=0
        # instead of column=2. Starting at column=2 is NOT A TYPO. self.tab_analytical_init is attached to
        # the notebook self.tab_inputs, so it inherits the first two columns of self.tab_inputs.
        self.AIC_frame = tk.ttk.Frame(self.tab_analytical_init)
        self.AIC_frame.grid(row=0,column=0, padx=(330,0))

        self.AIC_head = tk.ttk.Label(self.AIC_frame, text="Analytical Init. Cond.", style="Header.TLabel")
        self.AIC_head.grid(row=0,column=0,columnspan=3)

        # A sub-frame attached to a sub-frame
        # With these we can group related elements into a common region
        self.material_param_frame = tk.Frame(self.AIC_frame, highlightbackground="black", highlightthicknes=1)
        self.material_param_frame.grid(row=1,column=0)

        self.material_param_label = tk.Label(self.material_param_frame, text="Material Params - Select One")
        self.material_param_label.grid(row=0,column=0,columnspan=4)

        self.hline1_separator = tk.ttk.Separator(self.material_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline1_separator.grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        self.calc_AIC_expfactor = tk.ttk.Radiobutton(self.material_param_frame, variable=self.check_calculate_init_material_expfactor, value=1)
        self.calc_AIC_expfactor.grid(row=2,column=0)

        self.calc_AIC_expfactor_label = tk.Label(self.material_param_frame, text="Option 1")
        self.calc_AIC_expfactor_label.grid(row=2,column=1)

        self.A0_label = tk.Label(self.material_param_frame, text="A0 [cm^-1 eV^-γ]")
        self.A0_label.grid(row=2,column=2)

        self.A0_entry = tk.ttk.Entry(self.material_param_frame, width=9)
        self.A0_entry.grid(row=2,column=3)

        self.Eg_label = tk.Label(self.material_param_frame, text="Eg [eV]")
        self.Eg_label.grid(row=3,column=2)

        self.Eg_entry = tk.ttk.Entry(self.material_param_frame, width=9)
        self.Eg_entry.grid(row=3,column=3)

        self.direct_AIC_stim = tk.ttk.Radiobutton(self.material_param_frame, variable=self.AIC_stim_mode, value="direct")
        self.direct_AIC_stim.grid(row=4,column=2)

        self.direct_AIC_stim_label = tk.Label(self.material_param_frame,text="Direct (γ=1/2)")
        self.direct_AIC_stim_label.grid(row=4,column=3)

        self.indirect_AIC_stim = tk.ttk.Radiobutton(self.material_param_frame, variable=self.AIC_stim_mode, value="indirect")
        self.indirect_AIC_stim.grid(row=5,column=2)

        self.indirect_AIC_stim_label = tk.Label(self.material_param_frame,text="Indirect (γ=2)")
        self.indirect_AIC_stim_label.grid(row=5,column=3)

        self.hline2_separator = tk.ttk.Separator(self.material_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline2_separator.grid(row=6,column=0,columnspan=30, pady=(5,5), sticky="ew")

        self.enter_AIC_expfactor = tk.ttk.Radiobutton(self.material_param_frame, variable=self.check_calculate_init_material_expfactor, value=0)
        self.enter_AIC_expfactor.grid(row=7,column=0)

        self.enter_AIC_expfactor_label = tk.Label(self.material_param_frame, text="Option 2")
        self.enter_AIC_expfactor_label.grid(row=7,column=1)

        self.AIC_expfactor_label = tk.Label(self.material_param_frame, text="α [cm^-1]")
        self.AIC_expfactor_label.grid(row=8,column=2)

        self.AIC_expfactor_entry = tk.ttk.Entry(self.material_param_frame, width=9)
        self.AIC_expfactor_entry.grid(row=8,column=3)

        self.pulse_laser_frame = tk.Frame(self.AIC_frame, highlightbackground="black", highlightthicknes=1)
        self.pulse_laser_frame.grid(row=1,column=1, padx=(20,0))

        self.pulse_laser_label = tk.Label(self.pulse_laser_frame, text="Pulse Laser Params")
        self.pulse_laser_label.grid(row=0,column=0,columnspan=4)

        self.hline3_separator = tk.ttk.Separator(self.pulse_laser_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline3_separator.grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        self.pulse_freq_label = tk.Label(self.pulse_laser_frame, text="Pulse frequency [kHz]")
        self.pulse_freq_label.grid(row=2,column=2)

        self.pulse_freq_entry = tk.ttk.Entry(self.pulse_laser_frame, width=9)
        self.pulse_freq_entry.grid(row=2,column=3)

        self.pulse_wavelength_label = tk.Label(self.pulse_laser_frame, text="Wavelength [nm]")
        self.pulse_wavelength_label.grid(row=3,column=2)

        self.pulse_wavelength_entry = tk.ttk.Entry(self.pulse_laser_frame, width=9)
        self.pulse_wavelength_entry.grid(row=3,column=3)

        self.gen_power_param_frame = tk.Frame(self.AIC_frame, highlightbackground="black", highlightthicknes=1)
        self.gen_power_param_frame.grid(row=1,column=2, padx=(20,0))

        self.gen_power_param_label = tk.Label(self.gen_power_param_frame, text="Generation/Power Params - Select One")
        self.gen_power_param_label.grid(row=0,column=0,columnspan=4)

        self.hline4_separator = tk.ttk.Separator(self.gen_power_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline4_separator.grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        self.power_spot = tk.ttk.Radiobutton(self.gen_power_param_frame, variable=self.AIC_gen_power_mode, value="power-spot")
        self.power_spot.grid(row=2,column=0)

        self.power_spot_label = tk.Label(self.gen_power_param_frame, text="Option 1")
        self.power_spot_label.grid(row=2,column=1)

        self.power_label = tk.Label(self.gen_power_param_frame, text="Power [uW]")
        self.power_label.grid(row=2,column=2)

        self.power_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.power_entry.grid(row=2,column=3)

        self.spotsize_label = tk.Label(self.gen_power_param_frame, text="Spot size [cm^2]")
        self.spotsize_label.grid(row=3,column=2)

        self.spotsize_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.spotsize_entry.grid(row=3,column=3)

        self.hline5_separator = tk.ttk.Separator(self.gen_power_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline5_separator.grid(row=4,column=0,columnspan=30, pady=(5,5), sticky="ew")

        self.power_density_rb = tk.ttk.Radiobutton(self.gen_power_param_frame, variable=self.AIC_gen_power_mode, value="density")
        self.power_density_rb.grid(row=5,column=0)

        self.power_density_rb_label = tk.Label(self.gen_power_param_frame,text="Option 2")
        self.power_density_rb_label.grid(row=5,column=1)

        self.power_density_label = tk.Label(self.gen_power_param_frame, text="Power Density [uW/cm^2]")
        self.power_density_label.grid(row=5,column=2)

        self.power_density_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.power_density_entry.grid(row=5,column=3)

        self.hline6_separator = tk.ttk.Separator(self.gen_power_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline6_separator.grid(row=6,column=0,columnspan=30, pady=(5,5), sticky="ew")

        self.max_gen_rb = tk.ttk.Radiobutton(self.gen_power_param_frame, variable=self.AIC_gen_power_mode, value="max-gen")
        self.max_gen_rb.grid(row=7,column=0)

        self.max_gen_rb_label = tk.Label(self.gen_power_param_frame, text="Option 3")
        self.max_gen_rb_label.grid(row=7,column=1)

        self.max_gen_label = tk.Label(self.gen_power_param_frame, text="Max Generation [carr/cm^3]")
        self.max_gen_label.grid(row=7,column=2)

        self.max_gen_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.max_gen_entry.grid(row=7,column=3)

        self.hline7_separator = tk.ttk.Separator(self.gen_power_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline7_separator.grid(row=8,column=0,columnspan=30, pady=(5,5), sticky="ew")

        self.total_gen_rb = tk.ttk.Radiobutton(self.gen_power_param_frame, variable=self.AIC_gen_power_mode, value="total-gen")
        self.total_gen_rb.grid(row=9,column=0)

        self.total_gen_rb_label = tk.Label(self.gen_power_param_frame, text="Option 4")
        self.total_gen_rb_label.grid(row=9,column=1)

        self.total_gen_label = tk.Label(self.gen_power_param_frame, text="Total Generation [carr/cm^3]")
        self.total_gen_label.grid(row=9,column=2)

        self.total_gen_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.total_gen_entry.grid(row=9,column=3)

        self.load_AIC_button = tk.ttk.Button(self.AIC_frame, text="Generate Initial Condition", command=self.add_AIC)
        self.load_AIC_button.grid(row=2,column=0,columnspan=3)

        self.AIC_description = tk.Message(self.AIC_frame, text="The Analytical Initial Condition uses the above numerical parameters to generate an initial carrier distribution based on an exponential decay equation.", width=320)
        self.AIC_description.grid(row=3,column=0,columnspan=3)
        
        self.AIC_fig = Figure(figsize=(5,3.1))
        self.AIC_subplot = self.AIC_fig.add_subplot(111)
        self.AIC_canvas = tkagg.FigureCanvasTkAgg(self.AIC_fig, master=self.AIC_frame)
        self.AIC_plotwidget = self.AIC_canvas.get_tk_widget()
        self.AIC_plotwidget.grid(row=4, column=0, columnspan=3)
        
        self.AIC_toolbar_frame = tk.ttk.Frame(master=self.AIC_frame)
        self.AIC_toolbar_frame.grid(row=5,column=0,columnspan=3)
        self.AIC_toolbar = tkagg.NavigationToolbar2Tk(self.AIC_canvas, self.AIC_toolbar_frame)
        
        ## Heuristic Initial Condition(HIC):

        self.param_rules_frame = tk.ttk.Frame(self.tab_rules_init)
        self.param_rules_frame.grid(row=0,column=0,padx=(370,0))

        self.HIC_list_title = tk.ttk.Label(self.param_rules_frame, text="Parameter Rules", style="Header.TLabel")
        self.HIC_list_title.grid(row=0,column=0,columnspan=3)

        self.HIC_listbox = tk.Listbox(self.param_rules_frame, width=86,height=8)
        self.HIC_listbox.grid(row=1,rowspan=3,column=0,columnspan=3, padx=(32,32))

        self.HIC_var_label = tk.ttk.Label(self.param_rules_frame, text="Select parameter to edit:")
        self.HIC_var_label.grid(row=4,column=0)
        
        self.HIC_var_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, self.init_var_selection, var_dropdown_list[0], *var_dropdown_list)
        self.HIC_var_dropdown.grid(row=4,column=1)

        self.HIC_method_label = tk.ttk.Label(self.param_rules_frame, text="Select calculation method:")
        self.HIC_method_label.grid(row=5,column=0)

        self.HIC_method_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, self.init_shape_selection, HIC_method_dropdown_list[0], *HIC_method_dropdown_list)
        self.HIC_method_dropdown.grid(row=5, column=1)

        self.HIC_lbound_label = tk.ttk.Label(self.param_rules_frame, text="Left bound coordinate:")
        self.HIC_lbound_label.grid(row=6, column=0)

        self.HIC_lbound_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.HIC_lbound_entry.grid(row=6,column=1)

        self.HIC_rbound_label = tk.ttk.Label(self.param_rules_frame, text="Right bound coordinate:")
        self.HIC_rbound_label.grid(row=7, column=0)

        self.HIC_rbound_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.HIC_rbound_entry.grid(row=7,column=1)

        self.HIC_lvalue_label = tk.ttk.Label(self.param_rules_frame, text="Left bound value:")
        self.HIC_lvalue_label.grid(row=8, column=0)

        self.HIC_lvalue_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.HIC_lvalue_entry.grid(row=8,column=1)

        self.HIC_rvalue_label = tk.ttk.Label(self.param_rules_frame, text="Right bound value:")
        self.HIC_rvalue_label.grid(row=9, column=0)

        self.HIC_rvalue_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.HIC_rvalue_entry.grid(row=9,column=1)

        self.add_HIC_button = tk.ttk.Button(self.param_rules_frame, text="Add new parameter rule", command=self.add_HIC)
        self.add_HIC_button.grid(row=10,column=0,columnspan=2)

        self.delete_HIC_button = tk.ttk.Button(self.param_rules_frame, text="Delete highlighted rule", command=self.delete_HIC)
        self.delete_HIC_button.grid(row=4,column=2)

        self.deleteall_HIC_button = tk.ttk.Button(self.param_rules_frame, text="Delete all rules for this parameter", command=self.deleteall_HIC)
        self.deleteall_HIC_button.grid(row=5,column=2)

        self.HIC_description = tk.Message(self.param_rules_frame, text="The Parameter Toolkit uses a series of rules and patterns to build a spatially dependent distribution for any parameter.", width=250)
        self.HIC_description.grid(row=6,rowspan=3,column=2,columnspan=2)

        self.HIC_description2 = tk.Message(self.param_rules_frame, text="Warning: Rules are applied from top to bottom. Order matters!", width=250)
        self.HIC_description2.grid(row=9,rowspan=3,column=2,columnspan=2)
        
        # These plots were previously attached to self.tab_inputs so that it was visible on all three IC tabs,
        # but it was hard to position them correctly.
        # Attaching to the Parameter Toolkit makes them easier to position
        self.custom_param_fig = Figure(figsize=(5,3.1))
        self.custom_param_subplot = self.custom_param_fig.add_subplot(111)
        self.custom_param_subplot.format_coord = lambda x, y: ""
        self.custom_param_canvas = tkagg.FigureCanvasTkAgg(self.custom_param_fig, master=self.param_rules_frame)
        self.custom_param_plotwidget = self.custom_param_canvas.get_tk_widget()
        self.custom_param_plotwidget.grid(row=12, column=0, columnspan=2)

        self.custom_param_toolbar_frame = tk.ttk.Frame(master=self.param_rules_frame)
        self.custom_param_toolbar_frame.grid(row=13,column=0,columnspan=2)
        self.custom_param_toolbar = tkagg.NavigationToolbar2Tk(self.custom_param_canvas, self.custom_param_toolbar_frame)
        
        self.recent_param_fig = Figure(figsize=(5,3.1))
        self.recent_param_subplot = self.recent_param_fig.add_subplot(111)
        self.recent_param_subplot.format_coord = lambda x, y: ""
        self.recent_param_canvas = tkagg.FigureCanvasTkAgg(self.recent_param_fig, master=self.param_rules_frame)
        self.recent_param_plotwidget = self.recent_param_canvas.get_tk_widget()
        self.recent_param_plotwidget.grid(row=12,column=2,columnspan=2)

        self.recent_param_toolbar_frame = tk.ttk.Frame(master=self.param_rules_frame)
        self.recent_param_toolbar_frame.grid(row=13,column=2,columnspan=2)
        self.recent_param_toolbar = tkagg.NavigationToolbar2Tk(self.recent_param_canvas, self.recent_param_toolbar_frame)

        self.moveup_HIC_button = tk.ttk.Button(self.param_rules_frame, text="⇧", command=self.moveup_HIC)
        self.moveup_HIC_button.grid(row=1,column=4)

        self.HIC_viewer_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, self.HIC_viewer_selection, unitless_dropdown_list[0], *unitless_dropdown_list)
        self.HIC_viewer_dropdown.grid(row=2,column=4)

        self.HIC_view_button = tk.ttk.Button(self.param_rules_frame, text="Change view", command=self.refresh_paramrule_listbox)
        self.HIC_view_button.grid(row=2,column=5)

        self.movedown_HIC_button = tk.ttk.Button(self.param_rules_frame, text="⇩", command=self.movedown_HIC)
        self.movedown_HIC_button.grid(row=3,column=4)

        ## Explicit Inital Condition(EIC):

        self.listupload_frame = tk.ttk.Frame(self.tab_explicit_init)
        self.listupload_frame.grid(row=0,column=0,padx=(440,0))

        self.EIC_description = tk.Message(self.listupload_frame, text="This tab provides an option to directly import a list of data points, on which the TED will do linear interpolation to fit to the specified spacing mesh.", width=360)
        self.EIC_description.grid(row=0,column=0)
        
        self.EIC_dropdown = tk.ttk.OptionMenu(self.listupload_frame, self.EIC_var_selection, unitless_dropdown_list[0], *unitless_dropdown_list)
        self.EIC_dropdown.grid(row=1,column=0)

        self.add_EIC_button = tk.ttk.Button(self.listupload_frame, text="Import", command=self.add_EIC)
        self.add_EIC_button.grid(row=2,column=0)
        
        self.EIC_fig = Figure(figsize=(6,3.8))
        self.EIC_subplot = self.EIC_fig.add_subplot(111)
        self.EIC_canvas = tkagg.FigureCanvasTkAgg(self.EIC_fig, master=self.listupload_frame)
        self.EIC_plotwidget = self.EIC_canvas.get_tk_widget()
        self.EIC_plotwidget.grid(row=0, rowspan=3,column=1)
        
        self.EIC_toolbar_frame = tk.ttk.Frame(master=self.listupload_frame)
        self.EIC_toolbar_frame.grid(row=3,column=1)
        self.EIC_toolbar = tkagg.NavigationToolbar2Tk(self.EIC_canvas, self.EIC_toolbar_frame)

        # Dictionaries of parameter entry boxes
        
        self.sys_flag_dict = {"ignore_alpha":Flag(self.ignore_recycle_checkbutton, self.check_ignore_recycle),
                              "symmetric_system":Flag(self.symmetry_checkbutton, self.check_symmetric)}

        self.analytical_entryboxes_dict = {"A0":self.A0_entry, "Eg":self.Eg_entry, "AIC_expfactor":self.AIC_expfactor_entry, "Pulse_Freq":self.pulse_freq_entry, 
                                           "Pulse_Wavelength":self.pulse_wavelength_entry, "Power":self.power_entry, "Spotsize":self.spotsize_entry, "Power_Density":self.power_density_entry,
                                           "Max_Gen":self.max_gen_entry, "Total_Gen":self.total_gen_entry}

        # Attach sub-frames to input tab and input tab to overall notebook
        self.tab_inputs.add(self.tab_rules_init, text="Parameter Toolkit")
        self.tab_inputs.add(self.tab_analytical_init, text="Analytical Init. Cond.")
        self.tab_inputs.add(self.tab_explicit_init, text="Parameter List Upload")
        self.notebook.add(self.tab_inputs, text="Inputs")
        return

    def add_tab_simulate(self):
        self.tab_simulate = tk.ttk.Frame(self.notebook)

        self.choose_ICfile_title = tk.ttk.Label(self.tab_simulate, text="Select Init. Cond.", style="Header.TLabel")
        self.choose_ICfile_title.grid(row=0,column=0,columnspan=2, padx=(9,12))

        self.simtime_label = tk.ttk.Label(self.tab_simulate, text="Simulation Time [ns]")
        self.simtime_label.grid(row=2,column=0)

        self.simtime_entry = tk.ttk.Entry(self.tab_simulate, width=9)
        self.simtime_entry.grid(row=2,column=1)

        self.dt_label = tk.ttk.Label(self.tab_simulate, text="dt [ns]")
        self.dt_label.grid(row=3,column=0)

        self.dt_entry = tk.ttk.Entry(self.tab_simulate, width=9)
        self.dt_entry.grid(row=3,column=1)

        self.do_ss_checkbutton = tk.ttk.Checkbutton(self.tab_simulate, text="Steady State External Stimulation?", variable=self.check_do_ss, onvalue=1, offvalue=0)
        self.do_ss_checkbutton.grid(row=5,column=0)

        self.calculate_NP = tk.ttk.Button(self.tab_simulate, text="Calculate ΔN,ΔP", command=self.do_Batch)
        self.calculate_NP.grid(row=6,column=0,columnspan=2,padx=(9,12))

        self.status_label = tk.ttk.Label(self.tab_simulate, text="Status")
        self.status_label.grid(row=7, column=0, columnspan=2)

        self.status = tk.Text(self.tab_simulate, width=28,height=4)
        self.status.grid(row=8, rowspan=2, column=0, columnspan=2)
        self.status.configure(state='disabled')

        self.line3_separator = tk.ttk.Separator(self.tab_simulate, orient="vertical", style="Grey Bar.TSeparator")
        self.line3_separator.grid(row=0,rowspan=30,column=2,sticky="ns")

        self.subtitle = tk.ttk.Label(self.tab_simulate, text="1-D Carrier Sim (rk4 mtd), with photon propagation")
        self.subtitle.grid(row=0,column=3,columnspan=3)
        
        self.sim_fig = Figure(figsize=(10,6.2))
        self.n_subplot = self.sim_fig.add_subplot(221)
        self.p_subplot = self.sim_fig.add_subplot(222)
        self.E_subplot = self.sim_fig.add_subplot(223)
        self.sim_canvas = tkagg.FigureCanvasTkAgg(self.sim_fig, master=self.tab_simulate)
        self.sim_plot_widget = self.sim_canvas.get_tk_widget()
        self.sim_plot_widget.grid(row=1,column=3,rowspan=12,columnspan=2)
        
        self.simfig_toolbar_frame = tk.ttk.Frame(master=self.tab_simulate)
        self.simfig_toolbar_frame.grid(row=13,column=3,columnspan=2)
        self.simfig_toolbar = tkagg.NavigationToolbar2Tk(self.sim_canvas, self.simfig_toolbar_frame)

        self.notebook.add(self.tab_simulate, text="Simulate")
        return

    def add_tab_analyze(self):
        self.tab_analyze = tk.ttk.Frame(self.notebook)

        self.analysis_title = tk.ttk.Label(self.tab_analyze, text="Plot and Integrate Saved Datasets", style="Header.TLabel")
        self.analysis_title.grid(row=0,column=0,columnspan=8)
        
        self.analyze_fig = Figure(figsize=(9.8,6))
        # add_subplot() starts counting indices with 1 instead of 0
        self.analyze_subplot0 = self.analyze_fig.add_subplot(221)
        self.analyze_subplot1 = self.analyze_fig.add_subplot(222)
        self.analyze_subplot2 = self.analyze_fig.add_subplot(223)
        self.analyze_subplot3 = self.analyze_fig.add_subplot(224)
        self.analysis_plots[0].plot_obj = self.analyze_subplot0
        self.analysis_plots[1].plot_obj = self.analyze_subplot1
        self.analysis_plots[2].plot_obj = self.analyze_subplot2
        self.analysis_plots[3].plot_obj = self.analyze_subplot3
        
        self.analyze_canvas = tkagg.FigureCanvasTkAgg(self.analyze_fig, master=self.tab_analyze)
        self.analyze_widget = self.analyze_canvas.get_tk_widget()
        self.analyze_widget.grid(row=1,column=0,rowspan=1,columnspan=4, padx=(12,0))

        self.analyze_plotselector_frame = tk.ttk.Frame(master=self.tab_analyze)
        self.analyze_plotselector_frame.grid(row=2,rowspan=2,column=0,columnspan=4)
        
        self.analysisplot_topleft = tk.ttk.Radiobutton(self.analyze_plotselector_frame, variable=self.active_analysisplot_ID, value=0)
        self.analysisplot_topleft.grid(row=0,column=0)

        self.analysisplot_topleft_label = tk.ttk.Label(self.analyze_plotselector_frame, text="Use: Top Left")
        self.analysisplot_topleft_label.grid(row=0,column=1)
        
        self.analysisplot_topright = tk.ttk.Radiobutton(self.analyze_plotselector_frame, variable=self.active_analysisplot_ID, value=1)
        self.analysisplot_topright.grid(row=0,column=2)

        self.analysisplot_topright_label = tk.ttk.Label(self.analyze_plotselector_frame, text="Use: Top Right")
        self.analysisplot_topright_label.grid(row=0,column=3)
        
        self.analysisplot_bottomleft = tk.ttk.Radiobutton(self.analyze_plotselector_frame, variable=self.active_analysisplot_ID, value=2)
        self.analysisplot_bottomleft.grid(row=1,column=0)

        self.analysisplot_bottomleft_label = tk.ttk.Label(self.analyze_plotselector_frame, text="Use: Bottom Left")
        self.analysisplot_bottomleft_label.grid(row=1,column=1)
        
        self.analysisplot_bottomright = tk.ttk.Radiobutton(self.analyze_plotselector_frame, variable=self.active_analysisplot_ID, value=3)
        self.analysisplot_bottomright.grid(row=1,column=2)

        self.analysisplot_bottomright_label = tk.ttk.Label(self.analyze_plotselector_frame, text="Use: Bottom Right")
        self.analysisplot_bottomright_label.grid(row=1,column=3)
        
        self.analyze_toolbar_frame = tk.ttk.Frame(master=self.tab_analyze)
        self.analyze_toolbar_frame.grid(row=4,column=0,rowspan=3,columnspan=4)
        self.analyze_toolbar = tkagg.NavigationToolbar2Tk(self.analyze_canvas, self.analyze_toolbar_frame)
        self.analyze_toolbar.grid(row=0,column=0,columnspan=6)

        self.analyze_plot_button = tk.ttk.Button(self.analyze_toolbar_frame, text="Plot", command=partial(self.fetch_dataset))
        self.analyze_plot_button.grid(row=1,column=0)
        
        self.analyze_tstep_entry = tk.ttk.Entry(self.analyze_toolbar_frame, width=9)
        self.analyze_tstep_entry.grid(row=1,column=1)

        self.analyze_tstep_button = tk.ttk.Button(self.analyze_toolbar_frame, text="Step >>", command=partial(self.plot_tstep))
        self.analyze_tstep_button.grid(row=1,column=2)

        self.calculate_PL_button = tk.ttk.Button(self.analyze_toolbar_frame, text=">> Integrate <<", command=partial(self.do_Integrate))
        self.calculate_PL_button.grid(row=1,column=3)

        self.analyze_axis_button = tk.ttk.Button(self.analyze_toolbar_frame, text="Axis Settings", command=partial(self.do_change_axis_popup, from_integration=0))
        self.analyze_axis_button.grid(row=1,column=4)

        self.analyze_export_button = tk.ttk.Button(self.analyze_toolbar_frame, text="Export", command=partial(self.export_plot, from_integration=0))
        self.analyze_export_button.grid(row=1,column=5)

        self.analyze_IC_carry_button = tk.ttk.Button(self.analyze_toolbar_frame, text="Generate IC", command=partial(self.do_IC_carry_popup))
        self.analyze_IC_carry_button.grid(row=1,column=6)

        self.integration_fig = Figure(figsize=(9,5))
        self.integration_subplot = self.integration_fig.add_subplot(121)
        self.taudiff_subplot = self.integration_fig.add_subplot(122)
        self.integration_plots[0].plot_obj = self.integration_subplot
        self.integration_plots[1].plot_obj = self.taudiff_subplot
        
        self.integration_canvas = tkagg.FigureCanvasTkAgg(self.integration_fig, master=self.tab_analyze)
        self.integration_widget = self.integration_canvas.get_tk_widget()
        self.integration_widget.grid(row=1,column=5,rowspan=1,columnspan=1, padx=(20,0))

        self.integration_plotselector_frame = tk.ttk.Frame(master=self.tab_analyze)
        self.integration_plotselector_frame.grid(row=2,column=5)
        
        self.integrationplot_left = tk.ttk.Radiobutton(self.integration_plotselector_frame, variable=self.active_integrationplot_ID, value=0)
        self.integrationplot_left.grid(row=0,column=0)

        self.integrationplot_topleft_label = tk.ttk.Label(self.integration_plotselector_frame, text="Use: Integration")
        self.integrationplot_topleft_label.grid(row=0,column=1)
        
        self.integrationplot_topright = tk.ttk.Radiobutton(self.integration_plotselector_frame, variable=self.active_integrationplot_ID, value=1)
        self.integrationplot_topright.grid(row=0,column=2)

        self.integrationplot_topright_label = tk.ttk.Label(self.integration_plotselector_frame, text="Use: Tau_Diff")
        self.integrationplot_topright_label.grid(row=0,column=3)

        self.integration_toolbar_frame = tk.ttk.Frame(master=self.tab_analyze)
        self.integration_toolbar_frame.grid(row=3,column=5, rowspan=2,columnspan=1)
        self.integration_toolbar = tkagg.NavigationToolbar2Tk(self.integration_canvas, self.integration_toolbar_frame)
        self.integration_toolbar.grid(row=0,column=0,columnspan=5)

        self.integration_axis_button = tk.ttk.Button(self.integration_toolbar_frame, text="Axis Settings", command=partial(self.do_change_axis_popup, from_integration=1))
        self.integration_axis_button.grid(row=1,column=0)

        self.integration_export_button = tk.ttk.Button(self.integration_toolbar_frame, text="Export", command=partial(self.export_plot, from_integration=1))
        self.integration_export_button.grid(row=1,column=1)

        self.integration_bayesim_button = tk.ttk.Button(self.integration_toolbar_frame, text="Bayesim", command=partial(self.do_bayesim_popup))
        self.integration_bayesim_button.grid(row=1,column=2)

        self.analysis_status = tk.Text(self.tab_analyze, width=28,height=3)
        self.analysis_status.grid(row=5,rowspan=3,column=5,columnspan=1)
        self.analysis_status.configure(state="disabled")

        self.notebook.add(self.tab_analyze, text="Analyze")
        return

    def DEBUG(self):
        print(self.nanowire.DEBUG_print())
        return

    def update_system_summary(self):
        if self.sys_summary_popup_isopen:
            self.write(self.summary_textbox, self.nanowire.DEBUG_print())
            
        return
    ## Functions to create popups and manage
    
    def do_sys_summary_popup(self):
        if not self.sys_summary_popup_isopen: # Don't open more than one of this window at a time
            self.sys_summary_popup = tk.Toplevel(self.root)
            
            self.summary_textbox = tkscrolledtext.ScrolledText(self.sys_summary_popup, width=100,height=30)
            self.summary_textbox.grid(row=0,column=0,padx=(20,0), pady=(20,20))
            
            self.sys_summary_popup_isopen = True
            
            self.update_system_summary()
            
            self.sys_summary_popup.protocol("WM_DELETE_WINDOW", self.on_sys_summary_popup_close)
            return
        
    def on_sys_summary_popup_close(self):
        try:
            self.sys_summary_popup.destroy()
            self.sys_summary_popup_isopen = False
        except FloatingPointError:
            print("Error #2022: Failed to close shortcut popup.")
        return
        
    def do_sys_param_shortcut_popup(self):
        # V2: An overhaul of the old method for inputting (spatially constant) parameters
        if not self.sys_param_shortcut_popup_isopen:
            try:
                self.set_init_x()
    
            except ValueError:
                self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
                return
    
            except Exception as oops:
                self.write(self.ICtab_status, oops)
                return
        
            self.sys_param_shortcut_popup = tk.Toplevel(self.root)
            
            self.sys_param_shortcut_title_label = tk.ttk.Label(self.sys_param_shortcut_popup, text="Parameter Short-cut Tool", style="Header.TLabel")
            self.sys_param_shortcut_title_label.grid(row=0,column=0)
            
            self.sys_param_instruction = tk.Message(self.sys_param_shortcut_popup, text="Are the values of certain parameters constant across the system? " +
                                                 "Enter those values here and press \"Continue\" to apply them on all space grid points.", width=300)
            self.sys_param_instruction.grid(row=0,column=1)
            
            self.sys_param_list_frame = tk.ttk.Frame(self.sys_param_shortcut_popup)
            self.sys_param_list_frame.grid(row=1,column=0,columnspan=2)
            
            self.sys_param_entryboxes_dict = {}
            self.sys_param_labels_dict = {}
            row_count = 0
            col_count = 0
            max_per_col = 6
            for param in self.nanowire.param_dict:
                self.sys_param_labels_dict[param] = tk.ttk.Label(self.sys_param_list_frame, text="{} {}".format(param, self.nanowire.param_dict[param].units))
                self.sys_param_labels_dict[param].grid(row=row_count, column=col_count)
                self.sys_param_entryboxes_dict[param] = tk.ttk.Entry(self.sys_param_list_frame, width=9)
                self.sys_param_entryboxes_dict[param].grid(row=row_count, column=col_count + 1)
                row_count += 1
                if row_count == max_per_col:
                    row_count = 0
                    col_count += 2
                    
            self.shortcut_continue_button = tk.Button(self.sys_param_shortcut_popup, text="Continue", command=partial(self.on_sys_param_shortcut_popup_close, True))
            self.shortcut_continue_button.grid(row=2,column=1)
                    
            self.sys_param_shortcut_popup.protocol("WM_DELETE_WINDOW", self.on_sys_param_shortcut_popup_close)
            
            ## Temporarily disable the main window while this popup is active
            self.sys_param_shortcut_popup.grab_set()
            
            self.sys_param_shortcut_popup_isopen = True
                    
        else:
            print("Error #2020: Opened more than one sys param shortcut popup at a time")
            
    def on_sys_param_shortcut_popup_close(self, continue_=False):
        try:

            if continue_:
                changed_params = []
                for param in self.nanowire.param_dict:
                    val = self.sys_param_entryboxes_dict[param].get()
                    if val == "": continue
                    else:
                        try:
                            val = float(val)
                            
                        except:
                            continue
                    
                    self.HIC_listbox_currentparam = param
                    self.deleteall_HIC()
                    self.nanowire.param_dict[param].value = val
                    changed_params.append(param)
                    
                if changed_params.__len__() > 0:
                    self.update_IC_plot(plot_ID="recent")
                    self.write(self.ICtab_status, "Updated: {}".format(changed_params))
                    
                else:
                    self.write(self.ICtab_status, "")
                    
            self.sys_param_shortcut_popup.destroy()
            self.sys_param_shortcut_popup_isopen = False
        except FloatingPointError:
            print("Error #2021: Failed to close shortcut popup.")
        
        return

    def do_batch_popup(self):
        try:
            self.set_init_x()
        except:
            self.write(self.ICtab_status, "Error: missing space grid")
            return

        if not self.batch_popup_isopen: 
            max_batchable_params = 4
            self.batch_param = tk.StringVar()

            self.batch_popup = tk.Toplevel(self.root)
            
            self.batch_title_label = tk.ttk.Label(self.batch_popup, text="Batch IC Tool", style="Header.TLabel")
            self.batch_title_label.grid(row=0,column=0)
            
            self.batch_instruction1 = tk.Message(self.batch_popup, text="This Batch Tool allows you to generate many copies of the currently-loaded IC, varying up to {} parameters between all of them.".format(max_batchable_params), width=300)
            self.batch_instruction1.grid(row=1,column=0)

            self.batch_instruction2 = tk.Message(self.batch_popup, text="All copies will be stored in a new folder with the name you enter into the appropriate box.", width=300)
            self.batch_instruction2.grid(row=2,column=0)

            self.batch_instruction3 = tk.Message(self.batch_popup, text="For best results, load a complete IC file or fill in values for all params before using this tool.", width=300)
            self.batch_instruction3.grid(row=3,column=0)

            self.batch_param_label = tk.ttk.Label(self.batch_popup, text="Select Batch Parameter:")
            self.batch_param_label.grid(row=0,column=1)
            
            self.batch_entry_frame = tk.ttk.Frame(self.batch_popup)
            self.batch_entry_frame.grid(row=1,column=1,columnspan=3, rowspan=3)
           

            # Contextually-dependent options for batchable params
            self.batchables_array = []
            batchable_params = [param for param in self.nanowire.param_dict if not (self.using_AIC and (param == "init_deltaN" or param == "init_deltaP"))]
            
            if self.using_AIC:
                
                self.AIC_instruction1 = tk.Message(self.batch_popup, text="Additional options for generating init_deltaN and init_deltaP batches " +
                                                  "are available when using the Analytical Initial Condition tool", width=300)
                self.AIC_instruction1.grid(row=4,column=0)
                
                self.AIC_instruction2 = tk.Message(self.batch_popup, text="Please note that TEDs will use the values and settings on the A.I.C. tool's tab " +
                                                   "to complete the batches when one or more of these options are selected.", width=300)
                self.AIC_instruction2.grid(row=5,column=0)
                
                # Boolean logic is fun
                # The main idea is to hide certain parameters based on which options were used to construct the AIC
                AIC_params = [key for key in self.analytical_entryboxes_dict.keys() if not (
                            (self.check_calculate_init_material_expfactor.get() and (key == "AIC_expfactor")) or
                            (not self.check_calculate_init_material_expfactor.get() and (key == "A0" or key == "Eg")) or
                            (self.AIC_gen_power_mode.get() == "power-spot" and (key == "Power_Density" or key == "Max_Gen" or key == "Total_Gen")) or
                            (self.AIC_gen_power_mode.get() == "density" and (key == "Power" or key == "Spotsize" or key == "Max_Gen" or key == "Total_Gen")) or
                            (self.AIC_gen_power_mode.get() == "max-gen" and (key == "Power" or key == "Spotsize" or key == "Power_Density" or key == "Total_Gen")) or
                            (self.AIC_gen_power_mode.get() == "total-gen" and (key == "Power_Density" or key == "Power" or key == "Spotsize" or key == "Max_Gen"))
                            )]

            for i in range(max_batchable_params):
                batch_param_name = tk.StringVar()
                if self.using_AIC:
                    optionmenu = tk.ttk.OptionMenu(self.batch_entry_frame, batch_param_name, "", "", *batchable_params, *AIC_params)
                else:
                    optionmenu = tk.ttk.OptionMenu(self.batch_entry_frame, batch_param_name, "", "", *batchable_params)
                
                optionmenu.grid(row=i,column=0,padx=(20,20))
                batch_param_entry = tk.ttk.Entry(self.batch_entry_frame, width=80)
                batch_param_entry.grid(row=i,column=1,columnspan=2)
                
                if i == 0: self.enter(batch_param_entry, "Enter a list of space-separated values for the selected Batch Parameter")
                
                self.batchables_array.append(Batchable(optionmenu, batch_param_entry, batch_param_name))
                    
            self.batch_status = tk.Text(self.batch_popup, width=30,height=3)
            self.batch_status.grid(row=6,column=0)
            self.batch_status.configure(state='disabled')

            self.batch_name_entry = tk.ttk.Entry(self.batch_popup, width=24)
            self.enter(self.batch_name_entry, "Enter name for batch folder")
            self.batch_name_entry.grid(row=6,column=1)

            self.create_batch_button = tk.ttk.Button(self.batch_popup, text="Create Batch", command=self.create_batch_init)
            self.create_batch_button.grid(row=6,column=2)

            self.batch_popup.protocol("WM_DELETE_WINDOW", self.on_batch_popup_close)
            self.batch_popup.grab_set()
            self.batch_popup_isopen = True

        else:
            print("Error #102: Opened more than one batch popup at a time")
        return

    def on_batch_popup_close(self):
        try:
            self.batch_popup.destroy()
            print("Batch popup closed")
            self.batch_popup_isopen = False
        except:
            print("Error #103: Failed to close batch popup.")

        return

    def do_resetIC_popup(self):

        if not self.resetIC_popup_isopen:

            self.resetIC_popup = tk.Toplevel(self.root)

            self.resetIC_title_label1 = tk.ttk.Label(self.resetIC_popup, text="Which Parameters should be cleared?", style="Header.TLabel")
            self.resetIC_title_label1.grid(row=0,column=0,columnspan=2)

            self.resetIC_checkbutton_frame = tk.ttk.Frame(self.resetIC_popup)
            self.resetIC_checkbutton_frame.grid(row=1,column=0,columnspan=2)

            # Let's try some procedurally generated checkbuttons: one created automatically per nanowire parameter
            self.resetIC_checkparams = {}
            self.resetIC_checkbuttons = {}
            cb_row = 0
            cb_col = 0
            cb_per_col = 3
            for param in self.nanowire.param_dict:
                self.resetIC_checkparams[param] = tk.IntVar()

                self.resetIC_checkbuttons[param] = tk.ttk.Checkbutton(self.resetIC_checkbutton_frame, text=param, variable=self.resetIC_checkparams[param], onvalue=1, offvalue=0)

            for cb in self.resetIC_checkbuttons:
                self.resetIC_checkbuttons[cb].grid(row=cb_row,column=cb_col, pady=(6,6))
                cb_row += 1
                if cb_row == cb_per_col:
                    cb_row = 0
                    cb_col += 1

            
            self.hline10_separator = tk.ttk.Separator(self.resetIC_popup, orient="horizontal", style="Grey Bar.TSeparator")
            self.hline10_separator.grid(row=2,column=0,columnspan=2, pady=(10,10), sticky="ew")

            self.resetIC_check_clearall = tk.IntVar()
            self.resetIC_clearall_checkbutton = tk.Checkbutton(self.resetIC_popup, text="Clear All", variable=self.resetIC_check_clearall, onvalue=1, offvalue=0)
            self.resetIC_clearall_checkbutton.grid(row=3,column=0)

            self.resetIC_continue_button = tk.Button(self.resetIC_popup, text="Continue", command=partial(self.on_resetIC_popup_close, True))
            self.resetIC_continue_button.grid(row=3,column=1)

            self.resetIC_popup.protocol("WM_DELETE_WINDOW", self.on_resetIC_popup_close)
            self.resetIC_popup.grab_set()
            self.resetIC_popup_isopen = True
            return

        else:
            print("Error #700: Opened more than one resetIC popup at a time")

        return

    def on_resetIC_popup_close(self, continue_=False):
        try:
            self.resetIC_selected_params = []
            self.resetIC_do_clearall = False
            if continue_:
                self.resetIC_do_clearall = self.resetIC_check_clearall.get()
                if self.resetIC_do_clearall:
                    self.resetIC_selected_params = list(self.resetIC_checkparams.keys())
                else:
                    self.resetIC_selected_params = [param for param in self.resetIC_checkparams if self.resetIC_checkparams[param].get()]

            self.resetIC_popup.destroy()
            print("resetIC popup closed")
            self.resetIC_popup_isopen = False

        except:
            print("Error #601: Failed to close Bayesim popup")
        return

    def do_plotter_popup(self, plot_ID):
        if not self.plotter_popup_isopen:

            self.plotter_popup = tk.Toplevel(self.root)

            self.plotter_title_label = tk.ttk.Label(self.plotter_popup, text="Select a data type", style="Header.TLabel")
            self.plotter_title_label.grid(row=0,column=0,columnspan=2)

            self.var_select_menu = tk.OptionMenu(self.plotter_popup, self.data_var, "N", "P", "E_field", "RR", "NRR", "PL")
            self.var_select_menu.grid(row=1,column=0)

            self.autointegrate_checkbutton = tk.Checkbutton(self.plotter_popup, text="Auto integrate all space and time steps?", variable=self.check_autointegrate, onvalue=1, offvalue=0)
            self.autointegrate_checkbutton.grid(row=1,column=1)
            
            self.plotter_continue_button = tk.Button(self.plotter_popup, text="Continue", command=partial(self.on_plotter_popup_close, plot_ID, continue_=True))
            self.plotter_continue_button.grid(row=2,column=1)

            self.data_listbox = tk.Listbox(self.plotter_popup, width=20, height=20, selectmode="extended")
            self.data_listbox.grid(row=2,rowspan=13,column=0)
            self.data_listbox.delete(0,tk.END)
            self.data_list = [file for file in os.listdir(self.default_dirs["Data"]) if not file.endswith(".txt")]
            self.data_listbox.insert(0,*(self.data_list))

            self.plotter_status = tk.Text(self.plotter_popup, width=24,height=2)
            self.plotter_status.grid(row=3,rowspan=2,column=1)
            self.plotter_status.configure(state="disabled")

            self.plotter_popup.protocol("WM_DELETE_WINDOW", partial(self.on_plotter_popup_close, plot_ID, continue_=False))
            self.plotter_popup.grab_set()
            self.plotter_popup_isopen = True

        else:
            print("Error #501: Opened more than one plotter popup at a time")
        return

    def on_plotter_popup_close(self, plot_ID, continue_=False):
        try:
			# There are two ways for a popup to close: by the user pressing "Continue" or the user cancelling or pressing "X"
			# We only interpret the input on the popup if the user wants to continue
            if continue_:
                if self.data_var.get() == "": raise ValueError
                self.analysis_plots[plot_ID].data_filenames = []
                # This year for Christmas, I want Santa to implement tk.filedialog.askdirectories() so we can select multiple directories like we can do with files
                dir_names = [self.data_list[i] for i in self.data_listbox.curselection()]
                for next_dir in dir_names:
                    self.analysis_plots[plot_ID].data_filenames.append(next_dir)

                #self.analysis_plots[plot_ID].remove_duplicate_filenames()

            self.plotter_popup.destroy()
            print("Plotter popup closed")
            self.plotter_popup_isopen = False

        except ValueError:
            self.write(self.plotter_status, "Select a variable from the menu")
        except:
            print("Error #502: Failed to close plotter popup.")

        return

    def do_integration_popup(self):
        if not self.integration_popup_isopen:
            self.integration_popup = tk.Toplevel(self.root)

            self.integration_title_label = tk.ttk.Label(self.integration_popup, text="Select which time steps to integrate over", style="Header.TLabel")
            self.integration_title_label.grid(row=1,column=0,columnspan=3)

            self.overtime = tk.ttk.Radiobutton(self.integration_popup, variable=self.fetch_PLmode, value='All time steps')
            self.overtime.grid(row=2,column=0)

            self.overtime_label = tk.Label(self.integration_popup, text="All time steps")
            self.overtime_label.grid(row=2,column=1)

            self.currentTS = tk.ttk.Radiobutton(self.integration_popup, variable=self.fetch_PLmode, value='Current time step')
            self.currentTS.grid(row=3,column=0)

            self.currentTS_label = tk.Label(self.integration_popup, text="Current time step")
            self.currentTS_label.grid(row=3,column=1)

            self.integration_continue_button =  tk.Button(self.integration_popup, text="Continue", command=partial(self.on_integration_popup_close, continue_=True))
            self.integration_continue_button.grid(row=4,column=0,columnspan=3)

            self.integration_popup.protocol("WM_DELETE_WINDOW", partial(self.on_integration_popup_close, continue_=False))
            self.integration_popup.grab_set()
            self.integration_popup_isopen = True
            
        else:
            print("Error #420: Opened more than one integration popup at a time")
        return

    def on_integration_popup_close(self, continue_=False):
        try:
            if continue_:
                self.PL_mode = self.fetch_PLmode.get()
            else:
                self.PL_mode = ""

            self.integration_popup.destroy()
            print("Integration popup closed")
            self.integration_popup_isopen = False
        except:
            print("Error #421: Failed to close PLmode popup.")

        return

    def do_integration_getbounds_popup(self):
        if not self.integration_getbounds_popup_isopen:
            # Reset integration bounds
            # self.integration_lbound = ""
            # self.integration_ubound = ""

            self.integration_getbounds_popup = tk.Toplevel(self.root)

            self.single_intg = tk.ttk.Radiobutton(self.integration_getbounds_popup, variable=self.fetch_intg_mode, value='single')
            self.single_intg.grid(row=0,column=0, rowspan=3)

            self.single_intg_label = tk.ttk.Label(self.integration_getbounds_popup, text="Single integral", style="Header.TLabel")
            self.single_intg_label.grid(row=0,column=1, rowspan=3, padx=(0,20))

            self.integration_getbounds_title_label = tk.Label(self.integration_getbounds_popup, text="Enter bounds of integration [nm]")
            self.integration_getbounds_title_label.grid(row=0,column=2,columnspan=4)

            self.lower = tk.Label(self.integration_getbounds_popup, text="Lower bound: x=")
            self.lower.grid(row=1,column=2)

            self.integration_lbound_entry = tk.Entry(self.integration_getbounds_popup, width=9)
            self.integration_lbound_entry.grid(row=2,column=2)

            self.upper = tk.Label(self.integration_getbounds_popup, text="Upper bound: x=")
            self.upper.grid(row=1,column=5)

            self.integration_ubound_entry = tk.Entry(self.integration_getbounds_popup, width=9)
            self.integration_ubound_entry.grid(row=2,column=5)

            self.hline8_separator = tk.ttk.Separator(self.integration_getbounds_popup, orient="horizontal", style="Grey Bar.TSeparator")
            self.hline8_separator.grid(row=3,column=0,columnspan=30, pady=(10,10), sticky="ew")

            self.multi_intg = tk.ttk.Radiobutton(self.integration_getbounds_popup, variable=self.fetch_intg_mode, value='multiple')
            self.multi_intg.grid(row=4,column=0, rowspan=3)

            self.multi_intg_label = tk.ttk.Label(self.integration_getbounds_popup, text="Multiple integrals", style="Header.TLabel")
            self.multi_intg_label.grid(row=4,column=1, rowspan=3, padx=(0,20))

            self.integration_center_label = tk.Label(self.integration_getbounds_popup, text="Enter space-separated e.g. (100 200 300...) Centers [nm]: ")
            self.integration_center_label.grid(row=5,column=2)

            self.integration_center_entry = tk.Entry(self.integration_getbounds_popup, width=30)
            self.integration_center_entry.grid(row=5,column=3,columnspan=3)

            self.integration_width_label = tk.Label(self.integration_getbounds_popup, text="Width [nm]: +/- ")
            self.integration_width_label.grid(row=6,column=2)

            self.integration_width_entry = tk.Entry(self.integration_getbounds_popup, width=9)
            self.integration_width_entry.grid(row=6,column=3)

            self.hline9_separator = tk.ttk.Separator(self.integration_getbounds_popup, orient="horizontal", style="Grey Bar.TSeparator")
            self.hline9_separator.grid(row=7,column=0,columnspan=30, pady=(10,10), sticky="ew")

            self.integration_getbounds_continue_button = tk.Button(self.integration_getbounds_popup, text="Continue", command=partial(self.on_integration_getbounds_popup_close, continue_=True))
            self.integration_getbounds_continue_button.grid(row=8,column=5)

            self.integration_getbounds_status = tk.Text(self.integration_getbounds_popup, width=24,height=2)
            self.integration_getbounds_status.grid(row=8,rowspan=2,column=0,columnspan=5)
            self.integration_getbounds_status.configure(state="disabled")

            self.integration_getbounds_popup.protocol("WM_DELETE_WINDOW", partial(self.on_integration_getbounds_popup_close, continue_=False))
            self.integration_getbounds_popup.grab_set()
            self.integration_getbounds_popup_isopen = True
        else:
            print("Error #422: Opened more than one integration getbounds popup at a time")
        return

    def on_integration_getbounds_popup_close(self, continue_=False):
        # Read in the pairs of integration bounds as-is
        # Checking if they make sense is do_Integrate()'s job
        try:
            if continue_:
                self.integration_bounds = []
                if self.fetch_intg_mode.get() == "single":
                    print("Single integral")
                    lbound = float(self.integration_lbound_entry.get())
                    ubound = float(self.integration_ubound_entry.get())
                    if (lbound > ubound):
                        raise KeyError("Error: upper bound too small")

                    self.integration_bounds.append([lbound, ubound])
                    

                elif self.fetch_intg_mode.get() == "multiple":
                    print("Multiple integrals")
                    if self.integration_center_entry.get() == "Aboma":
                        centers = [0,2200,3400,5200,6400,7200,8600,10000]

                    else:
                        centers = list(set(extract_values(self.integration_center_entry.get(), ' ')))

                    width = float(self.integration_width_entry.get())

                    if width < 0: raise KeyError("Error: width must be non-negative")

                    for center in centers:
                        self.integration_bounds.append([center - width, center + width])

                else:
                    raise KeyError("Select \"Single\" or \"Multiple\"")

                print("Over: {}".format(self.integration_bounds))

            else:
                self.write(self.analysis_status, "Integration cancelled")

            self.integration_getbounds_popup.destroy()
            print("PL getbounds popup closed")
            self.integration_getbounds_popup_isopen = False

        except (OSError, KeyError) as uh_oh:
            self.write(self.integration_getbounds_status, uh_oh)

        except:
            self.write(self.integration_getbounds_status, "Error: missing or invalid paramters")

        return

    def do_PL_xaxis_popup(self):
        if not self.PL_xaxis_popup_isopen:
            self.xaxis_param = ""
            self.xaxis_selection = tk.StringVar()
            self.PL_xaxis_popup = tk.Toplevel(self.root)

            self.PL_xaxis_title_label = tk.ttk.Label(self.PL_xaxis_popup, text="Select parameter for x axis", style="Header.TLabel")
            self.PL_xaxis_title_label.grid(row=0,column=0,columnspan=3)

            self.xaxis_param_menu = tk.OptionMenu(self.PL_xaxis_popup, self.xaxis_selection, *[param for param in self.nanowire.param_dict])
            self.xaxis_param_menu.grid(row=1,column=1)

            self.PL_xaxis_continue_button = tk.Button(self.PL_xaxis_popup, text="Continue", command=partial(self.on_PL_xaxis_popup_close, continue_=True))
            self.PL_xaxis_continue_button.grid(row=1,column=2)

            self.PL_xaxis_status = tk.Text(self.PL_xaxis_popup, width=24,height=2)
            self.PL_xaxis_status.grid(row=2,rowspan=2,column=0,columnspan=3)
            self.PL_xaxis_status.configure(state="disabled")

            self.PL_xaxis_popup.protocol("WM_DELETE_WINDOW", partial(self.on_PL_xaxis_popup_close, continue_=False))
            self.PL_xaxis_popup.grab_set()
            self.PL_xaxis_popup_isopen = True
        else:
            print("Error #424: Opened more than one PL xaxis popup at a time")
        return

    def on_PL_xaxis_popup_close(self, continue_=False):
        try:
            if continue_:
                self.xaxis_param = self.xaxis_selection.get()
                if self.xaxis_param == "":
                    self.write(self.PL_xaxis_status, "Select a parameter")
                    return
            self.PL_xaxis_popup.destroy()
            print("PL xaxis popup closed")
            self.PL_xaxis_popup_isopen = False
        except:
            print("Error #425: Failed to close PL xaxis popup.")

        return
    
    def do_change_axis_popup(self, from_integration):
        # Don't open if no data plotted
        if from_integration:
            plot_ID = self.active_integrationplot_ID.get()
            if self.integration_plots[plot_ID].datagroup.size() == 0: return

        else:
            plot_ID = self.active_analysisplot_ID.get()
            if self.analysis_plots[plot_ID].datagroup.size() == 0: return

        if not self.change_axis_popup_isopen:
            self.change_axis_popup = tk.Toplevel(self.root)

            self.change_axis_title_label = tk.ttk.Label(self.change_axis_popup, text="Select axis settings", style="Header.TLabel")
            self.change_axis_title_label.grid(row=0,column=0,columnspan=2)

            self.xframe = tk.Frame(master=self.change_axis_popup)
            self.xframe.grid(row=1,column=0,padx=(0,20),pady=(20,0))

            self.xheader = tk.Label(self.xframe, text="X Axis")
            self.xheader.grid(row=0,column=0,columnspan=2)

            self.xlin = tk.ttk.Radiobutton(self.xframe, variable=self.xaxis_type, value='linear')
            self.xlin.grid(row=1,column=0)

            self.xlin_label = tk.Label(self.xframe, text="Linear")
            self.xlin_label.grid(row=1,column=1)

            self.xlog = tk.ttk.Radiobutton(self.xframe, variable=self.xaxis_type, value='log')
            self.xlog.grid(row=2,column=0)

            self.xlog_label = tk.Label(self.xframe, text="Log")
            self.xlog_label.grid(row=2,column=1)

            self.xlbound_label = tk.Label(self.xframe, text="Lower")
            self.xlbound_label.grid(row=3,column=0)

            self.xlbound = tk.Entry(self.xframe, width=9)
            self.xlbound.grid(row=3,column=1)

            self.xubound_label = tk.Label(self.xframe, text="Upper")
            self.xubound_label.grid(row=4,column=0)

            self.xubound = tk.Entry(self.xframe, width=9)
            self.xubound.grid(row=4,column=1)

            self.yframe = tk.Frame(master=self.change_axis_popup)
            self.yframe.grid(row=1,column=1,padx=(0,20),pady=(20,0))

            self.yheader = tk.Label(self.yframe, text="Y Axis")
            self.yheader.grid(row=0,column=0,columnspan=2)

            self.ylin = tk.ttk.Radiobutton(self.yframe, variable=self.yaxis_type, value='linear')
            self.ylin.grid(row=1,column=0)

            self.ylin_label = tk.Label(self.yframe, text="Linear")
            self.ylin_label.grid(row=1,column=1)

            self.ylog = tk.ttk.Radiobutton(self.yframe, variable=self.yaxis_type, value='log')
            self.ylog.grid(row=2,column=0)

            self.ylog_label = tk.Label(self.yframe, text="Log")
            self.ylog_label.grid(row=2,column=1)

            self.ylbound_label = tk.Label(self.yframe, text="Lower")
            self.ylbound_label.grid(row=3,column=0)

            self.ylbound = tk.Entry(self.yframe, width=9)
            self.ylbound.grid(row=3,column=1)

            self.yubound_label = tk.Label(self.yframe, text="Upper")
            self.yubound_label.grid(row=4,column=0)

            self.yubound = tk.Entry(self.yframe, width=9)
            self.yubound.grid(row=4,column=1)

            self.toggle_legend_checkbutton = tk.Checkbutton(self.change_axis_popup, text="Display legend?", variable=self.check_display_legend, onvalue=1, offvalue=0)
            self.toggle_legend_checkbutton.grid(row=2,column=0,columnspan=2)

            self.change_axis_continue_button = tk.Button(self.change_axis_popup, text="Continue", command=partial(self.on_change_axis_popup_close, from_integration, continue_=True))
            self.change_axis_continue_button.grid(row=3,column=0,columnspan=2)

            self.change_axis_status = tk.Text(self.change_axis_popup, width=24,height=2)
            self.change_axis_status.grid(row=4,rowspan=2,column=0,columnspan=2)
            self.change_axis_status.configure(state="disabled")

            # Set the default values in the entry boxes to be the current options of the plot (in case the user only wants to make a few changes)
            if not (from_integration):
                active_plot = self.analysis_plots[plot_ID]

            else:
                active_plot = self.integration_plots[plot_ID]

            self.enter(self.xlbound, active_plot.xlim[0])
            self.enter(self.xubound, active_plot.xlim[1])
            self.enter(self.ylbound, active_plot.ylim[0])
            self.enter(self.yubound, active_plot.ylim[1])
            self.xaxis_type.set(active_plot.xaxis_type)
            self.yaxis_type.set(active_plot.yaxis_type)
            self.check_display_legend.set(active_plot.display_legend)

            self.change_axis_popup.protocol("WM_DELETE_WINDOW", partial(self.on_change_axis_popup_close, from_integration, continue_=False))
            self.change_axis_popup.grab_set()
            self.change_axis_popup_isopen = True
        else:
            print("Error #440: Opened more than one PL change axis popup at a time")
        return

    def on_change_axis_popup_close(self, from_integration, continue_=False):
        try:
            if continue_:
                if not self.xaxis_type or not self.yaxis_type: raise ValueError("Error: invalid axis type")
                if self.xlbound.get() == "" or self.xubound.get() == "" or self.ylbound.get() == "" or self.yubound.get() == "": raise ValueError("Error: missing bounds")
                bounds = [float(self.xlbound.get()), float(self.xubound.get()), float(self.ylbound.get()), float(self.yubound.get())]
            
                if not (from_integration):
                    plot_ID = self.active_analysisplot_ID.get()
                    plot = self.analysis_plots[plot_ID].plot_obj
                    
                else:
                    plot_ID = self.active_integrationplot_ID.get()
                    plot = self.integration_plots[plot_ID].plot_obj

                # Set plot axis params and save in corresponding plot state object, if the selected plot has such an object
                plot.set_yscale(self.yaxis_type.get())
                plot.set_xscale(self.xaxis_type.get())

                plot.set_ylim(bounds[2], bounds[3])
                plot.set_xlim(bounds[0], bounds[1])

                if self.check_display_legend.get():
                    plot.legend().set_draggable(True)
                    
                else:
                    plot.legend('', frameon=False)

                if not (from_integration):
                    self.analyze_fig.tight_layout()
                    self.analyze_fig.canvas.draw()
                    
                else:
                    self.integration_fig.tight_layout()
                    self.integration_fig.canvas.draw()

                # Save these params to pre-populate the popup the next time it's opened
                if not (from_integration):
                    self.analysis_plots[plot_ID].yaxis_type = self.yaxis_type.get()
                    self.analysis_plots[plot_ID].xaxis_type = self.xaxis_type.get()
                    self.analysis_plots[plot_ID].ylim = (bounds[2], bounds[3])
                    self.analysis_plots[plot_ID].xlim = (bounds[0], bounds[1])
                    self.analysis_plots[plot_ID].display_legend = self.check_display_legend.get()
                else:
                    self.integration_plots[plot_ID].yaxis_type = self.yaxis_type.get()
                    self.integration_plots[plot_ID].xaxis_type = self.xaxis_type.get()
                    self.integration_plots[plot_ID].ylim = (bounds[2], bounds[3])
                    self.integration_plots[plot_ID].xlim = (bounds[0], bounds[1])
                    self.integration_plots[plot_ID].display_legend = self.check_display_legend.get()

            self.change_axis_popup.destroy()
            print("PL change axis popup closed")
            self.change_axis_popup_isopen = False

        except ValueError as oops:
            self.write(self.change_axis_status, oops)
            return
        except:
            print("Error #441: Failed to close PL xaxis popup.")

        return

    def do_IC_carry_popup(self):
        plot_ID = self.active_analysisplot_ID.get()
        # Don't open if no data plotted
        if self.analysis_plots[plot_ID].datagroup.size() == 0: return

        if not self.IC_carry_popup_isopen:
            self.IC_carry_popup = tk.Toplevel(self.root)

            self.IC_carry_title_label = tk.ttk.Label(self.IC_carry_popup, text="Select data to include in new IC", style="Header.TLabel")
            self.IC_carry_title_label.grid(row=0,column=0,columnspan=2)

            self.include_N_checkbutton = tk.Checkbutton(self.IC_carry_popup, text="N", variable=self.carry_include_N)
            self.include_N_checkbutton.grid(row=1,column=0)

            self.include_P_checkbutton = tk.Checkbutton(self.IC_carry_popup, text="P", variable=self.carry_include_P)
            self.include_P_checkbutton.grid(row=2,column=0)
            
            self.include_E_checkbutton = tk.Checkbutton(self.IC_carry_popup, text="E_field", variable=self.carry_include_E_field)
            self.include_E_checkbutton.grid(row=3,column=0)

            self.carry_IC_listbox = tk.Listbox(self.IC_carry_popup, width=30,height=10, selectmode='extended')
            self.carry_IC_listbox.grid(row=4,column=0,columnspan=2)
            for key in self.analysis_plots[plot_ID].datagroup.datasets:
                self.carry_IC_listbox.insert(tk.END, key)

            self.IC_carry_continue_button = tk.Button(self.IC_carry_popup, text="Continue", command=partial(self.on_IC_carry_popup_close, continue_=True))
            self.IC_carry_continue_button.grid(row=5,column=0,columnspan=2)

            self.IC_carry_popup.protocol("WM_DELETE_WINDOW", partial(self.on_IC_carry_popup_close, continue_=False))
            self.IC_carry_popup.grab_set()
            self.IC_carry_popup_isopen = True
        else:
            print("Error #510: Opened more than one PL change axis popup at a time")
        return

    def on_IC_carry_popup_close(self, continue_=False):
        try:
            if continue_:
                plot_ID = self.active_analysisplot_ID.get()
                active_sets = self.analysis_plots[plot_ID].datagroup.datasets
                pile = [self.carry_IC_listbox.get(i) for i in self.carry_IC_listbox.curselection()]
                for key in pile:
                    new_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["Initial"], title="Save IC text file for {}".format(key), filetypes=[("Text files","*.txt")])
                    if new_filename == "": continue

                    if new_filename.endswith(".txt"): new_filename = new_filename[:-4]
                    
                    param_dict_copy = dict(active_sets[key].params_dict)

                    node_x = active_sets[key].node_x
                    edge_x = active_sets[key].edge_x
                    param_dict_copy["init_deltaN"] = self.read_N(active_sets[key].filename, active_sets[key].show_index) - param_dict_copy["N0"] if self.carry_include_N.get() else np.zeros(node_x.__len__())
                    param_dict_copy["init_deltaP"] = self.read_P(active_sets[key].filename, active_sets[key].show_index) - param_dict_copy["P0"] if self.carry_include_P.get() else np.zeros(node_x.__len__())
                    param_dict_copy["init_E_field"] = self.read_E_field(active_sets[key].filename, active_sets[key].show_index) if self.carry_include_E_field.get() else np.zeros(edge_x.__len__())

                    with open(new_filename + ".txt", "w+") as ofstream:
                        ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
                        ofstream.write("$ Space Grid:\n")
                        ofstream.write("Total_length: {}\n".format(active_sets[key].params_dict["Total_length"]))
                        ofstream.write("Node_width: {}\n".format(active_sets[key].params_dict["Node_width"]))
                        ofstream.write("$ System Parameters:\n")
                        for param in param_dict_copy:
                            if not (param == "Total_length" or param == "Node_width" or param == "Total-Time" or param == "dt" or param == "steady_state_exc" or param in self.nanowire.flags_dict): 
                                param_values = param_dict_copy[param] * self.convert_out_dict[param]
                                if isinstance(param_values, np.ndarray):
                                    ofstream.write("{}: {:.8e}".format(param, param_values[0]))
                                    for value in param_values[1:]:
                                        ofstream.write("\t{:.8e}".format(value))
                                        
                                    ofstream.write('\n')
                                else:
                                    ofstream.write("{}: {}\n".format(param, param_values))

                        ofstream.write("$ System Flags:\n")
                        
                        for param in param_dict_copy:
                            if param in self.nanowire.flags_dict: 
                                ofstream.write("{}: {}\n".format(param, param_dict_copy[param]))

                self.write(self.analysis_status, "IC file generated")

            self.IC_carry_popup.destroy()
            print("IC carry popup closed")
            self.IC_carry_popup_isopen = False

        except OSError as oops:
            self.write(self.analysis_status, "IC file not created")
            
        # except:
        #     print("Error #511: Failed to close IC carry popup.")

        return

    def do_bayesim_popup(self):
        if self.integration_plots[0].datagroup.size() == 0: return

        if not self.bayesim_popup_isopen:

            self.bay_popup = tk.Toplevel(self.root)

            self.bay_title_label1 = tk.ttk.Label(self.bay_popup, text="Bayesim Tool", style="Header.TLabel")
            self.bay_title_label1.grid(row=0,column=0)

            self.bay_text1 = tk.Message(self.bay_popup, text=
                                        "Select \"Observation\" to save each curve as an experimentally observed data set or " +
                                        "(WIP) Model to combine all curves into a single model set.", width=320)
            self.bay_text1.grid(row=1,column=0)

            self.bay_text2 = tk.Message(self.bay_popup, text=
                                        "\"Observation\" data can be used to test Bayesim setups.", width=320)
            self.bay_text2.grid(row=2,column=0)

            self.bay_text3 = tk.Message(self.bay_popup, text=
                                        "Select system parameters to be included in the model.", width=320)
            self.bay_text3.grid(row=5,column=0)

            self.bay_line_separator = tk.ttk.Separator(self.bay_popup, orient="vertical", style="Grey Bar.TSeparator")
            self.bay_line_separator.grid(row=0,rowspan=30,column=1,padx=(6,6),sticky="ns")

            self.bay_title_label2 = tk.ttk.Label(self.bay_popup, text="\"Observation\" or Model?", style="Header.TLabel")
            self.bay_title_label2.grid(row=0,column=2,columnspan=4)

            self.bay_obs_mode = tk.ttk.Radiobutton(self.bay_popup, variable=self.bay_mode, value="obs")
            self.bay_obs_mode.grid(row=1,column=2)

            self.bay_obs_header = tk.Label(self.bay_popup, text="\"Observation\"")
            self.bay_obs_header.grid(row=1,column=3)

            self.bay_mod_mode = tk.ttk.Radiobutton(self.bay_popup, variable=self.bay_mode, value="model")
            self.bay_mod_mode.grid(row=2,column=2)

            self.bay_mod_header = tk.Label(self.bay_popup, text="(WIP) Model")
            self.bay_mod_header.grid(row=2,column=3)

            self.bay_title_label3 = tk.ttk.Label(self.bay_popup, text="Model Params", style="Header.TLabel")
            self.bay_title_label3.grid(row=4,column=2,columnspan=4)

            self.bay_mun_check = tk.Checkbutton(self.bay_popup, text="Mu_N", variable=self.check_bay_params["Mu_N"], onvalue=1, offvalue=0)
            self.bay_mun_check.grid(row=5,column=2, padx=(19,0))

            self.bay_mup_check = tk.Checkbutton(self.bay_popup, text="Mu_P", variable=self.check_bay_params["Mu_P"], onvalue=1, offvalue=0)
            self.bay_mup_check.grid(row=6,column=2, padx=(17,0))

            self.bay_n0_check = tk.Checkbutton(self.bay_popup, text="N0", variable=self.check_bay_params["N0"], onvalue=1, offvalue=0)
            self.bay_n0_check.grid(row=7,column=2, padx=(3,0))

            self.bay_p0_check = tk.Checkbutton(self.bay_popup, text="P0", variable=self.check_bay_params["P0"], onvalue=1, offvalue=0)
            self.bay_p0_check.grid(row=8,column=2)

            self.bay_B_check = tk.Checkbutton(self.bay_popup, text="B", variable=self.check_bay_params["B"], onvalue=1, offvalue=0)
            self.bay_B_check.grid(row=5,column=3, padx=(0,2))

            self.bay_taun_check = tk.Checkbutton(self.bay_popup, text="Tau_N", variable=self.check_bay_params["Tau_N"], onvalue=1, offvalue=0)
            self.bay_taun_check.grid(row=6,column=3, padx=(24,0))

            self.bay_taup_check = tk.Checkbutton(self.bay_popup, text="Tau_P", variable=self.check_bay_params["Tau_P"], onvalue=1, offvalue=0)
            self.bay_taup_check.grid(row=7,column=3, padx=(23,0))

            self.bay_sf_check = tk.Checkbutton(self.bay_popup, text="Sf", variable=self.check_bay_params["Sf"], onvalue=1, offvalue=0)
            self.bay_sf_check.grid(row=8,column=3, padx=(1,0))

            self.bay_sb_check = tk.Checkbutton(self.bay_popup, text="Sb", variable=self.check_bay_params["Sb"], onvalue=1, offvalue=0)
            self.bay_sb_check.grid(row=5,column=4, padx=(0,40))

            self.bay_temperature_check = tk.Checkbutton(self.bay_popup, text="Temperature", variable=self.check_bay_params["Temperature"], onvalue=1, offvalue=0)
            self.bay_temperature_check.grid(row=6,column=4, padx=(14,0))

            self.bay_relperm_check = tk.Checkbutton(self.bay_popup, text="Rel-Permitivity", variable=self.check_bay_params["Rel-Permitivity"], onvalue=1, offvalue=0)
            self.bay_relperm_check.grid(row=7,column=4, padx=(24,0))

            self.bay_theta_check = tk.Checkbutton(self.bay_popup, text="Theta", variable=self.check_bay_params["Theta"], onvalue=1, offvalue=0)
            self.bay_theta_check.grid(row=8,column=4, padx=(0,25))

            self.bay_alpha_check = tk.Checkbutton(self.bay_popup, text="Alpha", variable=self.check_bay_params["Alpha"], onvalue=1, offvalue=0)
            self.bay_alpha_check.grid(row=5,column=5, padx=(0,26))

            self.bay_delta_check = tk.Checkbutton(self.bay_popup, text="Delta", variable=self.check_bay_params["Delta"], onvalue=1, offvalue=0)
            self.bay_delta_check.grid(row=6,column=5, padx=(0,30))

            self.bay_fm_check = tk.Checkbutton(self.bay_popup, text="Frac-Emitted", variable=self.check_bay_params["Frac-Emitted"], onvalue=1, offvalue=0)
            self.bay_fm_check.grid(row=7,column=5, padx=(10,0))

            self.bay_continue_button = tk.Button(self.bay_popup, text="Continue", command=partial(self.on_bayesim_popup_close, continue_=True))
            self.bay_continue_button.grid(row=20,column=3)

            self.bay_popup.protocol("WM_DELETE_WINDOW", partial(self.on_bayesim_popup_close, continue_=False))
            self.bay_popup.grab_set()
            self.bayesim_popup_isopen = True
            return

        else:
            print("Error #600: Opened more than one Bayesim popup at a time")

        return

    def on_bayesim_popup_close(self, continue_=False):
        try:
            if continue_:
                print("Mode: {}".format(self.bay_mode.get()))
                for param in self.check_bay_params:
                    print("{}: {}".format(param, self.check_bay_params[param].get()))

                self.export_for_bayesim()

            self.bay_popup.destroy()
            print("Bayesim popup closed")
            self.bayesim_popup_isopen = False

        except:
            print("Error #601: Failed to close Bayesim popup")
        return
    ## Data File Readers for simulation and analysis tabs

    def read_TS(self, filename, index):
        ## Wrapper function: read a single time step from each of the data files
        self.sim_N = self.read_N(filename, index)
        self.sim_P = self.read_P(filename, index)
        self.sim_E_field = self.read_E_field(filename, index)
        return

    def read_N(self, filename, index):
        ## Read one time step's worth of data from N
        with tables.open_file(self.default_dirs["Data"] + "\\" + filename + "\\" + filename + "-n.h5", mode='r') as ifstream_N:
            return np.array(ifstream_N.root.N[index])

        return

    def read_P(self, filename, index):
        ## Read one time step from P
        with tables.open_file(self.default_dirs["Data"] + "\\" + filename + "\\" + filename + "-p.h5", mode='r') as ifstream_P:
            return np.array(ifstream_P.root.P[index])

        return

    def read_E_field(self, filename, index):
        ## Read one TS from E-Field
        with tables.open_file(self.default_dirs["Data"] + "\\" + filename + "\\" + filename + "-E_field.h5", mode='r') as ifstream_E_field:
            return np.array(ifstream_E_field.root.E_field[index])

        return

    ## Plotter for simulation tab    
    def update_sim_plots(self, index, do_clear_plots=True):
        ## V2: Update plots on Simulate tab
        ## FIXME: This abomination of 1x3 arrays
        
        # Why + 1e-30?
        # We want a log-scaled plot, and adding a tiny number to the y limits avoids the edge case of every value being zero.
        self.N_limits = [np.abs(np.amin(self.sim_N + 1e-30) * 1e-1), np.amax(self.sim_N + 1e-30) * 10]
        self.P_limits = [np.abs(np.amin(self.sim_P + 1e-30) * 1e-1), np.amax(self.sim_P + 1e-30) * 10]
        self.E_field_limits = [np.abs(np.amax(self.sim_E_field + 1e-30) * 1e-11), np.amax(self.sim_E_field + 1e-30) * 10]

        
        plot_list = [self.n_subplot, self.p_subplot, self.E_subplot]
        plot_ylabels = ["N", "P", "E_field"]
        plot_yunits = ["[cm^-3]", "[cm^-3]", "WIP"]
        plot_ylimits = np.array([self.N_limits, self.P_limits, self.E_field_limits])
        plot_ydata = [self.sim_N, self.sim_P, self.sim_E_field]
        plot_xdata = [self.nanowire.grid_x_nodes, self.nanowire.grid_x_nodes, self.nanowire.grid_x_edges]
        
        for i in range(plot_list.__len__()):
            
            plot = plot_list[i]
            
            if do_clear_plots: plot.cla()
        
            plot.set_ylim(plot_ylimits[i,0] * self.convert_out_dict[plot_ylabels[i]], plot_ylimits[i,1] * self.convert_out_dict[plot_ylabels[i]])
            plot.set_yscale('log')

            plot.plot(plot_xdata[i], plot_ydata[i] * self.convert_out_dict[plot_ylabels[i]])

            plot.set_xlabel("x [nm]")
            plot.set_ylabel("{} {}".format(plot_ylabels[i], plot_yunits[i]))

            plot.set_title("Time: {} ns".format(self.simtime * index / self.n))
        self.sim_fig.tight_layout()
        self.sim_fig.canvas.draw()
        return

    ## Sub plotters for analyze tab

    def plot_analyze(self, plot_ID, clear_plot=True):
        # Draw on analysis tab
        try:
            active_plot_data = self.analysis_plots[plot_ID]
            subplot = active_plot_data.plot_obj
            
            if clear_plot: subplot.cla()
            
            subplot.set_yscale(active_plot_data.yaxis_type)
            subplot.set_xscale(active_plot_data.xaxis_type)
            active_datagroup = active_plot_data.datagroup

            # subplot.set_ylim(*active_plot_data.ylim)
            # subplot.set_xlim(*active_plot_data.xlim)

            # This data is in TEDs units since we just used it in a calculation - convert back to common units first
            for dataset in active_datagroup.datasets.values():
                label = dataset.tag(for_matplotlib=True) + "*" if dataset.params_dict["symmetric_system"] else dataset.tag(for_matplotlib=True)
                subplot.plot(dataset.grid_x, dataset.data * self.convert_out_dict[active_datagroup.type], label=label)

            subplot.set_xlabel("x [nm]")
            subplot.set_ylabel(active_datagroup.type)
            subplot.legend().set_draggable(True)
            subplot.set_title("Time: " + str(active_datagroup.get_maxtime() * active_plot_data.time_index / active_datagroup.get_maxnumtsteps()) + " / " + str(active_datagroup.get_maxtime()) + "ns")
            self.analyze_fig.tight_layout()
            self.analyze_fig.canvas.draw()
            
            active_plot_data.ylim = subplot.get_ylim()
            active_plot_data.xlim = subplot.get_xlim()

        except:
            self.write(self.analysis_status, "Error #106: Plot failed")
            return

        return

    def read_data(self, data_filename, plot_ID, do_overlay):
        # Create a dataset object and prepare to plot on analysis tab
        # Select data type of incoming dataset from existing datasets
        active_plot = self.analysis_plots[plot_ID]
        if do_overlay: # If we already know what type of data is being plotted
            datatype = active_plot.datagroup.type
        else:
            try:
                datatype = self.data_var.get()
                if (datatype == ""): raise ValueError("Select a data type from the drop-down menu")
            except ValueError as oops:
                self.write(self.analysis_status, oops)
                return

        try:
            with open(self.default_dirs["Data"] + "\\" + data_filename + "\\" + "metadata.txt", "r") as ifstream:
                param_values_dict = {}
                for line in ifstream:
                    if "$" in line: continue

                    elif "#" in line: continue

                    else:
                        param = line[0:line.find(':')]
                        new_value = line[line.find(' ') + 1:].strip('\n')

                        if '\t' in new_value:
                            param_values_dict[param] = np.array(extract_values(new_value, '\t'))
                        else: param_values_dict[param] = float(new_value)

            data_n = int(0.5 + param_values_dict["Total-Time"] / param_values_dict["dt"])
            data_m = int(0.5 + param_values_dict["Total_length"] / param_values_dict["Node_width"])
            data_edge_x = np.linspace(0, param_values_dict["Total_length"],data_m+1)
            data_node_x = np.linspace(param_values_dict["Node_width"] / 2, param_values_dict["Total_length"] - param_values_dict["Node_width"] / 2, data_m)

            # Convert from cm, V, s to nm, V, ns
            for param in param_values_dict:
                if param in self.convert_in_dict:
                    param_values_dict[param] *= self.convert_in_dict[param]

        except:
            self.write(self.analysis_status, "Error: {} is missing or has unusual metadata.txt".format(data_filename))
            return

        active_datagroup = active_plot.datagroup
        if not do_overlay: 
            active_plot.time_index = 0
            active_datagroup.clear()

        active_show_index = active_plot.time_index

		# Now that we have the parameters from metadata, fetch the data itself
		# For N, P, E-field this is just reading the data but for others we'll calculate it in situ
        if (datatype == "N"):
            try:
                # Having data_node_x twice is NOT a typo - see definition of Data_Set() class for explanation
                new_data = Raw_Data_Set(self.read_N(data_filename, active_show_index), data_node_x, data_node_x, data_edge_x, param_values_dict, datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: The data set {} is missing -n data".format(data_filename))
                return

        elif (datatype == "P"):
            try:
                new_data = Raw_Data_Set(self.read_P(data_filename, active_show_index), data_node_x, data_node_x, data_edge_x, param_values_dict, datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: The data set {} is missing -p data".format(data_filename))
                return

        elif (datatype == "E_field"):
            try:
                new_data = Raw_Data_Set(self.read_E_field(data_filename, active_show_index), data_edge_x, data_node_x, data_edge_x, param_values_dict, datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: The data set {} is missing -E_field data".format(data_filename))
                return
        
        elif (datatype == "RR"):
            try:
                new_data = Raw_Data_Set(param_values_dict["B"] * ((self.read_N(data_filename, active_show_index)) * (self.read_P(data_filename, active_show_index)) - param_values_dict["N0"] * param_values_dict["P0"]), 
                                    data_node_x, data_node_x, data_edge_x, param_values_dict, datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: Unable to calculate Rad Rec")
                return

        elif (datatype == "NRR"):
            try:
                temp_N = self.read_N(data_filename, active_show_index)
                temp_P = self.read_P(data_filename, active_show_index)
                new_data = Raw_Data_Set((temp_N * temp_P - param_values_dict["N0"] * param_values_dict["P0"]) / 
                                    ((param_values_dict["Tau_N"] * temp_P) + (param_values_dict["Tau_P"] * temp_N)), 
                                    data_node_x, data_node_x, data_edge_x, param_values_dict, datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: Unable to calculate Non Rad Rec")
                return

        elif (datatype == "PL"):
            try:
                rad_rec = param_values_dict["B"] * (self.read_N(data_filename, active_show_index) * self.read_P(data_filename, active_show_index) - \
                    param_values_dict["N0"] * param_values_dict["P0"])

                max = param_values_dict["Total_length"]
                dx = param_values_dict["Node_width"]
                distance = np.linspace(0, max - dx, data_m)
                alphaCof = param_values_dict["Alpha"] if not param_values_dict["ignore_alpha"] else 0
                thetaCof = param_values_dict["Theta"]
                delta_frac = param_values_dict["Delta"]
                fracEmitted = param_values_dict["Frac-Emitted"]
                symmetric_system = param_values_dict["symmetric_system"]
                
                # Make room for one more value than necessary - this value at index j+1 will be used to pad the upper correction
                distance_matrix = np.zeros((data_m, data_m))
                lf_distance_matrix = np.zeros((data_m, data_m))

                # Each row in weight will represent the weight function centered around a different position
                # Total reflection is assumed to occur at either end of the system: 
                # Left (x=0) reflection is equivalent to a symmetric wire situation while right reflection (x=thickness) is usually negligible
                for n in range(0, data_m):
                    distance_matrix[n] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:data_m - n]))
                    lf_distance_matrix[n] = distance + (n * dx)

                weight = np.exp(-(alphaCof + thetaCof) * distance_matrix)
                lf_weight = np.exp(-(alphaCof + thetaCof) * lf_distance_matrix) if symmetric_system else 0

                combined_weight = (1 - fracEmitted) * 0.5 * thetaCof * delta_frac * (weight + lf_weight)

                weight2 = np.exp(-(thetaCof) * distance_matrix)
                lf_weight2 = np.exp(-(thetaCof) * lf_distance_matrix) if symmetric_system else 0

                combined_weight2 = (1 - fracEmitted) * 0.5 * thetaCof * (1 - delta_frac) * (weight2 + lf_weight2)

                PL_base = fracEmitted * rad_rec

                for p in range(0, data_m):
                    # To each value of the slice add the attenuation contribution with weight centered around that value's corresponding position
                    PL_base[p] += intg.trapz(combined_weight[p] * rad_rec, dx=dx) + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * rad_rec[p] + \
                        intg.trapz(combined_weight2[p] * rad_rec, dx=dx) + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * rad_rec[p]

                new_data = Raw_Data_Set(PL_base, data_node_x, data_node_x, data_edge_x, param_values_dict, datatype, data_filename, active_show_index)

            except OSError:
                self.write(self.analysis_status, "Error: Unable to calculate PL")
                return

        try:
            active_datagroup.add(new_data, new_data.tag())

        except ValueError:
            self.write(self.analysis_status, "Error: dt or total t mismatch")
        return

    def fetch_dataset(self):
        # Wrapper to apply read_data() on multiple selected datasets
        # THe Plot button on the Analyze tab calls this function
        plot_ID = self.active_analysisplot_ID.get()
        self.do_plotter_popup(plot_ID)
        self.root.wait_window(self.plotter_popup)
        
        active_plot = self.analysis_plots[plot_ID]
        if (active_plot.data_filenames.__len__() == 0): return

        data_filename = active_plot.data_filenames[0]
        short_filename = data_filename[data_filename.rfind('/') + 1:]

        self.read_data(short_filename, plot_ID, do_overlay=False)

        for i in range(1, active_plot.data_filenames.__len__()):
            data_filename = active_plot.data_filenames[i]
            short_filename = data_filename[data_filename.rfind('/') + 1:]
            self.read_data(short_filename, plot_ID, do_overlay=True)

        
        #active_plot.xlim = (0, active_plot.datagroup.get_max_x())
        #max_val = active_plot.datagroup.get_maxval() * self.convert_out_dict[active_plot.datagroup.type]
        #active_plot.ylim = (max_val * 1e-11, max_val * 10)
        active_plot.xaxis_type = 'linear'
        active_plot.yaxis_type = 'log'
        self.plot_analyze(plot_ID, clear_plot=True)
        
        if self.check_autointegrate.get():
            self.write(self.analysis_status, "Data read success; integrating...")
            self.do_Integrate(bypass_inputs=True)
            
        else:
            self.write(self.analysis_status, "Data read success")
        
        return

    def plot_tstep(self):
        # Step already plotted data forward (or backward) in time
        plot_ID = self.active_analysisplot_ID.get()
        active_plot = self.analysis_plots[plot_ID]
        try:
            active_plot.add_time_index(int(self.analyze_tstep_entry.get()))
        except ValueError:
            self.write(self.analysis_status, "Invalid number of time steps")
            return

        active_show_index = active_plot.time_index
        active_datagroup = active_plot.datagroup

        # Search data files for data at new time step
        if active_datagroup.type == "N":
            for tag in active_datagroup.datasets:
                active_datagroup.datasets[tag].data = self.read_N(active_datagroup.datasets[tag].filename, active_show_index)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "P":
            for tag in active_datagroup.datasets:
                active_datagroup.datasets[tag].data = self.read_P(active_datagroup.datasets[tag].filename, active_show_index)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "E_field":
            for tag in active_datagroup.datasets:
                active_datagroup.datasets[tag].data = self.read_E_field(active_datagroup.datasets[tag].filename, active_show_index)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "RR":
            for tag in active_datagroup.datasets:
                n0 = active_datagroup.datasets[tag].params_dict["N0"]
                p0 = active_datagroup.datasets[tag].params_dict["P0"]
                B = active_datagroup.datasets[tag].params_dict["B"]
                active_datagroup.datasets[tag].data = B * (self.read_N(active_datagroup.datasets[tag].filename, active_show_index) * self.read_P(active_datagroup.datasets[tag].filename, active_show_index) - n0 * p0)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "NRR":
            for tag in active_datagroup.datasets:
                temp_N = self.read_N(active_datagroup.datasets[tag].filename, active_show_index)
                temp_P = self.read_P(active_datagroup.datasets[tag].filename, active_show_index)
                n0 = active_datagroup.datasets[tag].params_dict["N0"]
                p0 = active_datagroup.datasets[tag].params_dict["P0"]
                tauN = active_datagroup.datasets[tag].params_dict["Tau_N"]
                tauP = active_datagroup.datasets[tag].params_dict["Tau_P"]
                active_datagroup.datasets[tag].data = (temp_N * temp_P - n0 * p0) / ((tauN * temp_P) + (tauP * temp_N))
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "PL":
            for tag in active_datagroup.datasets:
                filename = active_datagroup.datasets[tag].filename
                B = active_datagroup.datasets[tag].params_dict["B"]
                n0 = active_datagroup.datasets[tag].params_dict["N0"]
                p0 = active_datagroup.datasets[tag].params_dict["P0"]
                rad_rec = B * ((self.read_N(filename, active_show_index) * self.read_P(filename, active_show_index)) - n0 * p0)

                max = active_datagroup.datasets[tag].params_dict["Total_length"]
                dx = active_datagroup.datasets[tag].params_dict["Node_width"]
                data_m = int(0.5 + max / dx)
                distance = np.linspace(0, max - dx, data_m)
                alphaCof = active_datagroup.datasets[tag].params_dict["Alpha"] if not active_datagroup.datasets[tag].params_dict["ignore_alpha"] else 0
                thetaCof = active_datagroup.datasets[tag].params_dict["Theta"]
                delta_frac = active_datagroup.datasets[tag].params_dict["Delta"]
                fracEmitted = active_datagroup.datasets[tag].params_dict["Frac-Emitted"]

                # Make room for one more value than necessary - this value at index j+1 will be used to pad the upper correction
                distance_matrix = np.zeros((data_m, data_m))
                lf_distance_matrix = np.zeros((data_m, data_m))

                # Each row in weight will represent the weight function centered around a different position
                # Total reflection is assumed to occur at either end of the system: 
                # Left (x=0) reflection is equivalent to a symmetric wire situation while right reflection (x=thickness) is usually negligible
                for n in range(0, data_m):
                    distance_matrix[n] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:data_m - n]))
                    lf_distance_matrix[n] = distance + (n * dx)

                weight = np.exp(-(alphaCof + thetaCof) * distance_matrix)
                lf_weight = np.exp(-(alphaCof + thetaCof) * lf_distance_matrix) if active_datagroup.datasets[tag].params_dict["symmetric_system"] else 0

                combined_weight = (1 - fracEmitted) * 0.5 * thetaCof * delta_frac * (weight + lf_weight)

                weight2 = np.exp(-(thetaCof) * distance_matrix)
                lf_weight2 = np.exp(-(thetaCof) * lf_distance_matrix) if active_datagroup.datasets[tag].params_dict["symmetric_system"] else 0

                combined_weight2 = (1 - fracEmitted) * 0.5 * thetaCof * (1 - delta_frac) * (weight2 + lf_weight2)

                PL_base = fracEmitted * rad_rec

                for p in range(0, data_m):
                    # To each value of the slice add the attenuation contribution with weight centered around that value's corresponding position
                    PL_base[p] += intg.trapz(combined_weight[p] * rad_rec, dx=dx) + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * rad_rec[p] + \
                        intg.trapz(combined_weight2[p] * rad_rec, dx=dx) + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * rad_rec[p]

                active_datagroup.datasets[tag].data = PL_base
                active_datagroup.datasets[tag].show_index = active_show_index

        else:
            self.write(self.analysis_status, "Error #107: Data group has an invalid datatype")

        self.plot_analyze(plot_ID, clear_plot=True)
        self.write(self.analysis_status, "")
        return

    ## Status box update helpers
    # Edit the text in status boxes
	# The user is NOT meant to write into these
    def write(self, textBox, text):
        textBox.configure(state='normal')
        textBox.delete(1.0, tk.END)
        textBox.insert(tk.END, text)
        textBox.configure(state='disabled') # Prevents user from altering the status box
        return

	# Fill in entry boxes with parameters
	# The user IS meant to write into these
    def enter(self, entryBox, text):
        entryBox.delete(0,tk.END)
        entryBox.insert(0,text)
        return

    ## Tab change event handlers
	# This doesn't do anything meaningful at the moment but could be of use for things that need to be updated every time the user goes to a new tab
    def on_tab_selected(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")

        if (tab_text == "Inputs"):
            print("Inputs tab selected")
            #self.update_IC_filebox()

        elif (tab_text == "Simulate"):
            print("Simulate tab selected")
            

        elif (tab_text == "Analyze"):
            print("Analzye tab selected")

        return

    def on_input_subtab_selected(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")

        if (tab_text == "Analytical Init. Cond."):
            print("Analytical subtab selected")

        elif (tab_text == "Heuristic Init. Cond."):
            print("Heuristic subtab selected")

        return

    # Wrapper function to set up main N, P, E-field calculations
    def do_Batch(self):
        # We test that the two following entryboxes are valid before opening any popups
        # Imagine if you selected 37 files from the popup and TED refused to calculate because these entryboxes were empty!
        try:
            self.simtime = float(self.simtime_entry.get())      # [ns]
            self.dt = float(self.dt_entry.get())           # [ns]

            if (self.simtime <= 0): raise Exception("Error: Invalid simulation time")
            if (self.dt <= 0 or self.dt > self.simtime): raise Exception("Error: Invalid dt")
        
        except ValueError:
            self.write(self.status, "Error: Invalid parameters")
            return

        except Exception as oops:
            self.write(self.status, oops)
            return

        IC_files = tk.filedialog.askopenfilenames(initialdir = self.default_dirs["Initial"], title="Select IC text file", filetypes=[("Text files","*.txt")])
        if (IC_files.__len__() == 0): return

        batch_num = 0
        self.error_states = []
        for IC in IC_files:
            batch_num += 1
            self.IC_file_name = IC
            self.load_ICfile()
            self.write(self.status, "Now calculating {} : ({} of {})".format(self.IC_file_name[self.IC_file_name.rfind("/") + 1:self.IC_file_name.rfind(".txt")], str(batch_num), str(IC_files.__len__())))
            self.do_Calculate()
            
        if not len(self.error_states):
            self.write(self.status, "Simulations complete")
        else:
            self.write(self.status, "Warning: fatal errors logged in console for {} simulations".format(len(self.error_states)))

        return

	# The big function that does all the simulating
    # TODO: This should belong to Nanowire(), not Notebook()
    def do_Calculate(self):
        ## Setup parameters
        try:
            # Construct the data folder's name from the corresponding IC file's name
            shortened_IC_name = self.IC_file_name[self.IC_file_name.rfind("/") + 1:self.IC_file_name.rfind(".txt")]
            data_file_name = shortened_IC_name
            
            self.m = int(0.5 + self.nanowire.total_length / self.nanowire.dx)         # Number of space steps
            self.n = int(0.5 + self.simtime / self.dt)           # Number of time steps

            # Upper limit on number of time steps
            if (self.n > 2.5e5): raise Exception("Error: too many time steps")

            temp_sim_dict = {}

            # Convert into TEDs units
            for param in self.nanowire.param_dict:
                temp_sim_dict[param] = self.nanowire.param_dict[param].value * self.convert_in_dict[param]

            # The three true initial conditions for our three coupled ODEs
            # N = N0 + delta_N
            self.init_N = (self.nanowire.param_dict["N0"].value + self.nanowire.param_dict["init_deltaN"].value) * self.convert_in_dict["N"]
            self.init_P = (self.nanowire.param_dict["P0"].value + self.nanowire.param_dict["init_deltaP"].value) * self.convert_in_dict["P"]
            self.init_E_field = self.nanowire.param_dict["init_E_field"].value * self.convert_in_dict["E_field"]
            # "Typecast" single values to uniform arrays
            if not isinstance(self.init_N, np.ndarray):
                self.init_N = np.ones(self.nanowire.grid_x_nodes.__len__()) * self.init_N
                
            if not isinstance(self.init_P, np.ndarray):
                self.init_P = np.ones(self.nanowire.grid_x_nodes.__len__()) * self.init_P
                
            if not isinstance(self.init_E_field, np.ndarray):
                self.init_E_field = np.ones(self.nanowire.grid_x_edges.__len__()) * self.init_E_field
        
        except ValueError:
            print("Error: Invalid parameters for {}".format(data_file_name))
            self.error_states.append(data_file_name)
            return

        except Exception as oops:
            print("{} for {}".format(oops, data_file_name))
            self.error_states.append(data_file_name)
            return
    
        try:

            print("Attempting to create {} data folder".format(data_file_name))

            full_path_name = "{}\\{}".format(self.default_dirs["Data"], data_file_name)
            # Append a number to the end of the new directory's name if an overwrite would occur
            # This is what happens if you download my_file.txt twice and the second copy is saved as my_file(1).txt, for example
            ## TODO: Overwrite warning - alert user when this happens
            if os.path.isdir(full_path_name):
                print("{} folder already exists; trying alternate name".format(data_file_name))
                append = 1
                while (os.path.isdir("{}({})".format(full_path_name, append))):
                    append += 1

                full_path_name = "{}({})".format(full_path_name, append)
                data_file_name = "{}({})".format(data_file_name, append)

            os.mkdir("{}".format(full_path_name))

        except FileExistsError:
            print("Error: unable to create directory for results of simulation {}".format(shortened_IC_name))
            self.error_states.append(data_file_name)
            return


        ## Calculate!
        atom = tables.Float64Atom()

        ## Create data files
        with tables.open_file(full_path_name + "\\" + data_file_name + "-n.h5", mode='w') as ofstream_N, \
            tables.open_file(full_path_name + "\\" + data_file_name + "-p.h5", mode='w') as ofstream_P, \
            tables.open_file(full_path_name + "\\" + data_file_name + "-E_field.h5", mode='w') as ofstream_E_field:
            array_N = ofstream_N.create_earray(ofstream_N.root, "N", atom, (0, self.m))
            array_P = ofstream_P.create_earray(ofstream_P.root, "P", atom, (0, self.m))
            array_E_field = ofstream_E_field.create_earray(ofstream_E_field.root, "E_field", atom, (0, self.m+1))
            array_N.append(np.reshape(self.init_N, (1, self.m)))
            array_P.append(np.reshape(self.init_P, (1, self.m)))
            array_E_field.append(np.reshape(self.init_E_field, (1, self.m + 1)))
        
        ## Setup simulation plots and plot initial
        
        self.sim_N = self.init_N
        self.sim_P = self.init_P
        self.sim_E_field = self.init_E_field
        self.update_sim_plots(0)

        numTimeStepsDone = 0

        # WIP: Option for staggered calculate/plot: In this mode the program calculates a block of time steps, plots intermediate (N, P, E), and calculates the next block 
        # using the final time step from the previous block as the initial condition.
        # This mode can be disabled by inputting numPartitions = 1.
        #for i in range(1, self.numPartitions):
            
        #    #finite.simulate_nanowire(self.IC_file_name,self.m,int(self.n / self.numPartitions),self.dx,self.dt, *(boundaryParams), *(systemParams), False, self.alphaCof, self.thetaCof, self.fracEmitted, self.max_iter, self.init_N, self.init_P, self.init_E_field)
        #    finite.ode_nanowire(self.IC_file_name,self.m,int(self.n / self.numPartitions),self.dx,self.dt, *(boundaryParams), *(systemParams), False, self.alphaCof, self.thetaCof, self.fracEmitted, self.init_N, self.init_P, self.init_E_field)

        #    numTimeStepsDone += int(self.n / self.numPartitions)
        #    self.write(self.status, "Calculations {:.1f}% complete".format(100 * i / self.numPartitions))
            

        #    with tables.open_file("Data\\" + self.IC_file_name + "\\" + self.IC_file_name + "-n.h5", mode='r') as ifstream_N, \
        #        tables.open_file("Data\\" + self.IC_file_name + "\\" + self.IC_file_name + "-p.h5", mode='r') as ifstream_P, \
        #        tables.open_file("Data\\" + self.IC_file_name + "\\" + self.IC_file_name + "-E_field.h5", mode='r') as ifstream_E_field:
        #        self.init_N = ifstream_N.root.N[-1]
        #        self.init_P = ifstream_P.root.P[-1]
        #        self.init_E_field = ifstream_E_field.root.E_field[-1]


        #    self.update_sim_plots(int(self.n * i / self.numPartitions), self.numPartitions > 20)
            #self.update_err_plots()

        write_output = True

        try:
            error_dict = finite.ode_nanowire(full_path_name,data_file_name,self.m,self.n - numTimeStepsDone,self.nanowire.dx,self.dt, temp_sim_dict,
                                             not self.check_ignore_recycle.get(), self.check_symmetric.get(), self.check_do_ss.get(), write_output,
                                             self.init_N, self.init_P, self.init_E_field)
        except FloatingPointError:
            print("Error: an unusual value occurred while simulating {}".format(data_file_name))
            self.error_states.append(data_file_name)
            return
        
        grid_t = np.linspace(self.dt, self.simtime, self.n)

        try:
            np.savetxt(full_path_name + "\\convergence.csv", np.vstack((grid_t, error_dict['hu'], error_dict['tcur'],\
                error_dict['tolsf'], error_dict['tsw'], error_dict['nst'], error_dict['nfe'], error_dict['nje'], error_dict['nqu'],\
                error_dict['mused'])).transpose(), fmt='%.4e', delimiter=',', header="t, hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, mused")
        except PermissionError:
            print(self.status, "Error: unable to access convergence data export destination")
        
        self.write(self.status, "Finalizing...")

        #self.read_TS(self.n)
        #self.update_sim_plots(self.n, self.numPartitions > 20)

        for i in range(1,6):
            self.read_TS(data_file_name, int(self.n * i / 5))
            self.update_sim_plots(self.n, do_clear_plots=False)

        # Save metadata: list of param values used for the simulation
        # Inverting the unit conversion between the inputted params and the calculation engine is also necessary to regain the originally inputted param values

        with open(full_path_name + "\\metadata.txt", "w+") as ofstream:
            ofstream.write("$$ METADATA FOR CALCULATIONS PERFORMED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
            ofstream.write("Total_length: " + str(self.nanowire.total_length) + "\n")
            ofstream.write("Node_width: " + str(self.nanowire.dx) + "\n")
            
            for param in temp_sim_dict:
                param_values = temp_sim_dict[param] * self.convert_out_dict[param]
                if isinstance(param_values, np.ndarray):
                    ofstream.write("{}: {:.8e}".format(param, param_values[0]))
                    for value in param_values[1:]:
                        ofstream.write("\t{:.8e}".format(value))
                        
                    ofstream.write('\n')
                else:
                    ofstream.write("{}: {}\n".format(param, param_values))

            # The following params are exclusive to metadata files
            ofstream.write("Total-Time: " + str(self.simtime) + "\n")
            ofstream.write("dt: " + str(self.dt) + "\n")
            if self.check_ignore_recycle.get(): ofstream.write("ignore_alpha: 1\n")
            else: ofstream.write("ignore_alpha: 0\n")

            if self.check_symmetric.get(): ofstream.write("symmetric_system: 1\n")
            else: ofstream.write("symmetric_system: 0\n")
            
            if self.check_do_ss.get(): ofstream.write("steady_state_exc: 1\n")
            else: ofstream.write("steady_state_exc: 0\n")

        return

    def do_Integrate(self, bypass_inputs=False):
        plot_ID = self.active_analysisplot_ID.get()
        self.write(self.analysis_status, "")

        active_plot = self.analysis_plots[plot_ID]
        active_datagroup = active_plot.datagroup
        if active_datagroup.datasets.__len__() == 0: return

        # Collect instructions from user using a series of popup windows
        if not bypass_inputs:
            self.do_integration_popup()
            self.root.wait_window(self.integration_popup) # Pause here until popup is closed
            if self.PL_mode == "":
                self.write(self.analysis_status, "Integration cancelled")
                return
    
            self.do_integration_getbounds_popup()
            self.root.wait_window(self.integration_getbounds_popup)
    
            if self.PL_mode == "Current time step":
                self.do_PL_xaxis_popup()
                self.root.wait_window(self.PL_xaxis_popup)
                if self.xaxis_param == "":
                    self.write(self.analysis_status, "Integration cancelled")
                    return
                print("Selected param {}".format(self.xaxis_param))
                self.integration_plots[0].x_param = self.xaxis_param
    
            else:
                self.integration_plots[0].x_param = "Time"
                
        else:
            # A "default integration behavior": integrate the present data over all time and space steps
            self.PL_mode = "All time steps"
            self.integration_plots[0].x_param = "Time"
            self.integration_bounds = [[0,active_datagroup.get_max_x()]]
            
        # Clean up the I_plot and prepare to integrate given selections
        # A lot of the following is a data transfer between the sending active_datagroup and the receiving I_plot
        self.integration_plots[0].datagroup.clear()
        self.integration_plots[1].datagroup.clear()
        self.integration_plots[0].mode = self.PL_mode
        self.integration_plots[0].global_gridx = None

        
        n = active_datagroup.get_maxnumtsteps()
        
        counter = 0
        # Integrate for EACH dataset in chosen datagroup
        for tag in active_datagroup.datasets:
            data_filename = active_datagroup.datasets[tag].filename
            datatype = active_datagroup.datasets[tag].type
            print("Now integrating {}".format(data_filename))

            # Unpack needed params from the dictionaries of params
            dx = active_datagroup.datasets[tag].params_dict["Node_width"]
            total_length = active_datagroup.datasets[tag].params_dict["Total_length"]
            total_time = active_datagroup.datasets[tag].params_dict["Total-Time"]
            dt = active_datagroup.datasets[tag].params_dict["dt"]
            B_param = active_datagroup.datasets[tag].params_dict["B"]
            n0 = active_datagroup.datasets[tag].params_dict["N0"]
            p0 = active_datagroup.datasets[tag].params_dict["P0"]
            tauN = active_datagroup.datasets[tag].params_dict["Tau_N"]
            tauP = active_datagroup.datasets[tag].params_dict["Tau_P"]
            alpha = active_datagroup.datasets[tag].params_dict["Alpha"] if (active_datagroup.datasets[tag].params_dict["ignore_alpha"] == 0.0) else 0
            theta = active_datagroup.datasets[tag].params_dict["Theta"]
            delta = active_datagroup.datasets[tag].params_dict["Delta"]
            frac_emitted = active_datagroup.datasets[tag].params_dict["Frac-Emitted"]
            symmetric_flag = active_datagroup.datasets[tag].params_dict["symmetric_system"]

            if self.PL_mode == "Current time step":
                show_index = active_datagroup.datasets[tag].show_index

            # Clean up any bounds that extend past the confines of the system
            # The system usually exists from x=0 to x=total_length, but can accept x=-total_length to x=total_length if symmetric

            for bounds in self.integration_bounds:
                l_bound = bounds[0]
                u_bound = bounds[1]
               
                if (u_bound > total_length):
                    u_bound = total_length

                if symmetric_flag:

                    if (l_bound < -total_length):
                        l_bound = -total_length
                else:
                    if (l_bound < 0):
                        l_bound = 0

                include_negative = symmetric_flag and (l_bound < 0)

                print("Bounds after cleanup: {} to {}".format(l_bound, u_bound))

                if (datatype == "N"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-n.h5", mode='r') as ifstream_N:
                        data = ifstream_N.root.N
                        if include_negative:
                            I_data = finite.integrate(data, 0, -l_bound, dx, total_length) + \
                                finite.integrate(data, 0, u_bound, dx, total_length)
                        else:
                            I_data = finite.integrate(data, l_bound, u_bound, dx, total_length)
            
                elif (datatype == "P"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-p.h5", mode='r') as ifstream_P:
                        data = ifstream_P.root.P
                        if include_negative:
                            I_data = finite.integrate(data, 0, -l_bound, dx, total_length) + \
                                finite.integrate(data, 0, u_bound, dx, total_length)
                        else:
                            I_data = finite.integrate(data, l_bound, u_bound, dx, total_length)

                elif (datatype == "E_field"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-E_field.h5", mode='r') as ifstream_E_field:
                        data = ifstream_E_field.root.E_field
                        if include_negative:
                            I_data = finite.integrate(data, 0, -l_bound, dx, total_length) + \
                                finite.integrate(data, 0, u_bound, dx, total_length)
                        else:
                            I_data = finite.integrate(data, l_bound, u_bound, dx, total_length)

                elif (datatype == "RR"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-n.h5", mode='r') as ifstream_N, \
                        tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-p.h5", mode='r') as ifstream_P:
                        temp_N = np.array(ifstream_N.root.N)
                        temp_P = np.array(ifstream_P.root.P)

                        data = B_param * (temp_N) * (temp_P) - n0 * p0
                        if include_negative:
                            I_data = finite.integrate(data, 0, -l_bound, dx, total_length) + \
                                finite.integrate(data, 0, u_bound, dx, total_length)
                        else:
                            I_data = finite.integrate(data, l_bound, u_bound, dx, total_length)

                elif (datatype == "NRR"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-n.h5", mode='r') as ifstream_N, \
                        tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-p.h5", mode='r') as ifstream_P:
                        temp_N = np.array(ifstream_N.root.N)
                        temp_P = np.array(ifstream_P.root.P)
                        data = ((temp_N) * (temp_P) - n0 * p0) / (tauN * (temp_P) + tauP * (temp_N))
                        if include_negative:
                            I_data = finite.integrate(data, 0, -l_bound, dx, total_length) + \
                                finite.integrate(data, 0, u_bound, dx, total_length)
                        else:
                            I_data = finite.integrate(data, l_bound, u_bound, dx, total_length)

                else: # datatype = "PL"
                    if include_negative:
                        I_data = finite.propagatingPL(data_filename, 0, -l_bound, dx, 0, total_length, B_param, n0, p0, alpha, theta, delta, frac_emitted, symmetric_flag) + \
                            finite.propagatingPL(data_filename, 0, u_bound, dx, 0, total_length, B_param, n0, p0, alpha, theta, delta, frac_emitted, symmetric_flag)
                    else:
                        I_data = finite.propagatingPL(data_filename, l_bound, u_bound, dx, 0, total_length, B_param, n0, p0, alpha, theta, delta, frac_emitted, symmetric_flag)
                            

                if self.PL_mode == "Current time step":
                    # FIXME: We don't need to integrate everything just to extract a single time step
                    # Change the integration procedure above to work only with the needed data
                    I_data = I_data[show_index]

                    # Don't forget to change out of TEDs units, or the x axis won't match the parameters the user typed in
                    grid_xaxis = float(active_datagroup.datasets[tag].params_dict[self.xaxis_param] * self.convert_out_dict[self.xaxis_param])

                    xaxis_label = self.xaxis_param + " [WIP]"

                elif self.PL_mode == "All time steps":
                    self.integration_plots[0].global_gridx = np.linspace(0, total_time, n + 1)
                    grid_xaxis = -1 # A dummy value for the I_Set constructor
                    xaxis_label = "Time [ns]"

                self.integration_plots[0].datagroup.add(Integrated_Data_Set(I_data, grid_xaxis, active_datagroup.datasets[tag].params_dict, active_datagroup.datasets[tag].type, data_filename + "__" + str(l_bound) + "_to_" + str(u_bound)))

                # Do some bonus calculations involving PL's time derivative
                if (self.PL_mode == "All time steps" and datatype == "PL"):
                    self.integration_plots[1].global_gridx = np.linspace(0, total_time, n + 1)
                    tau_diff = finite.tau_diff(I_data, dt)
                    self.integration_plots[1].datagroup.add(Integrated_Data_Set(tau_diff, grid_xaxis, active_datagroup.datasets[tag].params_dict, active_datagroup.datasets[tag].type, data_filename + "__" + str(l_bound) + "_to_" + str(u_bound)))
                        
                counter += 1
                print("Integration: {} of {} complete".format(counter, active_datagroup.size() * self.integration_bounds.__len__()))

            
        subplot = self.integration_plots[0].plot_obj
        datagroup = self.integration_plots[0].datagroup
        subplot.cla()
        
        #max = datagroup.get_maxval() * self.convert_out_dict[datagroup.type]
        

        self.integration_plots[0].xaxis_type = 'linear'
        self.integration_plots[0].yaxis_type = 'log'
        #self.integration_plots[0].ylim = max * 1e-12, max * 10

        subplot.set_yscale(self.integration_plots[0].yaxis_type)
        #subplot.set_ylim(self.integration_plots[0].ylim)
        subplot.set_xlabel(xaxis_label)
        subplot.set_ylabel(datagroup.type)
        subplot.set_title("Integrated {}".format(datagroup.type))

        for key in datagroup.datasets:

            if self.PL_mode == "Current time step":
                subplot.scatter(datagroup.datasets[key].grid_x, datagroup.datasets[key].data * self.convert_out_dict[datagroup.type], label=datagroup.datasets[key].tag(for_matplotlib=True))

            elif self.PL_mode == "All time steps":
                subplot.plot(self.integration_plots[0].global_gridx, datagroup.datasets[key].data * self.convert_out_dict[datagroup.type], label=datagroup.datasets[key].tag(for_matplotlib=True))
                
        self.integration_plots[0].xlim = subplot.get_xlim()
        self.integration_plots[0].ylim = subplot.get_ylim()
                
        subplot.legend().set_draggable(True)
        
        self.taudiff_subplot.cla()
        if self.integration_plots[1].datagroup.size(): # If has tau_diff data to plot
            self.taudiff_subplot.set_ylabel("tau_diff")
            self.taudiff_subplot.set_xlabel("Time [ns]")
            self.taudiff_subplot.set_title("-(dln(PL)/dt)^(-1)")
            for key in self.integration_plots[1].datagroup.datasets:
                self.taudiff_subplot.plot(self.integration_plots[1].global_gridx, self.integration_plots[1].datagroup.datasets[key].data, label=self.integration_plots[1].datagroup.datasets[key].tag(for_matplotlib=True))
        
            self.taudiff_subplot.legend().set_draggable(True)
            
            self.integration_plots[1].xaxis_type = 'linear'
            self.integration_plots[1].yaxis_type = 'linear'
            self.integration_plots[1].ylim = self.taudiff_subplot.get_ylim()
            self.integration_plots[1].xlim = self.taudiff_subplot.get_xlim()

        self.integration_fig.tight_layout()
        self.integration_fig.canvas.draw()
        
        self.write(self.analysis_status, "Integration complete")

        return

    ## Initial Condition Managers

    def reset_IC(self, force=False):
        # V2 Update: On IC tab:
        # 1. Remove all param_rules from all selected Parameters in the listbox
        # 2. Remove all param_rules from all selected Parameters stored in Nanowire()
        # 3. Remove values stored in Nanowire()
        # + any visual changes to appeal to the user

        self.do_resetIC_popup()
        self.root.wait_window(self.resetIC_popup)

        if (not self.resetIC_selected_params):
            print("No params selected :(")
            return

        for param in self.resetIC_selected_params:
            # Step 1 and 2
            self.HIC_listbox_currentparam = param
            
            # These two lines changes the text displayed in the param_rule display box's menu and is for cosmetic purposes only
            self.update_paramrule_listbox(param)
            self.HIC_viewer_selection.set(param)
            
            self.deleteall_HIC()
            
            # Step 3
            self.nanowire.param_dict[param].value = 0
            self.update_IC_plot(plot_ID="recent")
        #if self.check_reset_params.get() or force:
        #    for key in self.sys_param_entryboxes_dict:
        #        self.enter(self.sys_param_entryboxes_dict[key], "")

        #    cleared_items += " Params,"

        if self.resetIC_do_clearall:
            self.set_thickness_and_dx_entryboxes(state='unlock')
            self.nanowire.total_length = None
            self.nanowire.dx = None
            self.nanowire.grid_x_edges = []
            self.nanowire.grid_x_nodes = []
            self.nanowire.spacegrid_is_set = False
            self.update_system_summary()

        self.write(self.ICtab_status, "Selected params cleared")
        return

	## This is a patch of a consistency issue involving initial conditions - we require different variables of a single initial condition
	## to fit to the same spatial mesh, which can get messed up if the user changes the mesh while editing initial conditions.

    # First, implement a way to temporarily remove the user's ability to change variables associated with the spatial mesh
    def set_thickness_and_dx_entryboxes(self, state):
        if state =='lock':
            self.thickness_entry.configure(state='disabled')
            self.dx_entry.configure(state='disabled')

        elif state =='unlock':
            self.thickness_entry.configure(state='normal')
            self.dx_entry.configure(state='normal')

        return

    # Second, create a function that generates and locks in new spatial meshes. A new mesh can only be generated when the previous mesh is discarded using reset_IC().
    def set_init_x(self):
        # Changed for V2

        if self.nanowire.spacegrid_is_set:
            return

        thickness = float(self.thickness_entry.get())
        dx = float(self.dx_entry.get())

        if (thickness <= 0 or dx <= 0): raise ValueError

        if not finite.check_valid_dx(thickness, dx):
            raise Exception("Error: space step size larger than thickness")

        # Upper limit on number of space steps
        if (int(0.5 + thickness / dx) > 1e6): 
            raise Exception("Error: too many space steps")

        self.nanowire.total_length = thickness
        self.nanowire.dx = dx
        self.nanowire.grid_x_nodes = np.linspace(dx / 2,thickness - dx / 2, int(0.5 + thickness / dx))
        self.nanowire.grid_x_edges = np.linspace(0, thickness, int(0.5 + thickness / dx) + 1)
        self.nanowire.spacegrid_is_set = True
        self.set_thickness_and_dx_entryboxes(state='lock')
        return

    def add_AIC(self):
        # Read AIC parameters and plot when relevant button pressed
        try:
            self.set_init_x()

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        # Check for valid option choices
        AIC_options = {"long_expfactor":self.check_calculate_init_material_expfactor.get(), 
                     "incidence":self.AIC_stim_mode.get(),
                     "power_mode":self.AIC_gen_power_mode.get()}
        try:
            if AIC_options["long_expfactor"] == '' or AIC_options["power_mode"] == '':
                raise ValueError("Error: select material param and power generation options ")
        except ValueError as oops:
            self.write(self.ICtab_status, oops)
            return

        # Remove all param_rules for init_deltaN and init_deltaP, as we will be reassigning them shortly.
        self.HIC_listbox_currentparam = "init_deltaN"
        self.deleteall_HIC()
        self.HIC_listbox_currentparam = "init_deltaP"
        self.deleteall_HIC()

        # Establish constants; calculate alpha
        h = 6.626e-34   # [J*s]
        c = 2.997e8     # [m/s]
        hc_evnm = h * c * 6.241e18 * 1e9    # [J*m] to [eV*nm]
        hc_nm = h * c * 1e9     # [J*m] to [J*nm] 

        if (AIC_options["long_expfactor"]):
            try: A0 = float(self.A0_entry.get())         # [cm^-1 eV^-1/2] or [cm^-1 eV^-2]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid A0")
                return

            try: Eg = float(self.Eg_entry.get())                  # [eV]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid Eg")
                return

            try: wavelength = float(self.pulse_wavelength_entry.get())              # [nm]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid pulsed laser wavelength")
                return

            if AIC_options["incidence"] == "direct":
                alpha = A0 * (hc_evnm / wavelength - Eg) ** 0.5     # [cm^-1]

            elif AIC_options["incidence"] == "indirect":
                alpha = A0 * (hc_evnm / wavelength - Eg) ** 2

            else:
                self.write(self.ICtab_status, "Select \"direct\" or \"indirect\"")
                return

        else:
            try: 
                alpha = float(self.AIC_expfactor_entry.get()) # [cm^-1]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid α")
                return

        alpha_nm = alpha * 1e-7 # [cm^-1] to [nm^-1]

        if (AIC_options["power_mode"] == "power-spot"):
            try: 
                power = float(self.power_entry.get()) * 1e-6  # [uJ/s] to [J/s]
                spotsize = float(self.spotsize_entry.get()) * ((1e7) ** 2)     # [cm^2] to [nm^2]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing power or spot size")
                return

            try: wavelength = float(self.pulse_wavelength_entry.get())              # [nm]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid pulsed laser wavelength")
                return

            if (self.pulse_freq_entry.get() == "cw"):
                freq = 1
            else:
                try:
                    freq = float(self.pulse_freq_entry.get()) * 1e3    # [kHz] to [1/s]

                except ValueError:
                    self.write(self.ICtab_status, "Error: missing or invalid pulse frequency")
                    return

            # Note: add_AIC() automatically converts into TEDs units. For consistency add_AIC should really deposit values in common units.
            self.nanowire.param_dict["init_deltaN"].value = finite.pulse_laser_power_spotsize(power, spotsize, freq, wavelength, alpha_nm, self.nanowire.grid_x_nodes, hc=hc_nm)
        
        elif (AIC_options["power_mode"] == "density"):
            try: power_density = float(self.power_density_entry.get()) * 1e-6 * ((1e-7) ** 2)  # [uW / cm^2] to [J/s nm^2]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing power density")
                return

            try: wavelength = float(self.pulse_wavelength_entry.get())              # [nm]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid pulsed laser wavelength")
                return
            if (self.pulse_freq_entry.get() == "cw"):
                freq = 1
            else:
                try:
                    freq = float(self.pulse_freq_entry.get()) * 1e3    # [kHz] to [1/s]

                except ValueError:
                    self.write(self.ICtab_status, "Error: missing or invalid pulse frequency")
                    return

            self.nanowire.param_dict["init_deltaN"].value = finite.pulse_laser_powerdensity(power_density, freq, wavelength, alpha_nm, self.nanowire.grid_x_nodes, hc=hc_nm)
        
        elif (AIC_options["power_mode"] == "max-gen"):
            try: max_gen = float(self.max_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing max gen")
                return

            self.nanowire.param_dict["init_deltaN"].value = finite.pulse_laser_maxgen(max_gen, alpha_nm, self.nanowire.grid_x_nodes)
        

        elif (AIC_options["power_mode"] == "total-gen"):
            try: total_gen = float(self.total_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing total gen")
                return

            self.nanowire.param_dict["init_deltaN"].value = finite.pulse_laser_totalgen(total_gen, self.nanowire.total_length, alpha_nm, self.nanowire.grid_x_nodes)
        
        else:
            self.write(self.ICtab_status, "An unexpected error occurred while calculating the power generation params")
            return
        
        
        ## TODO: Make AIC deposit in common units, so this patch isn't required
        self.nanowire.param_dict["init_deltaN"].value *= self.convert_out_dict["init_deltaN"]
        ## Assuming that the initial distributions of holes and electrons are identical
        self.nanowire.param_dict["init_deltaP"].value = self.nanowire.param_dict["init_deltaN"].value

        self.update_IC_plot(plot_ID="AIC")
        self.HIC_listbox_currentparam = "init_deltaN"
        self.update_IC_plot(plot_ID="custom")
        self.HIC_listbox_currentparam = "init_deltaP"
        self.update_IC_plot(plot_ID="recent")
        self.using_AIC = True
        return

    ## Special functions for Parameter Toolkit:
    def add_HIC(self):
        # V2 update
        # Set the value of one of Nanowire's Parameters

        # TODO: This check may be deprecated
        if (self.HIC_list.__len__() > 0 and isinstance(self.HIC_list[0], str)):
            print("Something happened!")
            self.deleteall_HIC(False)

        try:
            self.set_init_x()

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        try:
            new_param_name = self.init_var_selection.get()
            if "[" in new_param_name: new_param_name = new_param_name[:new_param_name.find("[")]

            if (self.init_shape_selection.get() == "POINT"):

                if (float(self.HIC_lbound_entry.get()) < 0):
                    raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "POINT", float(self.HIC_lbound_entry.get()), -1, float(self.HIC_lvalue_entry.get()), -1)

            elif (self.init_shape_selection.get() == "FILL"):
                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) > self.nanowire.total_length):
                	raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) > float(self.HIC_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.HIC_lbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "FILL", float(self.HIC_lbound_entry.get()), float(self.HIC_rbound_entry.get()), float(self.HIC_lvalue_entry.get()), -1)

            elif (self.init_shape_selection.get() == "LINE"):
                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) > self.nanowire.total_length):
                	raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) > float(self.HIC_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "LINE", float(self.HIC_lbound_entry.get()), float(self.HIC_rbound_entry.get()), 
                                            float(self.HIC_lvalue_entry.get()), float(self.HIC_rvalue_entry.get()))

            elif (self.init_shape_selection.get() == "EXP"):
                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) > self.nanowire.total_length):
                    raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) > float(self.HIC_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "EXP", float(self.HIC_lbound_entry.get()), float(self.HIC_rbound_entry.get()), 
                                            float(self.HIC_lvalue_entry.get()), float(self.HIC_rvalue_entry.get()))

            else:
                raise Exception("Error: No init. type selected")

        except ValueError:
            self.write(self.ICtab_status, "Error: Missing Parameters")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return


        #self.HIC_list.append(new_param_rule)
        #self.HIC_listbox.insert(self.HIC_list.__len__() - 1, new_param_rule.get())

        self.nanowire.add_param_rule(new_param_name, new_param_rule)

        self.HIC_viewer_selection.set(new_param_name)
        self.update_paramrule_listbox(new_param_name)

        #self.recalc_HIC()
        if new_param_name == "init_deltaN" or new_param_name == "init_deltaP": self.using_AIC = False
        self.update_IC_plot(plot_ID="recent")
        return

    def refresh_paramrule_listbox(self):
        # The View button has two jobs: change the listbox to the new param and display a snapshot of it
        self.update_paramrule_listbox(self.HIC_viewer_selection.get())
        self.update_IC_plot(plot_ID="custom")
        return
    
    def update_paramrule_listbox(self, param_name):
        # Grab current param's rules from Nanowire and show them in the param_rule listbox
        if param_name == "":
            self.write(self.ICtab_status, "Select a parameter")
            return

        # 1. Clear the viewer
        self.hideall_HIC()

        # 2. Write in the new rules
        current_param_rules = self.nanowire.param_dict[param_name].param_rules
        self.HIC_listbox_currentparam = param_name

        for param_rule in current_param_rules:
            self.HIC_list.append(param_rule)
            self.HIC_listbox.insert(self.HIC_list.__len__() - 1, param_rule.get())

        
        self.write(self.ICtab_status, "")

        return

    # These two reposition the order of param_rules
    def moveup_HIC(self):
        currentSelectionIndex = self.HIC_listbox.curselection()[0]
        
        if (currentSelectionIndex > 0):
            # Two things must be done here for a complete swap:
            # 1. Change the order param rules appear in the box
            self.HIC_list[currentSelectionIndex], self.HIC_list[currentSelectionIndex - 1] = self.HIC_list[currentSelectionIndex - 1], self.HIC_list[currentSelectionIndex]
            self.HIC_listbox.delete(currentSelectionIndex)
            self.HIC_listbox.insert(currentSelectionIndex - 1, self.HIC_list[currentSelectionIndex - 1].get())
            self.HIC_listbox.selection_set(currentSelectionIndex - 1)

            # 2. Change the order param rules are applied when calculating Parameter's values
            self.nanowire.swap_param_rules(self.HIC_listbox_currentparam, currentSelectionIndex)
            self.update_IC_plot(plot_ID="recent")
        return

    def movedown_HIC(self):
        currentSelectionIndex = self.HIC_listbox.curselection()[0] + 1
        
        if (currentSelectionIndex < self.HIC_list.__len__()):
            self.HIC_list[currentSelectionIndex], self.HIC_list[currentSelectionIndex - 1] = self.HIC_list[currentSelectionIndex - 1], self.HIC_list[currentSelectionIndex]
            self.HIC_listbox.delete(currentSelectionIndex)
            self.HIC_listbox.insert(currentSelectionIndex - 1, self.HIC_list[currentSelectionIndex - 1].get())
            self.HIC_listbox.selection_set(currentSelectionIndex)
            
            self.nanowire.swap_param_rules(self.HIC_listbox_currentparam, currentSelectionIndex)
            self.update_IC_plot(plot_ID="recent")
        return

    def hideall_HIC(self, doPlotUpdate=True):
        # Wrapper - Call hide_HIC() until listbox is empty
        while (self.HIC_list.__len__() > 0):
            # These first two lines mimic user repeatedly selecting topmost HIC in listbox
            self.HIC_listbox.select_set(0)
            self.HIC_listbox.event_generate("<<ListboxSelect>>")

            self.hide_HIC()
        return

    def hide_HIC(self):
        # Remove user-selected param rule from box (but don't touch Nanowire's saved info)
        self.HIC_list.pop(self.HIC_listbox.curselection()[0])
        self.HIC_listbox.delete(self.HIC_listbox.curselection()[0])
        return
    
    def deleteall_HIC(self, doPlotUpdate=True):
        # Wrapper - Call delete_HIC until Nanowire's list of param_rules is empty for current param
        while (self.nanowire.param_dict[self.HIC_listbox_currentparam].param_rules.__len__() > 0):
            self.HIC_listbox.select_set(0)
            self.HIC_listbox.event_generate("<<ListboxSelect>>")

            self.delete_HIC()
        return

    def delete_HIC(self):
        # Remove user-selected param rule from box AND from Nanowire's list of param_rules
        if (self.nanowire.param_dict[self.HIC_listbox_currentparam].param_rules.__len__() > 0):
            try:
                self.nanowire.remove_param_rule(self.HIC_listbox_currentparam, self.HIC_listbox.curselection()[0])
                self.hide_HIC()
                self.update_IC_plot(plot_ID="recent")
            except IndexError:
                self.write(self.ICtab_status, "No rule selected")
                return
        return

    # Fill IC arrays using list from .txt file
    def add_EIC(self):
        try:
            self.set_init_x()

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return
        
        warning_flag = False
        var = self.EIC_var_selection.get()
        is_edge = self.nanowire.param_dict[var].is_edge
        
        valuelist_filename = tk.filedialog.askopenfilename(initialdir="", title="Select Values from text file", filetypes=[("Text files","*.txt")])
        if valuelist_filename == "": # If no file selected
            return

        IC_values_list = []
        with open(valuelist_filename, 'r') as ifstream:
            for line in ifstream:
                if (line.strip('\n') == "" or "#" in line): continue

                else: IC_values_list.append(line.strip('\n'))

           
        temp_IC_values = np.zeros(self.nanowire.grid_x_nodes.__len__()) if not is_edge else np.zeros(self.nanowire.grid_x_edges.__len__())

        try:
            IC_values_list.sort(key = lambda x:float(x[0:x.find('\t')]))
            
            if IC_values_list.__len__() < 2: # if not enough points in list
                raise ValueError
        except:
            self.write(self.ICtab_status, "Error: Unable to read point list")
            return
        
    
        for i in range(IC_values_list.__len__() - 1):
            try:
                first_valueset = extract_values(IC_values_list[i], '\t') #[x1, y(x1)]
                second_valueset = extract_values(IC_values_list[i+1], '\t') #[x2, y(x2)]
                
            except ValueError:
                self.write(self.ICtab_status, "Warning: Unusual point list content")
                warning_flag = True

            # Linear interpolate from provided EIC list to specified grid points
            lindex = finite.toIndex(first_valueset[0], self.nanowire.dx, self.nanowire.total_length, is_edge)
            rindex = finite.toIndex(second_valueset[0], self.nanowire.dx, self.nanowire.total_length, is_edge)
            
            if (first_valueset[0] - finite.toCoord(lindex, self.nanowire.dx, is_edge) >= self.nanowire.dx / 2): lindex += 1

            intermediate_x_indices = np.arange(lindex, rindex + 1, 1)

            for j in intermediate_x_indices: # y-y0 = (y1-y0)/(x1-x0) * (x-x0)
                try:
                    if (second_valueset[0] > self.nanowire.total_length): raise IndexError
                    temp_IC_values[j] = first_valueset[1] + (finite.toCoord(j, self.nanowire.dx) - first_valueset[0]) * (second_valueset[1] - first_valueset[1]) / (second_valueset[0] - first_valueset[0])
                except IndexError:
                    self.write(self.ICtab_status, "Warning: some points out of bounds")
                    warning_flag = True
                except:
                    temp_IC_values[j] = 0
                    warning_flag = True
                
        if var == "init_deltaN" or var == "init_deltaP": self.using_AIC = False
        
        self.HIC_listbox_currentparam = var
        self.deleteall_HIC()
        self.nanowire.param_dict[var].value = temp_IC_values
        self.update_IC_plot(plot_ID="EIC", warn=warning_flag)
        self.update_IC_plot(plot_ID="recent", warn=warning_flag)
        return

    def update_IC_plot(self, plot_ID, warn=False):
        # V2 update: can now plot any parameter
        # Plot 2 is for recently changed parameter while plot 1 is for user-selected views

        if plot_ID=="recent": plot = self.recent_param_subplot
        elif plot_ID=="custom": plot = self.custom_param_subplot
        elif plot_ID=="AIC": plot = self.AIC_subplot
        elif plot_ID=="EIC": plot = self.EIC_subplot
        plot.cla()
        plot.set_yscale('log')

        if plot_ID=="AIC": param_name="init_deltaN"
        else: param_name = self.HIC_listbox_currentparam
        
        param_obj = self.nanowire.param_dict[param_name]
        grid_x = self.nanowire.grid_x_edges if param_obj.is_edge else self.nanowire.grid_x_nodes
        val_array = param_obj.value
        # Support for constant value shortcut: temporarily create distribution
        # simulating filling across nanowire with that value
        if not isinstance(val_array, np.ndarray):
            val_array = np.ones(grid_x.__len__()) * val_array
        max_val = np.amax(param_obj.value)
        

        plot.set_ylim((max_val + 1e-30) * 1e-12, (max_val + 1e-30) * 1e4)


        if self.check_symmetric.get():
            plot.plot(np.concatenate((-np.flip(grid_x), grid_x), axis=0), np.concatenate((np.flip(val_array), val_array), axis=0), label=param_name)

            ymin, ymax = plot.get_ylim()
            plot.fill([-grid_x[-1], 0, 0, -grid_x[-1]], [ymin, ymin, ymax, ymax], 'b', alpha=0.1, edgecolor='r')
        else:
            plot.plot(grid_x, val_array, label=param_name)

        plot.set_xlabel("x [nm]")
        plot.set_ylabel("{} {}".format(param_name, param_obj.units))
        
        if plot_ID=="recent": 
            self.update_system_summary()
            
            plot.set_title("Recently Changed: {}".format(param_name))
            self.recent_param_fig.tight_layout()
            self.recent_param_fig.canvas.draw()
        elif plot_ID=="custom": 
            plot.set_title("Snapshot: {}".format(param_name))
            self.custom_param_fig.tight_layout()
            self.custom_param_fig.canvas.draw()
        elif plot_ID=="AIC": 
            plot.set_title("Recent AIC")
            self.AIC_fig.tight_layout()
            self.AIC_fig.canvas.draw()
        elif plot_ID=="EIC": 
            plot.set_title("Recent list upload")
            self.EIC_fig.tight_layout()
            self.EIC_fig.canvas.draw()

        if not warn: self.write(self.ICtab_status, "Initial Condition Updated")
        return

    ## Initial Condition I/O

    def create_batch_init(self):
        try:
            batch_values = {}
            for batchable in self.batchables_array:
                if batchable.param_name.get():
                    batch_values[batchable.param_name.get()] = []
                    
            for batchable in self.batchables_array:
                if batchable.param_name.get():
                    batch_values[batchable.param_name.get()] += extract_values(batchable.tk_entrybox.get(), ' ')
            
            if not batch_values: # If no batch params were selected
                raise ValueError
        except ValueError:
            self.write(self.batch_status, "Error: Invalid batch values")
            return

        print(batch_values)

        try:
            batch_dir_name = self.batch_name_entry.get()
            if batch_dir_name == "": raise OSError("Error: Batch folder must have a name")
            if not check_valid_filename(batch_dir_name): raise OSError("File names may not contain certain symbols such as ., <, >, /, \\, *, ?, :, \", |")
        except Exception as e:
            self.write(self.batch_status, e)
            return

        try:
            os.mkdir("{}\\{}".format(self.default_dirs["Initial"], batch_dir_name))
        except FileExistsError:
            self.write(self.batch_status, "Error: {} folder already exists".format(batch_dir_name))
            return
        
        # Record the original values of the Nanowire, so we can restore them after the batch algo finishes
        original_param_values = {}
        for param in self.nanowire.param_dict:
            original_param_values[param] = self.nanowire.param_dict[param].value
        
        
        # This algorithm was shamelessly stolen from our bay.py script...
        # TODO: ...and should really, really be a standalone function!!
        # def get_all_combinations(param_values)
                
        batch_combinations = []
        batch_param_names = list(batch_values.keys())
        
        iterable_param_indexes = {}
        iterable_param_lengths = {}
        for param in batch_param_names:
            iterable_param_indexes[param] = 0
            iterable_param_lengths[param] = batch_values[param].__len__()
        
        pivot_index = batch_param_names.__len__() - 1

        current_params = dict(batch_values)
        # Create a list of all combinations of parameter values
        while(pivot_index >= 0):

            # Generate the next parameter set using lists of indices stored in the helper structures
            for iterable_param in batch_param_names:
                current_params[iterable_param] = batch_values[iterable_param][iterable_param_indexes[iterable_param]]

            batch_combinations.append(dict(current_params))

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
            pivot_index = batch_param_names.__len__() - 1
            while (pivot_index >= 0 and iterable_param_indexes[batch_param_names[pivot_index]] == iterable_param_lengths[batch_param_names[pivot_index]] - 1):
                pivot_index -= 1

            iterable_param_indexes[batch_param_names[pivot_index]] += 1

            for i in range(pivot_index + 1, batch_param_names.__len__()):
                iterable_param_indexes[batch_param_names[i]] = 0
                
        # Apply each combination to Nanowire, going through AIC if necessary

        for batch_set in batch_combinations:
            filename = ""
            for param in batch_set:
                filename += str("__{}_{:.4e}".format(param, batch_set[param]))
                
                if param in self.analytical_entryboxes_dict:
                    self.enter(self.analytical_entryboxes_dict[param], str(batch_set[param]))
                    
                else:
                    self.nanowire.param_dict[param].value = batch_set[param]

                
            if self.using_AIC: self.add_AIC()
                
            try:
                self.write_init_file("{}\\{}\\{}.txt".format(self.default_dirs["Initial"], batch_dir_name, filename))
            except:
                self.write(self.batch_status, "Error: failed to create batch file {}".format(filename))
                
        # Restore the original values of Nanowire
        for param in self.nanowire.param_dict:
            self.nanowire.param_dict[param].value = original_param_values[param]
        
        self.write(self.batch_status, "Batch \"{}\" created successfully".format(batch_dir_name))

        return

	# Wrapper for write_init_file() - this one is for IC files user saves from the Initial tab and is called when the Save button is clicked
    def save_ICfile(self):
        try:

            new_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["Initial"], title="Save IC text file", filetypes=[("Text files","*.txt")])
            
            if new_filename == "": return

            if new_filename.endswith(".txt"): new_filename = new_filename[:-4]
            self.write_init_file(new_filename + ".txt")

        except ValueError as uh_Oh:
            print(uh_Oh)
            
        return

    def write_init_file(self, newFileName, dir_name=""):

        try:
            with open(newFileName, "w+") as ofstream:
                print(dir_name + newFileName + " opened successfully")

                # We don't really need to note down the time of creation, but it could be useful for interaction with other programs.
                ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
                ofstream.write("$ Space Grid:\n")
                ofstream.write("Total_length: {}\n".format(self.nanowire.total_length))
                ofstream.write("Node_width: {}\n".format(self.nanowire.dx))
                
                ofstream.write("$ System Parameters:\n")
                
                # Saves occur as-is: any missing parameters are saved with whatever default value Nanowire gives them
                for param in self.nanowire.param_dict:
                    param_values = self.nanowire.param_dict[param].value
                    if isinstance(param_values, np.ndarray):
                        # Write the array in a more convenient format
                        ofstream.write("{}: {:.8e}".format(param, param_values[0]))
                        for value in param_values[1:]:
                            ofstream.write("\t{:.8e}".format(value))
                            
                        ofstream.write('\n')
                    else:
                        # The param value is just a single constant
                        ofstream.write("{}: {}\n".format(param, param_values))

                ofstream.write("$ System Flags:\n")
                
                for flag in self.nanowire.flags_dict:
                    ofstream.write("{}: {}\n".format(flag, self.sys_flag_dict[flag].tk_var.get()))
                
        except OSError:
            self.write(self.ICtab_status, "Error: failed to create IC file")
            return

        self.write(self.ICtab_status, "IC file generated")
        return

    # Wrapper for load_ICfile with user selection from IC tab
    def select_init_file(self):
        self.IC_file_name = tk.filedialog.askopenfilename(initialdir = self.default_dirs["Initial"], title="Select IC text files", filetypes=[("Text files","*.txt")])
        if self.IC_file_name == "": return # If user closes dialog box without selecting a file

        self.load_ICfile()
        return

    def load_ICfile(self, cycle_through_IC_plots=True):
        warning_flag = False

        try:
            print("Poked file: {}".format(self.IC_file_name))
            with open(self.IC_file_name, 'r') as ifstream:
                init_param_values_dict = {}

                flag_values_dict = {}
                total_length = 0
                dx = 0

                initFlag = 0
                file_is_valid = False
                # Extract parameters, ICs
                for line in ifstream:
                    # Attempt to determine if loaded file is valid initial condition.
                    # i.e. if the user mis-clicks on a non-TEDs file
                    if not(file_is_valid):
                        if ("$$ INITIAL CONDITION FILE CREATED ON" in line):
                            file_is_valid = True
                            continue
                        else:
                            raise OSError("Error: unable to read selected file")

                    if ("#" in line) or (line.strip('\n').__len__() == 0):
                        continue

                    # There are three "$" markers in an IC file: "Space Grid", "System Parameters" and "System Flags"
                    # each corresponding to a different section of the file
                    
                    elif "$ Space Grid" in line:
                        print("Now searching for space grid parameters: total length and dx")
                        initFlag = 1
                        
                    elif "$ System Parameters" in line:
                        print("Now searching for system parameters...")
                        initFlag = 2
                        
                    elif "$ System Flags" in line:
                        print("Now searching for flag values...")
                        initFlag = 3

                    elif (initFlag == 1):
                        line = line.strip('\n')
                        if line[0:line.find(':')] == "Total_length":
                            total_length = (line[line.find(' ') + 1:])
                            
                        elif line[0:line.find(':')] == "Node_width":
                            dx = (line[line.find(' ') + 1:])
                            
                    elif (initFlag == 2):
                        line = line.strip('\n')
                        init_param_values_dict[line[0:line.find(':')]] = (line[line.find(' ') + 1:])


                    elif (initFlag == 3):
                        line = line.strip('\n')
                        flag_values_dict[line[0:line.find(':')]] = (line[line.find(' ') + 1:])

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        ## At this point everything from the file has been read. 
        # Whether those values are valid is another story, but having a valid space grid is essential.
        try:
            total_length = float(total_length)
            dx = float(dx)
            if (total_length <= 0) or (dx <= 0) or (total_length < dx):
                raise ValueError
        except:
            self.write(self.ICtab_status, "Error: invalid space grid")
            return
        
        # Clear values in any IC generation areas; this is done to minimize ambiguity between IC's that came from the recently loaded file and any other values that may exist on the GUI
        for key in self.analytical_entryboxes_dict:
            self.enter(self.analytical_entryboxes_dict[key], "")
            
        for param in self.nanowire.param_dict:
            self.HIC_listbox_currentparam = param
            
            self.update_paramrule_listbox(param)            
            self.deleteall_HIC()
            
            self.nanowire.param_dict[param].value = 0

        self.set_thickness_and_dx_entryboxes(state='unlock')
        self.nanowire.total_length = None
        self.nanowire.dx = None
        self.nanowire.grid_x_edges = []
        self.nanowire.grid_x_nodes = []
        self.nanowire.spacegrid_is_set = False

        try:
            self.enter(self.thickness_entry, total_length)
            self.enter(self.dx_entry, dx)
            self.set_init_x()
            
        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        for flag in self.nanowire.flags_dict:
            # Again, we don't bother with Nanowire's flags_dict until it's time to simulate
            # All we need to do here is mark the appropriate GUI elements as selected
            self.sys_flag_dict[flag].tk_var.set(flag_values_dict[flag])
            
        for param in self.nanowire.param_dict:
            new_value = init_param_values_dict[param]
            try:
                if '\t' in new_value:
                    self.nanowire.param_dict[param].value = np.array(extract_values(new_value, '\t'))
                else: self.nanowire.param_dict[param].value = float(new_value)
                
                self.HIC_listbox_currentparam = param
                if cycle_through_IC_plots: self.update_IC_plot(plot_ID="recent")
            except:
                print("Warning: could not apply value for param: {}".format(param))
                warning_flag += 1
                
        self.using_AIC = False
        
        if not warning_flag: self.write(self.ICtab_status, "IC file loaded successfully")
        else: self.write(self.ICtab_status, "IC file loaded with {} issue(s); see console".format(warning_flag))
        return

    # Data I/O

    def export_plot(self, from_integration):

        if from_integration:
            datagroup = self.integration_plots[0].datagroup
            plot_info = self.integration_plots[0]
            if datagroup.size() == 0: return
            if plot_info.mode == "Current time step": 
                paired_data = [[datagroup.datasets[key].grid_x, datagroup.datasets[key].data * self.convert_out_dict[datagroup.type]] for key in datagroup.datasets]

                # TODO: Write both of these values with their units
                header = "{}, {}".format(plot_info.x_param, datagroup.type)

            else: # if self.I_plot.mode == "All time steps"
                raw_data = np.array([datagroup.datasets[key].data * self.convert_out_dict[datagroup.type] for key in datagroup.datasets])
                grid_x = np.reshape(plot_info.global_gridx, (1,plot_info.global_gridx.__len__()))
                paired_data = np.concatenate((grid_x, raw_data), axis=0).T
                header = "Time [ns],"
                for key in datagroup.datasets:
                    header += datagroup.datasets[key].tag().replace("Δ", "") + ","

        else:
            plot_ID = self.active_analysisplot_ID.get()
            if self.analysis_plots[plot_ID].datagroup.size() == 0: return
            paired_data = self.analysis_plots[plot_ID].datagroup.build()
            # We need some fancy footwork using itertools to transpose a non-rectangular array
            paired_data = np.array(list(map(list, itertools.zip_longest(*paired_data, fillvalue=-1))))
            header = "".join(["x [nm]," + self.analysis_plots[plot_ID].datagroup.datasets[key].filename + "," for key in self.analysis_plots[plot_ID].datagroup.datasets])

        PL_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["PL"], title="Save data", filetypes=[("csv (comma-separated-values)","*.csv")])
        
        # Export to .csv
        # TODO: Fix export for over time mode
        if not (PL_filename == ""):
            try:
                if PL_filename.endswith(".csv"): PL_filename = PL_filename[:-4]
                #np.savetxt("{}.csv".format(PL_filename), np.vstack((self.grid_xaxis, data)).transpose(), fmt='%.4e', delimiter=',', header='t,PL from [nm],' + str(self.integration_lbound) + ',to,' + str(self.integration_ubound))
                np.savetxt("{}.csv".format(PL_filename), paired_data, fmt='%.4e', delimiter=',', header=header)
                self.write(self.analysis_status, "Export complete")
            except PermissionError:
                self.write(self.analysis_status, "Error: unable to access PL export destination")
        
        return

    def export_for_bayesim(self):
        # Note: DO NOT convert_out any of these values - bayesim models are created in TEDs units.
        datagroup = self.integration_plots[0].datagroup
        plot_info = self.integration_plots[0]
        if datagroup.size() == 0: return
            
        if (plot_info.mode == "All time steps"):
            if self.bay_mode.get() == "obs":
                for key in datagroup.datasets:  # For each curve on the integration plot
                    raw_data = datagroup.datasets[key].data
                    grid_x = plot_info.global_gridx   # grid_x refers to what is on the x-axis, which in this case is technically 'time'
                    unc = raw_data * 0.1
                    full_data = np.vstack((grid_x, raw_data, unc)).T
                    full_data = pd.DataFrame.from_records(data=full_data,columns=['time', datagroup.type, 'uncertainty'])
                    
                    #FIXME: dd.save has no visible file overwrite handler
                    # If the file name already exists, dd.save will simply not save anything
                    dd.save("{}//{}.h5".format(self.default_dirs["PL"], datagroup.datasets[key].tag()), full_data)

            elif self.bay_mode.get() == "model":
                active_bay_params = []
                for param in self.check_bay_params:
                    if self.check_bay_params[param].get(): active_bay_params.append(param)

                is_first = True
                for key in datagroup.datasets:
                    raw_data = datagroup.datasets[key].data
                    grid_x = plot_info.global_gridx
                    paired_data = np.vstack((grid_x, raw_data))

                    for param in active_bay_params:
                        param_column = np.ones((1,raw_data.__len__())) * datagroup.datasets[key].params_dict[param] * self.convert_out_dict[param]
                        paired_data = np.concatenate((param_column, paired_data), axis=0)

                    paired_data = paired_data.T

                    if is_first:
                        full_data = paired_data
                        is_first = False

                    else:
                        full_data = np.concatenate((full_data, paired_data), axis=0)

                panda_columns = []
                for param in active_bay_params:
                    panda_columns.insert(0,param)

                panda_columns.append('time')
                panda_columns.append(datagroup.type)

                full_data = pd.DataFrame.from_records(data=full_data, columns=panda_columns)
                
                new_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["PL"], title="Save Bayesim Model", filetypes=[("HDF5 Archive","*.h5")])
                if new_filename == "": return

                if not new_filename.endswith(".h5"): new_filename += ".h5"

                dd.save(new_filename, full_data)
        else:
            print("WIP =(")

        self.write(self.analysis_status, "Bayesim export complete")
        return

nb = Notebook("ted")
nb.run()

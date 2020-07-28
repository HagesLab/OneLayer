#################################################
# Transient Electron Dynamics Simulator
# Model photoluminescent behavior in one-dimensional nanowire
# Last modified: July 8, 2020
# Author: Calvin Fai, Charles Hages
# Contact:
################################################# 

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plot
import matplotlib.backends.backend_tkagg as tkagg
from scipy import integrate as intg

import tkinter.filedialog
import tkinter as tk
from tkinter import ttk # ttk is a sort of expansion pack to Tkinter, featuring additional elements and features.
import time
import datetime
import os
import tables
import itertools
from functools import partial # This lets us pass params to functions called by tkinter buttons
import finite
import csv

import pandas as pd # For bayesim compatibility
import bayesim.hdf5io as dd

np.seterr(divide='raise', over='warn', under='warn', invalid='raise')
class Initial_Condition:
    # Object containing group of parameters used to create heuristic initial conditions
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

class Data_Set:
    # Object containing all the metadata required to plot and integrate saved data sets
    def __init__(self, data, grid_x, node_x, edge_x, mu_n, mu_p, n0, p0, total_x, dx, B, tau_N, tau_P, Sf, Sb, temperature, rel_permitivity, ext_E_field, theta, alpha, delta, frac_emitted, total_t, dt, ignored_alpha, type, filename, show_index):
        self.data = data            # The actual data e.g. N(x,t) associated with this set
        self.grid_x = grid_x        # Array of x-coordinates at which data was calculated - plotter uses these as x values
        self.node_x = node_x        # Array of x-coordinates corresponding to system nodes - needed to generate initial condition from data
        self.edge_x = edge_x        # node_x but for system node edges - also needed to regenerate ICs

        # node_x and grid_x will usually be identical, unless the data is a type (like E-field) that exists on edges
        # There's a little optimization that can be made here because grid_x will either be identical to node_x or edge_x, but that makes the code harder to follow

        self.num_tsteps = int(0.5 + total_t / dt)
        self.type = type            # String identifying variable the data is for e.g. N, P
        self.filename = filename    # String identifying file from which data set was read
        self.show_index = show_index# Time step number data belongs to
		
		# Python dictionaries are otherwise known as hash tables in other languages - each value in the table is assigned a key we can use to access it
        self.params_dict = {"Mu_N": mu_n, "Mu_P": mu_p, "N0": n0, "P0": p0, "Thickness": total_x, "dx": dx, "B": B, "Tau_N": tau_N, "Tau_P": tau_P, "Sf": Sf, \
            "Sb": Sb, "Temperature": temperature, "Rel-Permitivity": rel_permitivity, "Ext_E-Field": ext_E_field, "Theta": theta, "Alpha": alpha, "Delta": delta, "Frac-Emitted": frac_emitted, \
            "Total-Time": total_t,"dt": dt,"ignore_alpha": ignored_alpha}
        return

    def tag(self):
        return self.filename + "_" + self.type

    def build(self):
        return np.vstack((self.grid_x, self.data))

class Data_Group:
    # Object containing list of Data_Sets; there is one Data_Group for each of the two small plots on analysis tab
    def __init__(self, ID):
        self.ID = ID
        self.type = "None"
        self.dt = -1
        self.total_t = -1
        self.datasets = {}
        return

    def set_type(self, new_type):
        self.type = new_type
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
            raise ValueError("Cannot plot selected data sets: dt or total t mismatch")

        return

    def get_data(self):
        return np.array(self.datasets.values())

    def build(self):
        #result = [item for item in self.datasets[key].build() for key in self.datasets]
        result = []
        for key in self.datasets:
            result.append(self.datasets[key].grid_x)
            result.append(self.datasets[key].data)
        return result

    def get_max_x(self):
        return np.amax([self.datasets[tag].params_dict["Thickness"] for tag in self.datasets])

    def get_maxtime(self):
        return np.amax([self.datasets[tag].params_dict["Total-Time"] for tag in self.datasets])

    def get_maxnumtsteps(self):
        return np.amax([self.datasets[tag].num_tsteps for tag in self.datasets])

    def get_maxval(self):
        return np.amax([np.amax(self.datasets[tag].data) for tag in self.datasets])

    def size(self):
        return len(self.datasets)

    def clear(self):
        self.datasets.clear()
        return

class Plot_State:
    # Object containing variables needed for each small plot on analysis tab
	# This is really a wrapper that enhances interactions between the Data_Group object and the embedded plot
    # There are currently two but hopefully this class helps simplify adding more
    def __init__(self, ID, plot_obj=None):
        self.ID = ID
        self.plot_obj = plot_obj
        self.xaxis_type = 'linear'
        self.yaxis_type = 'log'
        self.xlim = (-1,-1)
        self.ylim = (-1,-1)
        self.fig_ID = -1 # The plot.figure(#) the object is associated with
        self.time_index = 0
        self.datagroup = Data_Group(ID)
        self.data_filenames = []
        self.display_legend = True
        return

    def remove_duplicate_filenames(self):
        # Sets, unlike arrays, contain at most one copy of each item. Forcing an array into a set like this
        # is a fast way to scrub duplicate entries, which is needed because we don't want to waste time
        # plotting the same data set multiple times.
        # Unfortunately, sets are not indexable, so we convert back into arrays to regain index access ability.
            
        self.data_filenames = list(set(self.data_filenames))
        return

    def add_time_index(self, offset):
        self.time_index += offset
        if self.time_index < 0: self.time_index = 0
        if self.time_index > self.datagroup.get_maxnumtsteps(): 
            self.time_index = self.datagroup.get_maxnumtsteps()
        return

class I_Set:
    # I_Sets are similar to Data_Sets but store exclusively the integrated data generated from Data_Sets
    def __init__(self, I_data, grid_x, params_dict, type, filename):
        self.I_data = I_data    # This is usually PL values
        self.grid_x = grid_x    # This is usually time values
        self.params_dict = params_dict
        self.type = type
        self.filename = filename
        return

    def tag(self):
        return self.filename + "_" + self.type

class I_Group:
    # A batch of I_sets generated from the same Integrate operation
    def __init__(self):
        self.I_sets = {}
        self.type = "None"      # This is usually "PL"
        self.mode = ""
        self.x_param = "None"   # This is usually "Time"
        self.global_gridx = None    # In some modes of operation every I_Set will have the same grid_x
        self.xaxis_type = 'linear'
        self.yaxis_type = 'log'
        self.xlim = (-1,-1)
        self.ylim = (-1,-1)
        self.display_legend = True
        return

    def set_type(self, new_type):
        self.type = new_type
        return

    def add(self, new_set):
        if (len(self.I_sets) == 0): # Allow the first set in to set the type restriction
           self.type = new_set.type

        # Only allow datasets with identical time step size and total time - this should always be the case after any integration; otherwise something has gone wrong
        if (self.type == new_set.type):
            self.I_sets[new_set.tag] = new_set

        return

    def get_maxval(self):
        return np.amax([np.amax(self.I_sets[tag].I_data) for tag in self.I_sets])

    def size(self):
        return len(self.I_sets)

    def clear(self):
        self.I_sets.clear()
        return

class Flag:
    # This class exists to solve a little problem involving tkinter checkbuttons: we get the value of a checkbutton using its tk.IntVar() 
    # but we interact with the checkbutton using the actual tk.CheckButton() element
    # So wrap both of those together in a single object and call it a day
    def __init__(self, tk_element, tk_var, value=0):
        self.tk_element = tk_element
        self.tk_var = tk_var
        self.value = value

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
        # Multiply the parameter values TEDs is using by the corresponding coefficient in this dictionary to convert back into common units
        self.convert_out_dict = {"Mu_N": ((1e-7) ** 2) / (1e-9), "Mu_P": ((1e-7) ** 2) / (1e-9), "N0": ((1e7) ** 3), "P0": ((1e7) ** 3), "Thickness": 1, "dx": 1, \
                        "B": ((1e-7) ** 3) / (1e-9), "Tau_N": 1, "Tau_P": 1, "Sf": (1e-7) / (1e-9), "Sb": (1e-7) / (1e-9), "Temperature": 1, "Rel-Permitivity": 1, "Ext_E-Field": 1e3, \
                        "Theta": 1e7, "Alpha": 1e7, "Delta": 1, "Frac-Emitted": 1}

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
                                "Delta": 1, "Frac-Emitted": 1}

        # Tkinter elements require special variables to extract user input
        self.check_ignore_recycle = tk.IntVar()
        self.check_do_ss = tk.BooleanVar()
        self.check_reset_params = tk.BooleanVar()
        self.check_reset_inits = tk.BooleanVar()
        self.check_display_legend = tk.BooleanVar()

        self.calculate_init_material_expfactor = tk.IntVar()
        self.AIC_stim_mode = tk.StringVar()
        self.AIC_gen_power_mode = tk.StringVar()

        self.init_shape_selection = tk.StringVar()
        self.init_var_selection = tk.StringVar()
        self.EIC_var_selection = tk.StringVar()
        self.display_selection = tk.StringVar()

        self.bay_params = {"Mu_N":tk.BooleanVar(), "Mu_P":tk.BooleanVar(), "N0":tk.BooleanVar(), "P0":tk.BooleanVar(),
                        "B":tk.BooleanVar(), "Tau_N":tk.BooleanVar(), "Tau_P":tk.BooleanVar(), "Sf":tk.BooleanVar(), \
                        "Sb":tk.BooleanVar(), "Temperature":tk.BooleanVar(), "Rel-Permitivity":tk.BooleanVar(), \
                        "Theta":tk.BooleanVar(), "Alpha":tk.BooleanVar(), "Delta":tk.BooleanVar(), "Frac-Emitted":tk.BooleanVar()}
        
        self.bay_mode = tk.StringVar(value="model")

        # Flags and containers for IC arrays
        self.HIC_list = []
        self.IC_file_list = None
        self.init_N = None
        self.init_P = None
        self.init_E_field = None
        self.init_Ec = None
        self.init_Chi = None
        self.IC_file_name = ""
        self.IC_is_AIC = False

        self.sim_N = None
        self.sim_P = None
        self.sim_E_field = None

        self.carry_include_N = tk.IntVar()
        self.carry_include_P = tk.IntVar()
        self.carry_include_E_field = tk.IntVar()
        
        # Helpers, flags, and containers for analysis plots
        self.analysis_plots = [Plot_State(ID=0), Plot_State(ID=1)]
        self.I_plot = I_Group()
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
        self.batch_popup_isopen = False
        self.overwrite_popup_isopen = False
        self.integration_popup_isopen = False
        self.integration_getbounds_popup_isopen = False
        self.PL_xaxis_popup_isopen = False
        self.change_axis_popup_isopen = False
        self.plotter_popup_isopen = False
        self.IC_carry_popup_isopen = False
        self.bayesim_popup_isopen = False

        self.root.config(menu=self.menu_bar)
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
        self.tab_analytical_init = tk.Frame(self.tab_inputs)
        self.tab_rules_init = tk.Frame(self.tab_inputs)
        self.tab_explicit_init = tk.Frame(self.tab_inputs)

        self.line_sep_style = tk.ttk.Style()
        self.line_sep_style.configure("Grey Bar.TSeparator", background='#000000', padding=160)

        self.header_style = tk.ttk.Style()
        self.header_style.configure("Header.TLabel", background='#D0FFFF',highlightbackground='#000000')

		# We use the grid location specifier for general placement and padx/pady for fine-tuning
		# The other two options are the pack specifier, which doesn't really provide enough versatility,
		# and absolute coordinates, which maximize versatility but are a pain to adjust manually.
        self.load_ICfile_button = tk.Button(self.tab_inputs, text="Load", command=self.select_init_file)
        self.load_ICfile_button.grid(row=0,column=0, padx=(0,60), pady=(24,0))

        self.save_ICfile_button = tk.Button(self.tab_inputs, text="Save", command=self.save_ICfile)
        self.save_ICfile_button.grid(row=0,column=1, padx=(0,30), pady=(24,0))

        self.DEBUG_BUTTON = tk.Button(self.tab_inputs, text="debug", command=self.DEBUG)
        self.DEBUG_BUTTON.grid(row=0,column=0,columnspan=2, pady=(24,0))

        self.system_params_head = tk.ttk.Label(self.tab_inputs, text="System Parameters",style="Header.TLabel")
        self.system_params_head.grid(row=1, column=0,columnspan=2)

        self.N_mobility_label = tk.Label(self.tab_inputs, text="N Mobility [cm^2 / V s]")
        self.N_mobility_label.grid(row=2, column=0)

        self.N_mobility_entry = tk.Entry(self.tab_inputs, width=9)
        self.N_mobility_entry.grid(row=2,column=1)

        self.P_mobility_label = tk.Label(self.tab_inputs, text="P Mobility [cm^2 / V s]")
        self.P_mobility_label.grid(row=3, column=0)

        self.P_mobility_entry = tk.Entry(self.tab_inputs, width=9)
        self.P_mobility_entry.grid(row=3,column=1)

        self.n0_label = tk.Label(self.tab_inputs, text="n0 [cm^-3]")
        self.n0_label.grid(row=4,column=0)

        self.n0_entry = tk.Entry(self.tab_inputs, width=9)
        self.n0_entry.grid(row=4,column=1)

        self.p0_label = tk.Label(self.tab_inputs, text="p0 [cm^-3]")
        self.p0_label.grid(row=5,column=0)

        self.p0_entry = tk.Entry(self.tab_inputs, width=9)
        self.p0_entry.grid(row=5,column=1)

        self.B_label = tk.Label(self.tab_inputs, text="B (cm^3 / s)")
        self.B_label.grid(row=6,column=0)

        self.B_entry = tk.Entry(self.tab_inputs, width=9)
        self.B_entry.grid(row=6,column=1)

        self.tauN_label = tk.Label(self.tab_inputs, text="Tau_n (ns)")
        self.tauN_label.grid(row=7, column=0)

        self.tauN_entry = tk.Entry(self.tab_inputs, width=9)
        self.tauN_entry.grid(row=7, column=1)

        self.tauP_label = tk.Label(self.tab_inputs, text="Tau_p (ns)")
        self.tauP_label.grid(row=8, column=0)

        self.tauP_entry = tk.Entry(self.tab_inputs, width=9)
        self.tauP_entry.grid(row=8, column=1)

        self.Sf_label = tk.Label(self.tab_inputs, text="Sf (cm / s)")
        self.Sf_label.grid(row=9, column=0)

        self.Sf_entry = tk.Entry(self.tab_inputs, width=9)
        self.Sf_entry.grid(row=9, column=1)

        self.Sb_label = tk.Label(self.tab_inputs, text="Sb (cm / s)")
        self.Sb_label.grid(row=10, column=0)

        self.Sb_entry = tk.Entry(self.tab_inputs, width=9)
        self.Sb_entry.grid(row=10, column=1)

        self.temperature_label = tk.Label(self.tab_inputs, text="Temperature (K)")
        self.temperature_label.grid(row=11, column=0)

        self.temperature_entry = tk.Entry(self.tab_inputs, width=9)
        self.temperature_entry.grid(row=11, column=1)

        self.rel_permitivity_label = tk.Label(self.tab_inputs, text="Rel. Permitivity")
        self.rel_permitivity_label.grid(row=12, column=0)

        self.rel_permitivity_entry = tk.Entry(self.tab_inputs, width=9)
        self.rel_permitivity_entry.grid(row=12, column=1)

        self.ext_efield_label = tk.Label(self.tab_inputs, text="External E-field [V/um]")
        self.ext_efield_label.grid(row=13,column=0)

        self.ext_efield_entry = tk.Entry(self.tab_inputs, width=9)
        self.ext_efield_entry.grid(row=13,column=1)

        self.special_var_head = tk.ttk.Label(self.tab_inputs, text="Photon behavior parameters", style="Header.TLabel")
        self.special_var_head.grid(row=14,column=0,columnspan=2)

        self.theta_label = tk.Label(self.tab_inputs, text="Theta Cof. [cm^-1]")
        self.theta_label.grid(row=15,column=0)

        self.theta_entry = tk.Entry(self.tab_inputs, width=9)
        self.theta_entry.grid(row=15,column=1)

        self.alpha_label = tk.Label(self.tab_inputs, text="Alpha Cof. [cm^-1]")
        self.alpha_label.grid(row=16,column=0)

        self.alpha_entry = tk.Entry(self.tab_inputs, width=9)
        self.alpha_entry.grid(row=16,column=1)

        self.ignore_recycle_checkbutton = tk.Checkbutton(self.tab_inputs, text="Ignore photon recycle?", variable=self.check_ignore_recycle, onvalue=1, offvalue=0)
        self.ignore_recycle_checkbutton.grid(row=17,column=0)

        self.delta_label = tk.Label(self.tab_inputs, text="Delta Frac.")
        self.delta_label.grid(row=18,column=0)

        self.delta_entry = tk.Entry(self.tab_inputs, width=9)
        self.delta_entry.grid(row=18,column=1)

        self.frac_emitted_label = tk.Label(self.tab_inputs, text="Frac. Emitted (0 to 1)")
        self.frac_emitted_label.grid(row=19,column=0)

        self.frac_emitted_entry = tk.Entry(self.tab_inputs, width=9)
        self.frac_emitted_entry.grid(row=19,column=1)

        self.steps_head = tk.ttk.Label(self.tab_inputs, text="Resolution Setting", style="Header.TLabel")
        self.steps_head.grid(row=20,column=0,columnspan=2)

        self.thickness_label = tk.Label(self.tab_inputs, text="Thickness (nm)")
        self.thickness_label.grid(row=21,column=0)

        self.thickness_entry = tk.Entry(self.tab_inputs, width=9)
        self.thickness_entry.grid(row=21,column=1)

        self.dx_label = tk.Label(self.tab_inputs, text="Space step size [nm]")
        self.dx_label.grid(row=22,column=0)

        self.dx_entry = tk.Entry(self.tab_inputs, width=9)
        self.dx_entry.grid(row=22,column=1)

        self.ICtab_status = tk.Text(self.tab_inputs, width=20,height=6)
        self.ICtab_status.grid(row=23, rowspan=4, column=0, columnspan=2)
        self.ICtab_status.configure(state='disabled')

        self.reset_params_checkbutton = tk.Checkbutton(self.tab_inputs, text="Reset System Parameters", variable=self.check_reset_params, onvalue=True, offvalue=False)
        self.reset_params_checkbutton.grid(row=27,column=0)

        self.reset_params_checkbutton = tk.Checkbutton(self.tab_inputs, text="Reset Initial Distributions", variable=self.check_reset_inits, onvalue=True, offvalue=False)
        self.reset_params_checkbutton.grid(row=28,column=0)

        self.reset_IC_button = tk.Button(self.tab_inputs, text="Reset", command=self.reset_IC)
        self.reset_IC_button.grid(row=29,column=0, columnspan=2)

        self.line1_separator = tk.ttk.Separator(self.tab_inputs, orient="vertical", style="Grey Bar.TSeparator")
        self.line1_separator.grid(row=0,rowspan=30,column=2,pady=(24,0),sticky="ns")
     
        ## Analytical Initial Condition (AIC):

        # An empty GUI element is used to force the analytical IC elements into the correct position.
        # Note that self.tab_analytical_init is a sub-frame attached to the overall self.tab_inputs
        # Normally, the first element of a frame like self.tab_analytical_init would start at row=0, column=0
        # instead of column=2. Starting at column=2 is NOT A TYPO. self.tab_analytical_init is attached to
        # the notebook self.tab_inputs, so it inherits the first two columns of self.tab_inputs.
        self.spacing_box1 = tk.Label(self.tab_analytical_init, text="")
        self.spacing_box1.grid(row=0,rowspan=4,column=2, padx=(300,0))

        self.AIC_head = tk.ttk.Label(self.tab_analytical_init, text="Analytical Init. Cond.", style="Header.TLabel")
        self.AIC_head.grid(row=0,column=3,columnspan=3)

        # A sub-frame attached to a sub-frame
        # With these we can group related elements into a common region
        self.material_param_frame = tk.Frame(master=self.tab_analytical_init, highlightbackground="black", highlightthicknes=1)
        self.material_param_frame.grid(row=1,column=3)

        self.material_param_label = tk.Label(self.material_param_frame, text="Material Params - Select One")
        self.material_param_label.grid(row=0,column=0,columnspan=4)

        self.hline1_separator = tk.ttk.Separator(self.material_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline1_separator.grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        self.calc_AIC_expfactor = tk.ttk.Radiobutton(self.material_param_frame, variable=self.calculate_init_material_expfactor, value=1)
        self.calc_AIC_expfactor.grid(row=2,column=0)

        self.calc_AIC_expfactor_label = tk.Label(self.material_param_frame, text="Option 1")
        self.calc_AIC_expfactor_label.grid(row=2,column=1)

        self.A0_label = tk.Label(self.material_param_frame, text="A0 [cm^-1 eV^-γ]")
        self.A0_label.grid(row=2,column=2)

        self.A0_entry = tk.Entry(self.material_param_frame, width=9)
        self.A0_entry.grid(row=2,column=3)

        self.Eg_label = tk.Label(self.material_param_frame, text="Eg [eV]")
        self.Eg_label.grid(row=3,column=2)

        self.Eg_entry = tk.Entry(self.material_param_frame, width=9)
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

        self.enter_AIC_expfactor = tk.ttk.Radiobutton(self.material_param_frame, variable=self.calculate_init_material_expfactor, value=0)
        self.enter_AIC_expfactor.grid(row=7,column=0)

        self.enter_AIC_expfactor_label = tk.Label(self.material_param_frame, text="Option 2")
        self.enter_AIC_expfactor_label.grid(row=7,column=1)

        self.AIC_expfactor_label = tk.Label(self.material_param_frame, text="α [cm^-1]")
        self.AIC_expfactor_label.grid(row=8,column=2)

        self.AIC_expfactor_entry = tk.Entry(self.material_param_frame, width=9)
        self.AIC_expfactor_entry.grid(row=8,column=3)


        self.pulse_laser_frame = tk.Frame(master=self.tab_analytical_init, highlightbackground="black", highlightthicknes=1)
        self.pulse_laser_frame.grid(row=1,column=4, padx=(20,0))

        self.pulse_laser_label = tk.Label(self.pulse_laser_frame, text="Pulse Laser Params")
        self.pulse_laser_label.grid(row=0,column=0,columnspan=4)

        self.hline3_separator = tk.ttk.Separator(self.pulse_laser_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline3_separator.grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        self.pulse_freq_label = tk.Label(self.pulse_laser_frame, text="Pulse frequency [kHz]")
        self.pulse_freq_label.grid(row=2,column=2)

        self.pulse_freq_entry = tk.Entry(self.pulse_laser_frame, width=9)
        self.pulse_freq_entry.grid(row=2,column=3)

        self.pulse_wavelength_label = tk.Label(self.pulse_laser_frame, text="Wavelength [nm]")
        self.pulse_wavelength_label.grid(row=3,column=2)

        self.pulse_wavelength_entry = tk.Entry(self.pulse_laser_frame, width=9)
        self.pulse_wavelength_entry.grid(row=3,column=3)

        self.gen_power_param_frame = tk.Frame(master=self.tab_analytical_init, highlightbackground="black", highlightthicknes=1)
        self.gen_power_param_frame.grid(row=1,column=5, padx=(20,0))

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

        self.power_entry = tk.Entry(self.gen_power_param_frame, width=9)
        self.power_entry.grid(row=2,column=3)

        self.spotsize_label = tk.Label(self.gen_power_param_frame, text="Spot size [cm^2]")
        self.spotsize_label.grid(row=3,column=2)

        self.spotsize_entry = tk.Entry(self.gen_power_param_frame, width=9)
        self.spotsize_entry.grid(row=3,column=3)

        self.hline5_separator = tk.ttk.Separator(self.gen_power_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline5_separator.grid(row=4,column=0,columnspan=30, pady=(5,5), sticky="ew")

        self.power_density_rb = tk.ttk.Radiobutton(self.gen_power_param_frame, variable=self.AIC_gen_power_mode, value="density")
        self.power_density_rb.grid(row=5,column=0)

        self.power_density_rb_label = tk.Label(self.gen_power_param_frame,text="Option 2")
        self.power_density_rb_label.grid(row=5,column=1)

        self.power_density_label = tk.Label(self.gen_power_param_frame, text="Power Density [uW/cm^2]")
        self.power_density_label.grid(row=5,column=2)

        self.power_density_entry = tk.Entry(self.gen_power_param_frame, width=9)
        self.power_density_entry.grid(row=5,column=3)

        self.hline6_separator = tk.ttk.Separator(self.gen_power_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline6_separator.grid(row=6,column=0,columnspan=30, pady=(5,5), sticky="ew")

        self.max_gen_rb = tk.ttk.Radiobutton(self.gen_power_param_frame, variable=self.AIC_gen_power_mode, value="max-gen")
        self.max_gen_rb.grid(row=7,column=0)

        self.max_gen_rb_label = tk.Label(self.gen_power_param_frame, text="Option 3")
        self.max_gen_rb_label.grid(row=7,column=1)

        self.max_gen_label = tk.Label(self.gen_power_param_frame, text="Max Generation [carr/cm^3]")
        self.max_gen_label.grid(row=7,column=2)

        self.max_gen_entry = tk.Entry(self.gen_power_param_frame, width=9)
        self.max_gen_entry.grid(row=7,column=3)

        self.hline7_separator = tk.ttk.Separator(self.gen_power_param_frame, orient="horizontal", style="Grey Bar.TSeparator")
        self.hline7_separator.grid(row=8,column=0,columnspan=30, pady=(5,5), sticky="ew")

        self.total_gen_rb = tk.ttk.Radiobutton(self.gen_power_param_frame, variable=self.AIC_gen_power_mode, value="total-gen")
        self.total_gen_rb.grid(row=9,column=0)

        self.total_gen_rb_label = tk.Label(self.gen_power_param_frame, text="Option 4")
        self.total_gen_rb_label.grid(row=9,column=1)

        self.total_gen_label = tk.Label(self.gen_power_param_frame, text="Total Generation [carr/cm^3]")
        self.total_gen_label.grid(row=9,column=2)

        self.total_gen_entry = tk.Entry(self.gen_power_param_frame, width=9)
        self.total_gen_entry.grid(row=9,column=3)

        self.load_AIC_button = tk.Button(self.tab_analytical_init, text="Generate Initial Condition", command=self.add_AIC)
        self.load_AIC_button.grid(row=2,column=3,columnspan=3)

        self.AIC_description = tk.Message(self.tab_analytical_init, text="The Analytical Initial Condition uses the above numerical parameters to generate an initial carrier distribution based on an exponential decay equation.", width=320)
        self.AIC_description.grid(row=3,column=3,columnspan=3)
        
        ## Heuristic Initial Condition(HIC):

        self.spacing_box2 = tk.Label(self.tab_rules_init, text="")
        self.spacing_box2.grid(row=0,rowspan=14,column=0,columnspan=3, padx=(370,0))

        self.HIC_list_title = tk.ttk.Label(self.tab_rules_init, text="Heuristic Initial Condition Manager", style="Header.TLabel")
        self.HIC_list_title.grid(row=0,column=3,columnspan=3)

        self.HIC_listbox = tk.Listbox(self.tab_rules_init, width=86,height=8)
        self.HIC_listbox.grid(row=1,rowspan=6,column=3,columnspan=3, padx=(12,0))

        self.add_HIC_title = tk.Label(self.tab_rules_init, text="Add a custom initial condition (see manual for details)")
        self.add_HIC_title.grid(row=7, column=3,columnspan=2, padx=(6,0))

        self.HIC_var_label = tk.Label(self.tab_rules_init, text="Is this init. cond. for ΔN, ΔP [nm^-3], dEc, or chi?")
        self.HIC_var_label.grid(row=8,column=3)

        self.HIC_var_dropdown = tk.OptionMenu(self.tab_rules_init, self.init_var_selection, "ΔN", "ΔP", "dEc", "chi")
        self.HIC_var_dropdown.grid(row=8,column=4)

        self.HIC_method_label = tk.Label(self.tab_rules_init, text="Select condition method:")
        self.HIC_method_label.grid(row=9,column=3)

        self.HIC_method_dropdown = tk.OptionMenu(self.tab_rules_init, self.init_shape_selection, "POINT", "FILL", "LINE", "EXP")
        self.HIC_method_dropdown.grid(row=9, column=4)

        self.HIC_lbound_label = tk.Label(self.tab_rules_init, text="Left bound coordinate:")
        self.HIC_lbound_label.grid(row=10, column=3)

        self.HIC_lbound_entry = tk.Entry(self.tab_rules_init, width=8)
        self.HIC_lbound_entry.grid(row=10,column=4)

        self.HIC_rbound_label = tk.Label(self.tab_rules_init, text="Right bound coordinate:")
        self.HIC_rbound_label.grid(row=11, column=3)

        self.HIC_rbound_entry = tk.Entry(self.tab_rules_init, width=8)
        self.HIC_rbound_entry.grid(row=11,column=4)

        self.HIC_lvalue_label = tk.Label(self.tab_rules_init, text="Left bound value:")
        self.HIC_lvalue_label.grid(row=12, column=3)

        self.HIC_lvalue_entry = tk.Entry(self.tab_rules_init, width=8)
        self.HIC_lvalue_entry.grid(row=12,column=4)

        self.HIC_rvalue_label = tk.Label(self.tab_rules_init, text="Right bound value:")
        self.HIC_rvalue_label.grid(row=13, column=3)

        self.HIC_rvalue_entry = tk.Entry(self.tab_rules_init, width=8)
        self.HIC_rvalue_entry.grid(row=13,column=4)

        self.add_HIC_button = tk.Button(self.tab_rules_init, text="Add new initial condition", command=self.add_HIC)
        self.add_HIC_button.grid(row=14,column=3,columnspan=2)

        self.delete_HIC_button = tk.Button(self.tab_rules_init, text="Delete highlighted init. cond.", command=self.delete_HIC)
        self.delete_HIC_button.grid(row=8,column=5)

        self.deleteall_HIC_button = tk.Button(self.tab_rules_init, text="Delete all init. conds.", command=self.deleteall_HIC)
        self.deleteall_HIC_button.grid(row=9,column=5)

        self.HIC_description = tk.Message(self.tab_rules_init, text="The Heuristic Initial Condition uses a series of rules and patterns to generate an initial carrier distribution.", width=250)
        self.HIC_description.grid(row=10,rowspan=2,column=5,columnspan=2)

        self.HIC_description2 = tk.Message(self.tab_rules_init, text="Warning: Deleting HIC's will RESET the displayed initial condition values.", width=250)
        self.HIC_description2.grid(row=12,rowspan=2,column=5,columnspan=2)

        self.moveup_HIC_button = tk.Button(self.tab_rules_init, text="⇧", command=self.moveup_HIC)
        self.moveup_HIC_button.grid(row=2,column=6, padx=(0,12))

        self.movedown_HIC_button = tk.Button(self.tab_rules_init, text="⇩", command=self.movedown_HIC)
        self.movedown_HIC_button.grid(row=3,column=6, padx=(0,12))

        ## Explicit Inital Condition(EIC):

        self.spacing_box3 = tk.Label(self.tab_explicit_init, text="")
        self.spacing_box3.grid(row=0,rowspan=5,column=0,columnspan=3, padx=(440,0))

        self.EIC_description = tk.Message(self.tab_explicit_init, text="This tab provides an option to directly import a list of data points, on which the TED will do linear interpolation to fit to the specified spacing mesh.", width=360)
        self.EIC_description.grid(row=0,column=3)

        self.EIC_dropdown = tk.OptionMenu(self.tab_explicit_init, self.EIC_var_selection, "ΔN", "ΔP", "dEc", "chi")
        self.EIC_dropdown.grid(row=2,column=3)

        self.add_EIC_button = tk.Button(self.tab_explicit_init, text="Import", command=self.add_EIC)
        self.add_EIC_button.grid(row=3,column=3)

        self.init_fig_NP = plot.figure(1, figsize=(4.85,3))
        self.init_canvas_NP = tkagg.FigureCanvasTkAgg(self.init_fig_NP, master=self.tab_inputs)
        self.init_NP_plotwidget = self.init_canvas_NP.get_tk_widget()
        self.init_NP_plotwidget.grid(row=17,column=3,rowspan=10,columnspan=2)

        self.np_toolbar_frame = tk.Frame(master=self.tab_inputs)
        self.np_toolbar_frame.grid(row=28,column=3,columnspan=2)
        self.np_toolbar = tkagg.NavigationToolbar2Tk(self.init_canvas_NP, self.np_toolbar_frame)

        self.init_fig_ec = plot.figure(2, figsize=(4.85,3))
        self.init_canvas_ec = tkagg.FigureCanvasTkAgg(self.init_fig_ec, master=self.tab_inputs)
        self.init_ec_plotwidget = self.init_canvas_ec.get_tk_widget()
        self.init_ec_plotwidget.grid(row=17,column=5,rowspan=10,columnspan=2)

        self.ec_toolbar_frame = tk.Frame(master=self.tab_inputs)
        self.ec_toolbar_frame.grid(row=28,column=5,columnspan=2)
        self.ec_toolbar = tkagg.NavigationToolbar2Tk(self.init_canvas_ec, self.ec_toolbar_frame)

        # Dictionaries of parameter entry boxes
        self.sys_param_entryboxes_dict = {"Mu_N":self.N_mobility_entry, "Mu_P":self.P_mobility_entry, "N0":self.n0_entry, "P0":self.p0_entry, 
                                          "Thickness":self.thickness_entry, "B":self.B_entry, "Tau_N":self.tauN_entry, "Tau_P":self.tauP_entry,
                                          "Sf":self.Sf_entry, "Sb":self.Sb_entry, "Temperature":self.temperature_entry, "Rel-Permitivity":self.rel_permitivity_entry, "Ext_E-Field":self.ext_efield_entry,
                                          "Theta":self.theta_entry, "Alpha":self.alpha_entry, "Delta":self.delta_entry, "Frac-Emitted":self.frac_emitted_entry, "dx":self.dx_entry}

        self.sys_flag_dict = {"ignore_alpha":Flag(self.ignore_recycle_checkbutton, self.check_ignore_recycle)}

        self.analytical_entryboxes_dict = {"A0":self.A0_entry, "Eg":self.Eg_entry, "AIC_expfactor":self.AIC_expfactor_entry, "Pulse_Freq":self.pulse_freq_entry, 
                                           "Pulse_Wavelength":self.pulse_wavelength_entry, "Power":self.power_entry, "Spotsize":self.spotsize_entry, "Power_Density":self.power_density_entry,
                                           "Max_Gen":self.max_gen_entry, "Total_Gen":self.total_gen_entry}

        # Attach sub-frames to input tab and input tab to overall notebook
        self.tab_inputs.add(self.tab_analytical_init, text="Analytical Init. Cond.")
        self.tab_inputs.add(self.tab_rules_init, text="Heuristic Init. Cond.")
        self.tab_inputs.add(self.tab_explicit_init, text="Explicit Init. Cond.")
        self.notebook.add(self.tab_inputs, text="Inputs")
        return

    def add_tab_simulate(self):
        self.tab_simulate = tk.ttk.Frame(self.notebook)

        self.choose_ICfile_title = tk.ttk.Label(self.tab_simulate, text="Select Init. Cond.", style="Header.TLabel")
        self.choose_ICfile_title.grid(row=0,column=0,columnspan=2, padx=(9,12))

        self.simtime_label = tk.Label(self.tab_simulate, text="Simulation Time [ns]")
        self.simtime_label.grid(row=2,column=0)

        self.simtime_entry = tk.Entry(self.tab_simulate, width=9)
        self.simtime_entry.grid(row=2,column=1)

        self.dt_label = tk.Label(self.tab_simulate, text="dt [ns]")
        self.dt_label.grid(row=3,column=0)

        self.dt_entry = tk.Entry(self.tab_simulate, width=9)
        self.dt_entry.grid(row=3,column=1)

        self.do_ss_checkbutton = tk.Checkbutton(self.tab_simulate, text="Steady State External Stimulation?", variable=self.check_do_ss, onvalue=True, offvalue=False)
        self.do_ss_checkbutton.grid(row=5,column=0)

        self.calculate_NP = tk.Button(self.tab_simulate, text="Calculate ΔN,ΔP", command=self.do_Batch)
        self.calculate_NP.grid(row=6,column=0,columnspan=2,padx=(9,12))

        self.status_label = tk.Label(self.tab_simulate, text="Status")
        self.status_label.grid(row=7, column=0, columnspan=2)

        self.status = tk.Text(self.tab_simulate, width=28,height=4)
        self.status.grid(row=8, rowspan=4, column=0, columnspan=2)
        self.status.configure(state='disabled')

        self.line3_separator = tk.ttk.Separator(self.tab_simulate, orient="vertical", style="Grey Bar.TSeparator")
        self.line3_separator.grid(row=0,rowspan=30,column=2,sticky="ns")

        self.subtitle = tk.Label(self.tab_simulate, text="1-D Carrier Sim (rk4 mtd), with photon propagation")
        self.subtitle.grid(row=0,column=3,columnspan=3)

        self.n_fig = plot.figure(3, figsize=(4.85,3))
        self.n_canvas = tkagg.FigureCanvasTkAgg(self.n_fig, master=self.tab_simulate)
        self.n_plot_widget = self.n_canvas.get_tk_widget()
        self.n_plot_widget.grid(row=1,column=3,rowspan=10,columnspan=2)

        self.nfig_toolbar_frame = tk.Frame(master=self.tab_simulate)
        self.nfig_toolbar_frame.grid(row=12,column=3,columnspan=2)
        self.nfig_toolbar = tkagg.NavigationToolbar2Tk(self.n_canvas, self.nfig_toolbar_frame)

        self.p_fig = plot.figure(4, figsize=(4.85,3))
        self.p_canvas = tkagg.FigureCanvasTkAgg(self.p_fig, master=self.tab_simulate)
        self.p_plot_widget = self.p_canvas.get_tk_widget()
        self.p_plot_widget.grid(row=1,column=5,rowspan=10,columnspan=2)

        self.pfig_toolbar_frame = tk.Frame(master=self.tab_simulate)
        self.pfig_toolbar_frame.grid(row=12,column=5,columnspan=2)
        self.pfig_toolbar = tkagg.NavigationToolbar2Tk(self.p_canvas, self.pfig_toolbar_frame)

        self.E_fig = plot.figure(5, figsize=(4.85,3))
        self.E_canvas = tkagg.FigureCanvasTkAgg(self.E_fig, master=self.tab_simulate)
        self.E_plot_widget = self.E_canvas.get_tk_widget()
        self.E_plot_widget.grid(row=13,column=3,rowspan=10,columnspan=2)

        self.Efig_toolbar_frame = tk.Frame(master=self.tab_simulate)
        self.Efig_toolbar_frame.grid(row=24,column=3,columnspan=2)
        self.Efig_toolbar = tkagg.NavigationToolbar2Tk(self.E_canvas, self.Efig_toolbar_frame)

        self.notebook.add(self.tab_simulate, text="Simulate")
        return

    def add_tab_analyze(self):
        self.tab_analyze = tk.ttk.Frame(self.notebook)

        self.main_fig1 = plot.figure(7, figsize=(4.85,3))
        self.main1_canvas = tkagg.FigureCanvasTkAgg(self.main_fig1, master=self.tab_analyze)
        self.main1_widget = self.main1_canvas.get_tk_widget()
        self.main1_widget.grid(row=0,column=0,rowspan=10,columnspan=2, padx=(12,0))
        self.analysis_plots[0].fig_ID = 7
        self.analysis_plots[0].plot_obj = self.main_fig1

        self.main1_toolbar_frame = tk.Frame(master=self.tab_analyze)
        self.main1_toolbar_frame.grid(row=10,column=0,rowspan=2,columnspan=2)
        self.main1_toolbar = tkagg.NavigationToolbar2Tk(self.main1_canvas, self.main1_toolbar_frame)
        self.main1_toolbar.grid(row=0,column=0,columnspan=99)

        self.main1_plot_button = tk.Button(self.main1_toolbar_frame, text="Plot", command=partial(self.a_plot, plot_ID=0))
        self.main1_plot_button.grid(row=1,column=0,padx=(0,5))
        
        self.main1_tstep_entry = tk.Entry(self.main1_toolbar_frame, width=9)
        self.main1_tstep_entry.grid(row=1,column=1,padx=(0,5))

        self.main1_tstep_button = tk.Button(self.main1_toolbar_frame, text="Step >>", command=partial(self.plot_tstep, plot_ID=0))
        self.main1_tstep_button.grid(row=1,column=2,padx=(0,5))

        self.calculate_PL1 = tk.Button(self.main1_toolbar_frame, text=">> Integrate <<", command=partial(self.do_Integrate, plot_ID=0))
        self.calculate_PL1.grid(row=1,column=3,padx=(0,5))

        self.main1_axis_button = tk.Button(self.main1_toolbar_frame, text="Axis Settings", command=partial(self.do_change_axis_popup, plot_ID=0))
        self.main1_axis_button.grid(row=1,column=4, padx=(0,5))

        self.main1_export_button = tk.Button(self.main1_toolbar_frame, text="Export", command=partial(self.export_plot, plot_ID=0))
        self.main1_export_button.grid(row=1,column=5, padx=(0,5))

        self.main1_IC_carry_button = tk.Button(self.main1_toolbar_frame, text="Generate IC", command=partial(self.do_IC_carry_popup, plot_ID=0))
        self.main1_IC_carry_button.grid(row=1,column=6,padx=(0,5))

        self.main_fig2 = plot.figure(8, figsize=(4.85,3))
        self.main2_canvas = tkagg.FigureCanvasTkAgg(self.main_fig2, master=self.tab_analyze)
        self.main2_widget = self.main2_canvas.get_tk_widget()
        self.main2_widget.grid(row=13,column=0,rowspan=10,columnspan=2)
        self.analysis_plots[1].fig_ID = 8
        self.analysis_plots[1].plot_obj = self.main_fig2

        self.main2_toolbar_frame = tk.Frame(master=self.tab_analyze)
        self.main2_toolbar_frame.grid(row=23,column=0,rowspan=2,columnspan=2)
        self.main2_toolbar = tkagg.NavigationToolbar2Tk(self.main2_canvas, self.main2_toolbar_frame)
        self.main2_toolbar.grid(row=0,column=0,columnspan=99)

        self.main2_plot_button = tk.Button(self.main2_toolbar_frame, text="Plot", command=partial(self.a_plot, plot_ID=1))
        self.main2_plot_button.grid(row=1,column=0,padx=(0,5))

        self.main2_tstep_entry = tk.Entry(self.main2_toolbar_frame, width=9)
        self.main2_tstep_entry.grid(row=1,column=1,padx=(0,5))

        self.main2_tstep_button = tk.Button(self.main2_toolbar_frame, text="Step >>", command=partial(self.plot_tstep, plot_ID=1))
        self.main2_tstep_button.grid(row=1,column=2, padx=(0,5))

        self.calculate_PL2 = tk.Button(self.main2_toolbar_frame, text=">> Integrate <<", command=partial(self.do_Integrate, plot_ID=1))
        self.calculate_PL2.grid(row=1,column=3, padx=(0,5))

        self.main2_axis_button = tk.Button(self.main2_toolbar_frame, text="Axis Settings", command=partial(self.do_change_axis_popup, plot_ID=1))
        self.main2_axis_button.grid(row=1,column=4, padx=(0,5))

        self.main2_export_button = tk.Button(self.main2_toolbar_frame, text="Export", command=partial(self.export_plot, plot_ID=1))
        self.main2_export_button.grid(row=1,column=5, padx=(0,5))

        self.main2_IC_carry_button = tk.Button(self.main2_toolbar_frame, text="Generate IC", command=partial(self.do_IC_carry_popup, plot_ID=1))
        self.main2_IC_carry_button.grid(row=1,column=6,padx=(0,5))

        self.analysis_title = tk.ttk.Label(self.tab_analyze, text="Plot and Integrate Saved Datasets", style="Header.TLabel")
        self.analysis_title.grid(row=0,column=3,columnspan=1, padx=(9,12))

        self.main_fig3 = plot.figure(9, figsize=(5.8,3.6))
        self.main3_canvas = tkagg.FigureCanvasTkAgg(self.main_fig3, master=self.tab_analyze)
        self.main3_widget = self.main3_canvas.get_tk_widget()
        self.main3_widget.grid(row=3,column=3,rowspan=16,columnspan=1, padx=(20,0))

        self.main3_toolbar_frame = tk.Frame(master=self.tab_analyze)
        self.main3_toolbar_frame.grid(row=18,column=3,columnspan=1, pady=(110,0))
        self.main3_toolbar = tkagg.NavigationToolbar2Tk(self.main3_canvas, self.main3_toolbar_frame)
        self.main3_toolbar.grid(row=0,column=0,columnspan=99)

        self.main3_axis_button = tk.Button(self.main3_toolbar_frame, text="Axis Settings", command=partial(self.do_change_axis_popup, plot_ID=-1))
        self.main3_axis_button.grid(row=1,column=0, padx=(0,5))

        self.main3_export_button = tk.Button(self.main3_toolbar_frame, text="Export", command=partial(self.export_plot, plot_ID=-1))
        self.main3_export_button.grid(row=1,column=1, padx=(0,5))

        self.main3_bayesim_button = tk.Button(self.main3_toolbar_frame, text="Bayesim", command=partial(self.do_bayesim_popup))
        self.main3_bayesim_button.grid(row=1,column=2,padx=(0,5))

        self.analysis_status = tk.Text(self.tab_analyze, width=28,height=3)
        self.analysis_status.grid(row=20,rowspan=3,column=3,columnspan=1)
        self.analysis_status.configure(state="disabled")

        self.main_tstep_entryboxes = [self.main1_tstep_entry, self.main2_tstep_entry]
        self.notebook.add(self.tab_analyze, text="Analyze")
        return

    def DEBUG(self):
        self.enter(self.AIC_expfactor_entry, "0.004")
        self.enter(self.max_gen_entry, "20")
        self.enter(self.pulse_freq_entry, "1")
        self.enter(self.pulse_wavelength_entry, "520")
        print("IC_IS_AIC = {}".format(self.IC_is_AIC))
        return

    ## Functions to create popups and manage

    def do_batch_popup(self):
        # Check that user has filled in all parameters
        if not (self.test_entryboxes_valid(self.sys_param_entryboxes_dict)):
            self.write(self.ICtab_status, "Error: Missing or invalid parameters")
            return

        if not self.batch_popup_isopen: # Don't open more than one of this window at a time
            self.batch_param = tk.StringVar()

            self.batch_popup = tk.Toplevel(self.root)
            self.batch_title_label = tk.ttk.Label(self.batch_popup, text="Batch IC Tool", style="Header.TLabel")
            self.batch_title_label.grid(row=0,column=0)
            self.batch_instruction1 = tk.Message(self.batch_popup, text="This Batch Tool allows you to generate many copies of the currently-loaded IC, varying exactly one parameter between all of them.", width=300)
            self.batch_instruction1.grid(row=1,column=0)

            self.batch_instruction2 = tk.Message(self.batch_popup, text="An IC is considered currently-loaded when plots appear on the main Inputs tab and the system parameters are filled in.", width=300)
            self.batch_instruction2.grid(row=2,column=0)

            self.batch_instruction3 = tk.Message(self.batch_popup, text="Please ensure that this is the case before using this tool.", width=300)
            self.batch_instruction3.grid(row=3,column=0)

            self.batch_param_label = tk.Label(self.batch_popup, text="Select Batch Parameter:")
            self.batch_param_label.grid(row=0,column=1)

            # Contextually-dependent options for batchable params
            if self.IC_is_AIC:
                self.batch_param_select = tk.OptionMenu(self.batch_popup, self.batch_param, *[key for key in self.sys_param_entryboxes_dict.keys() if not (key == "Thickness" or key == "dx")], 
                                                        *[key for key in self.analytical_entryboxes_dict.keys() if not (
                                                            (self.calculate_init_material_expfactor.get() and (key == "AIC_expfactor")) or
                                                            (not self.calculate_init_material_expfactor.get() and (key == "A0" or key == "Eg")) or
                                                            (self.AIC_gen_power_mode.get() == "power-spot" and (key == "Power_Density" or key == "Max_Gen" or key == "Total_Gen")) or
                                                            (self.AIC_gen_power_mode.get() == "density" and (key == "Power" or key == "Spotsize" or key == "Max_Gen" or key == "Total_Gen")) or
                                                            (self.AIC_gen_power_mode.get() == "max-gen" and (key == "Power" or key == "Spotsize" or key == "Power_Density" or key == "Total_Gen")) or
                                                            (self.AIC_gen_power_mode.get() == "total-gen" and (key == "Power_Density" or key == "Power" or key == "Spotsize" or key == "Max_Gen"))
                                                            )])
            else:
                self.batch_param_select = tk.OptionMenu(self.batch_popup, self.batch_param, *[key for key in self.sys_param_entryboxes_dict.keys() if not (key == "Thickness" or key == "dx")])
            self.batch_param_select.grid(row=0,column=2)

            self.batch_param_entry = tk.Entry(self.batch_popup, width=80)
            self.enter(self.batch_param_entry, "Enter a list of space-separated values for the selected Batch Parameter")
            self.batch_param_entry.grid(row=0,column=3)

            self.batch_status = tk.Text(self.batch_popup, width=40,height=2)
            self.batch_status.grid(row=1,column=3)
            self.batch_status.configure(state='disabled')

            self.batch_name_label = tk.Label(self.batch_popup, text="Enter a name for the new batch folder:")
            self.batch_name_label.grid(row=2,column=3,columnspan=1)

            self.batch_name_entry = tk.Entry(self.batch_popup, width=24)
            self.batch_name_entry.grid(row=3,column=3,padx=(0,120))

            self.create_batch_button = tk.Button(self.batch_popup, text="Create Batch", command=self.create_batch_init)
            self.create_batch_button.grid(row=3,column=3, padx=(130,0))

            ## Temporarily disable the main window while this popup is active
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

    def do_plotter_popup(self, plot_ID):
        if not self.plotter_popup_isopen:

            self.plotter_popup = tk.Toplevel(self.root)

            self.plotter_title_label = tk.ttk.Label(self.plotter_popup, text="Select a data type", style="Header.TLabel")
            self.plotter_title_label.grid(row=0,column=0,columnspan=3)

            self.var_select_menu = tk.OptionMenu(self.plotter_popup, self.data_var, "ΔN", "ΔP", "E-field", "RR", "NRR", "PL")
            self.var_select_menu.grid(row=1,column=1)

            self.plotter_continue_button = tk.Button(self.plotter_popup, text="Continue and select datasets", command=partial(self.on_plotter_popup_close, plot_ID, continue_=True))
            self.plotter_continue_button.grid(row=2,column=1)

            self.data_listbox = tk.Listbox(self.plotter_popup, width=20, height=20, selectmode="extended")
            self.data_listbox.grid(row=3,rowspan=13,column=0,columnspan=2)
            self.data_listbox.delete(0,tk.END)
            self.data_list = [file for file in os.listdir(self.default_dirs["Data"]) if not file.endswith(".txt")]
            self.data_listbox.insert(0,*(self.data_list))

            self.plotter_status = tk.Text(self.plotter_popup, width=24,height=2)
            self.plotter_status.grid(row=3,rowspan=2,column=2,columnspan=1)
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

                self.analysis_plots[plot_ID].remove_duplicate_filenames()

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
            self.integration_lbound = ""
            self.integration_ubound = ""

            self.integration_getbounds_popup = tk.Toplevel(self.root)

            self.single_intg = tk.ttk.Radiobutton(self.integration_getbounds_popup, variable=self.fetch_intg_mode, value='single')
            self.single_intg.grid(row=0,column=0, rowspan=3)

            self.single_intg_label = tk.ttk.Label(self.integration_getbounds_popup, text="Single integral", style="Header.TLabel")
            self.single_intg_label.grid(row=0,column=1, rowspan=3, padx=(0,20))

            self.integration_getbounds_title_label = tk.ttk.Label(self.integration_getbounds_popup, text="Enter bounds of integration [nm]")
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

            self.xaxis_param_menu = tk.OptionMenu(self.PL_xaxis_popup, self.xaxis_selection, "Mu_N", "Mu_P", "N0", "P0", "Thickness", "dx", "B", "Tau_N", "Tau_P", "Sf", \
            "Sb", "Temperature", "Rel-Permitivity", "Theta", "Alpha", "Delta", "Frac-Emitted","Total-Time","dt","ignore_alpha")
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
    
    def do_change_axis_popup(self, plot_ID):
        # Don't open if no data plotted
        if plot_ID == -1:
            if self.I_plot.size() == 0: return

        else:
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

            self.toggle_legend_checkbutton = tk.Checkbutton(self.change_axis_popup, text="Display legend?", variable=self.check_display_legend, onvalue=True, offvalue=False)
            self.toggle_legend_checkbutton.grid(row=2,column=0,columnspan=2)

            self.change_axis_continue_button = tk.Button(self.change_axis_popup, text="Continue", command=partial(self.on_change_axis_popup_close, plot_ID, continue_=True))
            self.change_axis_continue_button.grid(row=3,column=0,columnspan=2)

            self.change_axis_status = tk.Text(self.change_axis_popup, width=24,height=2)
            self.change_axis_status.grid(row=4,rowspan=2,column=0,columnspan=2)
            self.change_axis_status.configure(state="disabled")

            # Set the default values in the entry boxes to be the current options of the plot (in case the user only wants to make a few changes)
            if not (plot_ID == -1):
                self.enter(self.xlbound, self.analysis_plots[plot_ID].xlim[0])
                self.enter(self.xubound, self.analysis_plots[plot_ID].xlim[1])
                self.enter(self.ylbound, self.analysis_plots[plot_ID].ylim[0])
                self.enter(self.yubound, self.analysis_plots[plot_ID].ylim[1])
                self.xaxis_type.set(self.analysis_plots[plot_ID].xaxis_type)
                self.yaxis_type.set(self.analysis_plots[plot_ID].yaxis_type)
                if self.analysis_plots[plot_ID].display_legend: self.toggle_legend_checkbutton.select()

            else:
                self.enter(self.xlbound, self.I_plot.xlim[0])
                self.enter(self.xubound, self.I_plot.xlim[1])
                self.enter(self.ylbound, self.I_plot.ylim[0])
                self.enter(self.yubound, self.I_plot.ylim[1])
                self.xaxis_type.set(self.I_plot.xaxis_type)
                self.yaxis_type.set(self.I_plot.yaxis_type)
                if self.I_plot.display_legend: self.toggle_legend_checkbutton.select()

            self.change_axis_popup.protocol("WM_DELETE_WINDOW", partial(self.on_change_axis_popup_close, plot_ID, continue_=False))
            self.change_axis_popup.grab_set()
            self.change_axis_popup_isopen = True
        else:
            print("Error #440: Opened more than one PL change axis popup at a time")
        return

    def on_change_axis_popup_close(self, plot_ID, continue_=False):
        try:
            if continue_:
                if not self.xaxis_type or not self.yaxis_type: raise ValueError("Error: invalid axis type")
                if self.xlbound.get() == "" or self.xubound.get() == "" or self.ylbound.get() == "" or self.yubound.get() == "": raise ValueError("Error: missing bounds")
                bounds = [float(self.xlbound.get()), float(self.xubound.get()), float(self.ylbound.get()), float(self.yubound.get())]
            
                if not (plot_ID == -1):
                    plot.figure(self.analysis_plots[plot_ID].fig_ID)
                    
                else:
                    plot.figure(9)

                # Set plot axis params and save in corresponding plot state object, if the selected plot has such an object
                plot.yscale(self.yaxis_type.get())
                plot.xscale(self.xaxis_type.get())

                plot.ylim(bounds[2], bounds[3])
                plot.xlim(bounds[0], bounds[1])

                if self.check_display_legend.get():
                    plot.legend()
                else:
                    plot.legend('', frameon=False)

                plot.tight_layout()

                if not (plot_ID == -1):
                    self.analysis_plots[plot_ID].plot_obj.canvas.draw()
                    
                else:
                    self.main_fig3.canvas.draw()

                # Save these params to pre-populate the popup the next time it's opened
                if not (plot_ID == -1):
                    self.analysis_plots[plot_ID].yaxis_type = self.yaxis_type.get()
                    self.analysis_plots[plot_ID].xaxis_type = self.xaxis_type.get()
                    self.analysis_plots[plot_ID].ylim = (bounds[2], bounds[3])
                    self.analysis_plots[plot_ID].xlim = (bounds[0], bounds[1])
                    self.analysis_plots[plot_ID].display_legend = self.check_display_legend.get()
                else:
                    self.I_plot.yaxis_type = self.yaxis_type.get()
                    self.I_plot.xaxis_type = self.xaxis_type.get()
                    self.I_plot.ylim = (bounds[2], bounds[3])
                    self.I_plot.xlim = (bounds[0], bounds[1])
                    self.I_plot.display_legend = self.check_display_legend.get()

            self.change_axis_popup.destroy()
            print("PL change axis popup closed")
            self.change_axis_popup_isopen = False

        except ValueError as oops:
            self.write(self.change_axis_status, oops)
            return
        except:
            print("Error #441: Failed to close PL xaxis popup.")

        return

    def do_IC_carry_popup(self, plot_ID):
        # Don't open if no data plotted
        if self.analysis_plots[plot_ID].datagroup.size() == 0: return

        if not self.IC_carry_popup_isopen:
            self.IC_carry_popup = tk.Toplevel(self.root)

            self.IC_carry_title_label = tk.ttk.Label(self.IC_carry_popup, text="Select data to include in new IC", style="Header.TLabel")
            self.IC_carry_title_label.grid(row=0,column=0,columnspan=2)

            self.include_N_checkbutton = tk.Checkbutton(self.IC_carry_popup, text="ΔN", variable=self.carry_include_N)
            self.include_N_checkbutton.grid(row=1,column=0)

            self.include_P_checkbutton = tk.Checkbutton(self.IC_carry_popup, text="ΔP", variable=self.carry_include_P)
            self.include_P_checkbutton.grid(row=2,column=0)
            
            self.include_E_checkbutton = tk.Checkbutton(self.IC_carry_popup, text="E-field", variable=self.carry_include_E_field)
            self.include_E_checkbutton.grid(row=3,column=0)

            self.carry_IC_listbox = tk.Listbox(self.IC_carry_popup, width=30,height=10, selectmode='extended')
            self.carry_IC_listbox.grid(row=4,column=0,columnspan=2)
            for key in self.analysis_plots[plot_ID].datagroup.datasets:
                self.carry_IC_listbox.insert(tk.END, key)

            self.IC_carry_continue_button = tk.Button(self.IC_carry_popup, text="Continue", command=partial(self.on_IC_carry_popup_close, plot_ID, continue_=True))
            self.IC_carry_continue_button.grid(row=5,column=0,columnspan=2)

            self.IC_carry_popup.protocol("WM_DELETE_WINDOW", partial(self.on_IC_carry_popup_close, plot_ID, continue_=False))
            self.IC_carry_popup.grab_set()
            self.IC_carry_popup_isopen = True
        else:
            print("Error #510: Opened more than one PL change axis popup at a time")
        return

    def on_IC_carry_popup_close(self, plot_ID, continue_=False):
        try:
            if continue_:
                active_sets = self.analysis_plots[plot_ID].datagroup.datasets
                pile = [self.carry_IC_listbox.get(i) for i in self.carry_IC_listbox.curselection()]
                for key in pile:
                    new_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["Initial"], title="Save IC text file for {}".format(key), filetypes=[("Text files","*.txt")])
                    if new_filename == "": continue

                    if new_filename.endswith(".txt"): new_filename = new_filename[:-4]

                    node_x = active_sets[key].node_x
                    edge_x = active_sets[key].edge_x
                    init_N = self.read_N(active_sets[key].filename, active_sets[key].show_index) if self.carry_include_N.get() else np.zeros(node_x.__len__())
                    init_P = self.read_P(active_sets[key].filename, active_sets[key].show_index) if self.carry_include_P.get() else np.zeros(node_x.__len__())
                    init_E_field = self.read_E_field(active_sets[key].filename, active_sets[key].show_index) if self.carry_include_E_field.get() else np.zeros(edge_x.__len__())

                    with open(new_filename + ".txt", "w+") as ofstream:
                        ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
                        ofstream.write("$ System Parameters:\n")
                        for param in active_sets[key].params_dict:
                            if param == "Total-Time" or param == "dt" or param == "ignore_alpha": 
                                continue
                            ofstream.write(param + ": " + str(active_sets[key].params_dict[param] * self.convert_out_dict[param]) + "\n")

                        ofstream.write("$ Initial Conditions: (Nodes) x, N, P\n")
                        for i in range(node_x.__len__()):
                            ofstream.write("{:.8e}\t{:.8e}\t{:.8e}\n".format(node_x[i], init_N[i], init_P[i]))

                        ofstream.write("\n$ Initial Conditions: (Edges) x, E-field, Eg, Chi\n")
                        for i in range(edge_x.__len__()):
                            ofstream.write("{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\n".format(edge_x[i], init_E_field[i], 0, 0))

                self.write(self.analysis_status, "IC file generated")

            self.IC_carry_popup.destroy()
            print("IC carry popup closed")
            self.IC_carry_popup_isopen = False

        except OSError as oops:
            self.write(self.analysis_status, "IC file not created")
            
        except:
            print("Error #511: Failed to close IC carry popup.")

        return

    def do_bayesim_popup(self):
        if self.I_plot.size() == 0: return

        if not self.bayesim_popup_isopen:

            self.bay_popup = tk.Toplevel(self.root)

            self.bay_title_label1 = tk.ttk.Label(self.bay_popup, text="Bayesim Tool", style="Header.TLabel")
            self.bay_title_label1.grid(row=0,column=0)

            self.bay_text1 = tk.Message(self.bay_popup, text=
                                        "Select \"Observation\" to save each curve as an experimentally observed data set or " +
                                        "Model to combine all curves into a single model set.", width=320)
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

            self.bay_mod_header = tk.Label(self.bay_popup, text="Model")
            self.bay_mod_header.grid(row=2,column=3)

            self.bay_title_label3 = tk.ttk.Label(self.bay_popup, text="Model Params", style="Header.TLabel")
            self.bay_title_label3.grid(row=4,column=2,columnspan=4)

            self.bay_mun_check = tk.Checkbutton(self.bay_popup, text="Mu_N", variable=self.bay_params["Mu_N"], onvalue=True, offvalue=False)
            self.bay_mun_check.grid(row=5,column=2, padx=(19,0))

            self.bay_mup_check = tk.Checkbutton(self.bay_popup, text="Mu_P", variable=self.bay_params["Mu_P"], onvalue=True, offvalue=False)
            self.bay_mup_check.grid(row=6,column=2, padx=(17,0))

            self.bay_n0_check = tk.Checkbutton(self.bay_popup, text="N0", variable=self.bay_params["N0"], onvalue=True, offvalue=False)
            self.bay_n0_check.grid(row=7,column=2, padx=(3,0))

            self.bay_p0_check = tk.Checkbutton(self.bay_popup, text="P0", variable=self.bay_params["P0"], onvalue=True, offvalue=False)
            self.bay_p0_check.grid(row=8,column=2)

            self.bay_B_check = tk.Checkbutton(self.bay_popup, text="B", variable=self.bay_params["B"], onvalue=True, offvalue=False)
            self.bay_B_check.grid(row=5,column=3, padx=(0,2))

            self.bay_taun_check = tk.Checkbutton(self.bay_popup, text="Tau_N", variable=self.bay_params["Tau_N"], onvalue=True, offvalue=False)
            self.bay_taun_check.grid(row=6,column=3, padx=(24,0))

            self.bay_taup_check = tk.Checkbutton(self.bay_popup, text="Tau_P", variable=self.bay_params["Tau_P"], onvalue=True, offvalue=False)
            self.bay_taup_check.grid(row=7,column=3, padx=(23,0))

            self.bay_sf_check = tk.Checkbutton(self.bay_popup, text="Sf", variable=self.bay_params["Sf"], onvalue=True, offvalue=False)
            self.bay_sf_check.grid(row=8,column=3, padx=(1,0))

            self.bay_sb_check = tk.Checkbutton(self.bay_popup, text="Sb", variable=self.bay_params["Sb"], onvalue=True, offvalue=False)
            self.bay_sb_check.grid(row=5,column=4, padx=(0,40))

            self.bay_temperature_check = tk.Checkbutton(self.bay_popup, text="Temperature", variable=self.bay_params["Temperature"], onvalue=True, offvalue=False)
            self.bay_temperature_check.grid(row=6,column=4, padx=(14,0))

            self.bay_relperm_check = tk.Checkbutton(self.bay_popup, text="Rel-Permitivity", variable=self.bay_params["Rel-Permitivity"], onvalue=True, offvalue=False)
            self.bay_relperm_check.grid(row=7,column=4, padx=(24,0))

            self.bay_theta_check = tk.Checkbutton(self.bay_popup, text="Theta", variable=self.bay_params["Theta"], onvalue=True, offvalue=False)
            self.bay_theta_check.grid(row=8,column=4, padx=(0,25))

            self.bay_alpha_check = tk.Checkbutton(self.bay_popup, text="Alpha", variable=self.bay_params["Alpha"], onvalue=True, offvalue=False)
            self.bay_alpha_check.grid(row=5,column=5, padx=(0,26))

            self.bay_delta_check = tk.Checkbutton(self.bay_popup, text="Delta", variable=self.bay_params["Delta"], onvalue=True, offvalue=False)
            self.bay_delta_check.grid(row=6,column=5, padx=(0,30))

            self.bay_fm_check = tk.Checkbutton(self.bay_popup, text="Frac-Emitted", variable=self.bay_params["Frac-Emitted"], onvalue=True, offvalue=False)
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
                for key in self.bay_params:
                    print("{}: {}".format(key, self.bay_params[key].get()))

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
    def update_data_plots(self, index, do_clear_plots=True):
        ## Update plots on Simulate tab
        ## Update N 
        plot.figure(3)
        plot_labels = ("ns", "x [nm]", "ΔN [nm^-3]")
        if do_clear_plots: plot.clf()
        
        plot.ylim(*(self.N_limits))
        plot.yscale('log')

        plot.plot(self.node_x, self.sim_N)

        plot.xlabel(plot_labels[1])
        plot.ylabel(plot_labels[2])

        plot.title('Time: ' + str(self.simtime * index / self.n) + ' ' + plot_labels[2])
        plot.tight_layout()
        self.n_fig.canvas.draw()

        ## Update P
        plot.figure(4)
        plot_labels = ("ns", "x [nm]", "ΔP [nm^-3]")

        if do_clear_plots: plot.clf()

        plot.ylim(*(self.P_limits))
        plot.yscale('log')

        plot.plot(self.node_x, self.sim_P)

        plot.xlabel(plot_labels[1])
        plot.ylabel(plot_labels[2])

        plot.title('Time: ' + str(self.simtime * index / self.n) + ' ' + plot_labels[2])
        plot.tight_layout()
        self.p_fig.canvas.draw()

        ## Update E-field
        plot.figure(5)
        plot_labels = ("ns", "x [nm]", "Magnitude of E field [?]")

        if do_clear_plots: plot.clf()
 
        try:
            plot.ylim(np.amax(np.abs(self.sim_E_field)) * 1e-11, np.amax(np.abs(self.sim_E_field)) * 1e1)
            if (np.amax(np.abs(self.sim_E_field)) == 0): raise ValueError
        except:
            plot.ylim(*(self.N_limits))

        plot.yscale('log')

        plot.plot(self.edge_x, np.abs(self.sim_E_field))

        plot.xlabel(plot_labels[1])
        plot.ylabel(plot_labels[2])

        plot.title('Time: ' + str(self.simtime * index / self.n) + ' ' + plot_labels[2])
        plot.tight_layout()
        self.E_fig.canvas.draw()
        return

    ## Sub plotters for analyze tab

    def data_plot(self, plot_ID, clear_plot=True):
        # Draw on analysis tab
        try:
            active_plot = self.analysis_plots[plot_ID]
            plot.figure(active_plot.fig_ID)

            if clear_plot: plot.clf()
            
            plot.yscale(active_plot.yaxis_type)
            plot.xscale(active_plot.xaxis_type)
            active_datagroup = active_plot.datagroup

            plot.ylim(*active_plot.ylim)
            plot.xlim(*active_plot.xlim)

            for tag in active_datagroup.datasets:
                plot.plot(active_datagroup.datasets[tag].grid_x, active_datagroup.datasets[tag].data, label=tag)

            plot.xlabel("x [nm]")
            plot.ylabel(active_datagroup.type)
            plot.legend()
            plot.title("Time: " + str(active_datagroup.get_maxtime() * active_plot.time_index / active_datagroup.get_maxnumtsteps()) + " / " + str(active_datagroup.get_maxtime()) + "ns")
            plot.tight_layout()
            active_plot.plot_obj.canvas.draw()

        except OSError:
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
                param_values_dict = {"Mu_N":0, "Mu_P":0, "N0":0, "P0":0, "Thickness":0, "dx":0,
                                     "B":0, "Tau_N":0, "Tau_P":0,"Sf":0, "Sb":0, 
                                     "Temperature":0, "Rel-Permitivity":0, "Ext_E-Field":0, "Theta":0, "Alpha":0, "Delta":0, 
                                     "Frac-Emitted":0, "Total-Time":0, "dt":0, "ignore_alpha":0}
                for line in ifstream:
                    if "$" in line: continue

                    elif "#" in line: continue

                    else:
                        param_values_dict[line[0:line.find(':')]] = float(line[line.find(' ') + 1:].strip('\n'))

            data_n = int(0.5 + param_values_dict["Total-Time"] / param_values_dict["dt"])
            data_m = int(0.5 + param_values_dict["Thickness"] / param_values_dict["dx"])
            data_edge_x = np.linspace(0, param_values_dict["Thickness"],data_m+1)
            data_node_x = np.linspace(param_values_dict["dx"] / 2, param_values_dict["Thickness"] - param_values_dict["dx"] / 2, data_m)

            # Convert from cm, V, s to nm, V, ns
            param_values_dict["Sf"] = param_values_dict["Sf"] * self.convert_in_dict["Sf"]
            param_values_dict["Sb"] = param_values_dict["Sb"] * self.convert_in_dict["Sb"]
            param_values_dict["Mu_N"] = param_values_dict["Mu_N"] * self.convert_in_dict["Mu_N"]
            param_values_dict["Mu_P"] = param_values_dict["Mu_P"] * self.convert_in_dict["Mu_P"]
            param_values_dict["N0"] = param_values_dict["N0"] * self.convert_in_dict["N0"]
            param_values_dict["P0"] = param_values_dict["P0"] * self.convert_in_dict["P0"]
            param_values_dict["B"] = param_values_dict["B"] * self.convert_in_dict["B"]
            param_values_dict["Theta"] = param_values_dict["Theta"] * self.convert_in_dict["Theta"]
            param_values_dict["Alpha"] = param_values_dict["Alpha"] * self.convert_in_dict["Alpha"]
            param_values_dict["Ext_E-Field"] = param_values_dict["Ext_E-Field"] * self.convert_in_dict["Ext_E-Field"]

            param_tuple = (param_values_dict["Mu_N"], param_values_dict["Mu_P"], param_values_dict["N0"], param_values_dict["P0"], 
                           param_values_dict["Thickness"], param_values_dict["dx"], param_values_dict["B"], param_values_dict["Tau_N"], 
                           param_values_dict["Tau_P"], param_values_dict["Sf"], param_values_dict["Sb"], param_values_dict["Temperature"], 
                           param_values_dict["Rel-Permitivity"], param_values_dict["Ext_E-Field"], param_values_dict["Theta"], param_values_dict["Alpha"], param_values_dict["Delta"], 
                           param_values_dict["Frac-Emitted"], param_values_dict["Total-Time"], param_values_dict["dt"], param_values_dict["ignore_alpha"])
        except:
            self.write(self.analysis_status, "Error: This data set is missing or has unusual metadata.txt")
            return

        active_datagroup = active_plot.datagroup
        if not do_overlay: 
            active_plot.time_index = 0
            active_datagroup.clear()

        active_show_index = active_plot.time_index

		# Now that we have the parameters from metadata, fetch the data itself
		# For N, P, E-field this is just reading the data but for others we'll calculate it in situ
        if (datatype == "ΔN"):
            try:
                # Having data_node_x twice is NOT a typo - see definition of Data_Set() class for explanation
                new_data = Data_Set(self.read_N(data_filename, active_show_index), data_node_x, data_node_x, data_edge_x, *(param_tuple), datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: The data set {} is missing -n data".format(data_filename))
                return

        elif (datatype == "ΔP"):
            try:
                new_data = Data_Set(self.read_P(data_filename, active_show_index), data_node_x, data_node_x, data_edge_x, *(param_tuple), datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: The data set {} is missing -p data".format(data_filename))
                return

        elif (datatype == "E-field"):
            try:
                new_data = Data_Set(self.read_E_field(data_filename, active_show_index), data_edge_x, data_node_x, data_edge_x, *(param_tuple), datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: The data set {} is missing -E_field data".format(data_filename))
                return

        elif (datatype == "RR"):
            try:
                new_data = Data_Set(param_values_dict["B"] * ((self.read_N(data_filename, active_show_index) + param_values_dict["N0"]) * (self.read_P(data_filename, active_show_index) + param_values_dict["P0"]) - param_values_dict["N0"] * param_values_dict["P0"]), 
                                    data_node_x, data_node_x, data_edge_x, *(param_tuple), datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: Unable to calculate Rad Rec")
                return

        elif (datatype == "NRR"):
            try:
                temp_N = self.read_N(data_filename, active_show_index)
                temp_P = self.read_P(data_filename, active_show_index)
                new_data = Data_Set(((temp_N + param_values_dict["N0"]) * (temp_P + param_values_dict["P0"]) - param_values_dict["N0"] * param_values_dict["P0"]) / 
                                    ((param_values_dict["Tau_N"] * (temp_P + param_values_dict["P0"])) + (param_values_dict["Tau_P"] * (temp_N + param_values_dict["N0"]))), 
                                    data_node_x, data_node_x, data_edge_x, *(param_tuple), datatype, data_filename, active_show_index)

            except:
                self.write(self.analysis_status, "Error: Unable to calculate Non Rad Rec")
                return

        elif (datatype == "PL"):
            try:
                rad_rec = param_values_dict["B"] * ((self.read_N(data_filename, active_show_index) + param_values_dict["N0"]) * (self.read_P(data_filename, active_show_index) + param_values_dict["P0"]) - \
                    param_values_dict["N0"] * param_values_dict["P0"])

                max = param_values_dict["Thickness"]
                dx = param_values_dict["dx"]
                distance = np.linspace(0, max - dx, data_m)
                alphaCof = param_values_dict["Alpha"] if not param_values_dict["ignore_alpha"] else 0
                thetaCof = param_values_dict["Theta"]
                delta_frac = param_values_dict["Delta"]
                fracEmitted = param_values_dict["Frac-Emitted"]
                
                # Make room for one more value than necessary - this value at index j+1 will be used to pad the upper correction
                distance_matrix = np.zeros((data_m, data_m))
                lf_distance_matrix = np.zeros((data_m, data_m))
                rf_distance_matrix = np.zeros((data_m, data_m))

                # Each row in weight will represent the weight function centered around a different position
                # Total reflection is assumed to occur at either end of the system: 
                # Left (x=0) reflection is equivalent to a symmetric wire situation while right reflection (x=thickness) is usually negligible
                for n in range(0, data_m):
                    distance_matrix[n] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:data_m - n]))
                    lf_distance_matrix[n] = distance + (n * dx)
                    rf_distance_matrix[n] = (max - distance) + (max - n * dx)

                weight = np.exp(-(alphaCof + thetaCof) * distance_matrix)
                lf_weight = np.exp(-(alphaCof + thetaCof) * lf_distance_matrix)
                rf_weight = np.exp(-(alphaCof + thetaCof) * rf_distance_matrix) * 0

                combined_weight = (1 - fracEmitted) * 0.5 * thetaCof * delta_frac * (weight + lf_weight + rf_weight)

                weight2 = np.exp(-(thetaCof) * distance_matrix)
                lf_weight2 = np.exp(-(thetaCof) * lf_distance_matrix)

                combined_weight2 = (1 - fracEmitted) * 0.5 * thetaCof * (1 - delta_frac) * (weight2 + lf_weight2)

                PL_base = fracEmitted * rad_rec

                for p in range(0, data_m):
                    # To each value of the slice add the attenuation contribution with weight centered around that value's corresponding position
                    PL_base[p] += intg.trapz(combined_weight[p] * rad_rec, dx=dx) + thetaCof * (1 - fracEmitted) * 0.5 * delta_frac * rad_rec[p] + \
                        intg.trapz(combined_weight2[p] * rad_rec, dx=dx) + thetaCof * (1 - fracEmitted) * 0.5 * (1 - delta_frac) * rad_rec[p]

                new_data = Data_Set(PL_base, data_node_x, data_node_x, data_edge_x, *(param_tuple), datatype, data_filename, active_show_index)

            except OSError:
                self.write(self.analysis_status, "Error: Unable to calculate PL")
                return

        try:
            active_datagroup.add(new_data, new_data.tag())

        except ValueError:
            self.write(self.analysis_status, "Error: dt or total t mismatch")
        return

    def a_plot(self, plot_ID):
        # Wrapper to apply read_data() on multiple selected datasets
        # THe Plot button on the Analyze tab calls this function
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

        
        active_plot.xlim = (0, active_plot.datagroup.get_max_x())
        active_plot.ylim = (active_plot.datagroup.get_maxval() * 1e-11, active_plot.datagroup.get_maxval() * 10)
        active_plot.xaxis_type = 'linear'
        active_plot.yaxis_type = 'log'
        self.data_plot(plot_ID, clear_plot=True)
        return

    def plot_tstep(self, plot_ID):
        # Step already plotted data forward (or backward) in time
        active_plot = self.analysis_plots[plot_ID]
        try:
            active_plot.add_time_index(int(self.main_tstep_entryboxes[plot_ID].get()))
        except ValueError:
            self.write(self.analysis_status, "Invalid number of time steps")
            return

        active_show_index = active_plot.time_index
        active_datagroup = active_plot.datagroup

        # Search data files for data at new time step
        if active_datagroup.type == "ΔN":
            for tag in active_datagroup.datasets:
                active_datagroup.datasets[tag].data = self.read_N(active_datagroup.datasets[tag].filename, active_show_index)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "ΔP":
            for tag in active_datagroup.datasets:
                active_datagroup.datasets[tag].data = self.read_P(active_datagroup.datasets[tag].filename, active_show_index)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "E-field":
            for tag in active_datagroup.datasets:
                active_datagroup.datasets[tag].data = self.read_E_field(active_datagroup.datasets[tag].filename, active_show_index)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "RR":
            for tag in active_datagroup.datasets:
                n0 = active_datagroup.datasets[tag].params_dict["N0"]
                p0 = active_datagroup.datasets[tag].params_dict["P0"]
                B = active_datagroup.datasets[tag].params_dict["B"]
                active_datagroup.datasets[tag].data = B \
                    * ((self.read_N(active_datagroup.datasets[tag].filename, active_show_index) + n0) \
                    * (self.read_P(active_datagroup.datasets[tag].filename, active_show_index) + p0) - n0 * p0)
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "NRR":
            for tag in active_datagroup.datasets:
                temp_N = self.read_N(active_datagroup.datasets[tag].filename, active_show_index)
                temp_P = self.read_P(active_datagroup.datasets[tag].filename, active_show_index)
                n0 = active_datagroup.datasets[tag].params_dict["N0"]
                p0 = active_datagroup.datasets[tag].params_dict["P0"]
                tauN = active_datagroup.datasets[tag].params_dict["Tau_N"]
                tauP = active_datagroup.datasets[tag].params_dict["Tau_P"]
                active_datagroup.datasets[tag].data = ((temp_N + n0) * (temp_P + p0) - n0 * p0) / ((tauN * (temp_P + p0)) + (tauP * (temp_N + n0)))
                active_datagroup.datasets[tag].show_index = active_show_index

        elif active_datagroup.type == "PL":
            for tag in active_datagroup.datasets:
                filename = active_datagroup.datasets[tag].filename
                B = active_datagroup.datasets[tag].params_dict["B"]
                n0 = active_datagroup.datasets[tag].params_dict["N0"]
                p0 = active_datagroup.datasets[tag].params_dict["P0"]
                rad_rec = B * ((self.read_N(filename, active_show_index) + n0) * (self.read_P(filename, active_show_index) + p0) - n0 * p0)

                max = active_datagroup.datasets[tag].params_dict["Thickness"]
                dx = active_datagroup.datasets[tag].params_dict["dx"]
                data_m = int(0.5 + max / dx)
                distance = np.linspace(0, max - dx, data_m)
                alphaCof = active_datagroup.datasets[tag].params_dict["Alpha"] if not active_datagroup.datasets[tag].params_dict["ignore_alpha"] else 0
                thetaCof = active_datagroup.datasets[tag].params_dict["Theta"]
                delta_frac = active_datagroup.datasets[tag].params_dict["Delta"]
                fracEmitted = active_datagroup.datasets[tag].params_dict["Frac-Emitted"]

                # Make room for one more value than necessary - this value at index j+1 will be used to pad the upper correction
                distance_matrix = np.zeros((data_m, data_m))
                lf_distance_matrix = np.zeros((data_m, data_m))
                rf_distance_matrix = np.zeros((data_m, data_m))

                # Each row in weight will represent the weight function centered around a different position
                # Total reflection is assumed to occur at either end of the system: 
                # Left (x=0) reflection is equivalent to a symmetric wire situation while right reflection (x=thickness) is usually negligible
                for n in range(0, data_m):
                    distance_matrix[n] = np.concatenate((np.flip(distance[0:n+1], 0), distance[1:data_m - n]))
                    lf_distance_matrix[n] = distance + (n * dx)
                    rf_distance_matrix[n] = (max - distance) + (max - n * dx)

                weight = np.exp(-(alphaCof + thetaCof) * distance_matrix)
                lf_weight = np.exp(-(alphaCof + thetaCof) * lf_distance_matrix)
                rf_weight = np.exp(-(alphaCof + thetaCof) * rf_distance_matrix) * 0

                combined_weight = (1 - fracEmitted) * 0.5 * thetaCof * delta_frac * (weight + lf_weight + rf_weight)

                weight2 = np.exp(-(thetaCof) * distance_matrix)
                lf_weight2 = np.exp(-(thetaCof) * lf_distance_matrix)

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

        self.data_plot(plot_ID, clear_plot=True)
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
            test_simtime = float(self.simtime_entry.get())      # [ns]
            test_dt = float(self.dt_entry.get())           # [ns]

            if (test_simtime <= 0): raise Exception("Error: Invalid simulation time")
            if (test_dt <= 0 or test_dt > test_simtime): raise Exception("Error: Invalid dt")
        
        except ValueError:
            self.write(self.status, "Error: Invalid parameters")
            return

        except Exception as oops:
            self.write(self.status, oops)
            return

        IC_files = tk.filedialog.askopenfilenames(initialdir = self.default_dirs["Initial"], title="Select IC text file", filetypes=[("Text files","*.txt")])
        if (IC_files.__len__() == 0): return

        batch_num = 0
        for IC in IC_files:
            batch_num += 1
            self.IC_file_name = IC
            self.load_ICfile()
            self.write(self.status, "Now calculating {} : ({} of {})".format(self.IC_file_name[self.IC_file_name.rfind("/") + 1:self.IC_file_name.rfind(".txt")], str(batch_num), str(IC_files.__len__())))
            self.do_Calculate()
            time.sleep(2)

        return

	# The big function that does all the simulating
    def do_Calculate(self):
        ## Setup parameters
        try:

            self.thickness = float(self.thickness_entry.get())  # [nm]
            self.simtime = float(self.simtime_entry.get())      # [ns]
            self.dx = float(self.dx_entry.get())                # [nm]
            self.dt = float(self.dt_entry.get())                # [ns]

            if (self.simtime <= 0): raise Exception("Error: Invalid simulation time")
            if (self.dt <= 0 or self.dt > self.simtime): raise Exception("Error: Invalid dt")

            self.m = int(0.5 + self.thickness / self.dx)         # Number of space steps
            self.n = int(0.5 + self.simtime / self.dt)           # Number of time steps

            # Upper limit on number of time steps
            if (self.n > 2.5e5): raise Exception("Error: too many time steps")

            self.sf = float(self.Sf_entry.get())
            self.sb = float(self.Sb_entry.get())

            self.mu_N = float(self.N_mobility_entry.get())
            self.mu_P = float(self.P_mobility_entry.get())
            self.n0 = float(self.n0_entry.get())
            self.p0 = float(self.p0_entry.get())
            self.B_param = float(self.B_entry.get())
            self.tauNeg = float(self.tauN_entry.get())
            self.tauPos = float(self.tauP_entry.get())
            self.temperature = float(self.temperature_entry.get())                    # [K]
            self.rel_permitivity = float(self.rel_permitivity_entry.get())
            self.ext_E_field = float(self.ext_efield_entry.get())
            self.vac_permitivity = 8.854 * 1e-12 * (1e-9)                      # [F/m] to [F/nm]

            self.alphaCof = float(self.alpha_entry.get())
            self.thetaCof = float(self.theta_entry.get())
            self.fracEmitted = float(self.frac_emitted_entry.get())
            self.delta_frac = float(self.delta_entry.get())


            temp_sim_dict = {"Mu_N": self.mu_N, "Mu_P": self.mu_P, "N0": self.n0, "P0": self.p0, "Thickness": self.thickness, "dx": self.dx, \
                        "B": self.B_param, "Tau_N": self.tauNeg, "Tau_P": self.tauPos, "Sf": self.sf, "Sb": self.sb, "Temperature": self.temperature, \
                        "Rel-Permitivity": self.rel_permitivity, "Ext_E-Field": self.ext_E_field, \
                        "Theta": self.thetaCof, "Alpha": self.alphaCof, "Delta": self.delta_frac, "Frac-Emitted": self.fracEmitted}

            # Convert into TEDs units
            for param in temp_sim_dict:
                temp_sim_dict[param] = temp_sim_dict[param] * self.convert_in_dict[param]

        except ValueError:
            self.write(self.status, "Error: Invalid parameters")
            return

        except Exception as oops:
            self.write(self.status, oops)
            return

        try:
            # Construct the data folder's name from the corresponding IC file's name
            shortened_IC_name = self.IC_file_name[self.IC_file_name.rfind("/") + 1:self.IC_file_name.rfind(".txt")]
            data_file_name = shortened_IC_name

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
            return

        try:
            ## Calculate!
            atom = tables.Float64Atom()

            # Create data files
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

            self.edge_x = np.linspace(0,self.thickness,self.m+1)
            self.node_x = np.linspace(self.dx/2, self.thickness - self.dx/2, self.m)
            self.N_limits = (np.amax(self.init_N) * 1e-11, np.amax(self.init_N) * 10)
            self.P_limits = (np.amax(self.init_P) * 1e-11, np.amax(self.init_P) * 10)

            self.sim_N = self.init_N
            self.sim_P = self.init_P
            self.sim_E_field = self.init_E_field
            self.sim_Ec = self.init_Ec
            self.sim_Chi = self.init_Chi
            self.update_data_plots(0)

            self.write(self.status, "Now calculating ΔN, ΔP")
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


            #    self.update_data_plots(int(self.n * i / self.numPartitions), self.numPartitions > 20)
                #self.update_err_plots()

            # TODO: Why don't we just pass in the whole dictionary and let ode_nanowire extract the values?
            if self.check_ignore_recycle.get():
                error_dict = finite.ode_nanowire(full_path_name,data_file_name,self.m,self.n - numTimeStepsDone,self.dx,self.dt, temp_sim_dict["Sf"], temp_sim_dict["Sb"], 
                                                 temp_sim_dict["Mu_N"], temp_sim_dict["Mu_P"], temp_sim_dict["Temperature"], temp_sim_dict["N0"], temp_sim_dict["P0"], 
                                                 temp_sim_dict["Tau_N"], temp_sim_dict["Tau_P"], temp_sim_dict["B"], temp_sim_dict["Rel-Permitivity"], self.vac_permitivity,
                                                 not self.check_ignore_recycle.get(), self.check_do_ss.get(), 0, temp_sim_dict["Theta"], temp_sim_dict["Delta"], temp_sim_dict["Frac-Emitted"],
                                                 temp_sim_dict["Ext_E-Field"], self.init_N, self.init_P, self.init_E_field, self.init_Ec, self.init_Chi)
            
            else:
                error_dict = finite.ode_nanowire(full_path_name,data_file_name,self.m,self.n - numTimeStepsDone,self.dx,self.dt, temp_sim_dict["Sf"], temp_sim_dict["Sb"], 
                                                 temp_sim_dict["Mu_N"], temp_sim_dict["Mu_P"], temp_sim_dict["Temperature"], temp_sim_dict["N0"], temp_sim_dict["P0"], 
                                                 temp_sim_dict["Tau_N"], temp_sim_dict["Tau_P"], temp_sim_dict["B"], temp_sim_dict["Rel-Permitivity"], self.vac_permitivity,
                                                 not self.check_ignore_recycle.get(), self.check_do_ss.get(), temp_sim_dict["Alpha"], temp_sim_dict["Theta"], temp_sim_dict["Delta"], temp_sim_dict["Frac-Emitted"],
                                                 temp_sim_dict["Ext_E-Field"], self.init_N, self.init_P, self.init_E_field, self.init_Ec, self.init_Chi)
            
            grid_t = np.linspace(self.dt, self.simtime, self.n)

            try:
                np.savetxt(full_path_name + "\\convergence.csv", np.vstack((grid_t, error_dict['hu'], error_dict['tcur'],\
                    error_dict['tolsf'], error_dict['tsw'], error_dict['nst'], error_dict['nfe'], error_dict['nje'], error_dict['nqu'],\
                    error_dict['mused'])).transpose(), fmt='%.4e', delimiter=',', header="t, hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, mused")
            except PermissionError:
                self.write(self.status, "Error: unable to access convergence data export destination")
            self.write(self.status, "Finalizing...")

            #self.read_TS(self.n)
            #self.update_data_plots(self.n, self.numPartitions > 20)

            for i in range(1,6):
                self.read_TS(data_file_name, int(self.n * i / 5))
                self.update_data_plots(self.n, do_clear_plots=False)

            time.sleep(3)
            self.write(self.status, "Simulations complete")
            
        except FloatingPointError:
            self.write(self.status, "Overflow detected - calculation aborted")
            return

        # Save metadata: list of param values used for the simulation
        # Inverting the unit conversion between the inputted params and the calculation engine is also necessary to regain the originally inputted param values

        with open(full_path_name + "\\metadata.txt", "w+") as ofstream:
            ofstream.write("$$ METADATA FOR CALCULATIONS PERFORMED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
            for param in temp_sim_dict:
                ofstream.write("{}: {}\n".format(param, (temp_sim_dict[param] * self.convert_out_dict[param])))

            # The following params are exclusive to metadata files
            ofstream.write("Total-Time: " + str(self.simtime) + "\n")
            ofstream.write("dt: " + str(self.dt) + "\n")
            if self.check_ignore_recycle.get(): ofstream.write("ignore_alpha: 1\n")
            else: ofstream.write("ignore_alpha: 0\n")


        return

    def do_Integrate(self, plot_ID):
        self.write(self.analysis_status, "")

        active_plot = self.analysis_plots[plot_ID]
        if active_plot.datagroup.datasets.__len__() == 0: return

        # Collect instructions from user using a series of popup windows
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
            self.I_plot.x_param = self.xaxis_param

        else:
            self.I_plot.x_param = "Time"
            
        # Clean up the I_plot and prepare to integrate given selections
        # A lot of the following is a data transfer between the sending active_datagroup and the receiving I_plot
        self.I_plot.clear()
        self.I_plot.mode = self.PL_mode
        self.I_plot.global_gridx = None

        active_datagroup = active_plot.datagroup
        n = active_datagroup.get_maxnumtsteps()
        counter = 0
        # Integrate for EACH dataset in chosen datagroup
        for tag in active_datagroup.datasets:
            data_filename = active_datagroup.datasets[tag].filename
            print("Now integrating {}".format(data_filename))

            # Unpack needed params from the dictionaries of params
            dx = active_datagroup.datasets[tag].params_dict["dx"]
            total_length = active_datagroup.datasets[tag].params_dict["Thickness"]
            total_time = active_datagroup.datasets[tag].params_dict["Total-Time"]
            B_param = active_datagroup.datasets[tag].params_dict["B"]
            n0 = active_datagroup.datasets[tag].params_dict["N0"]
            p0 = active_datagroup.datasets[tag].params_dict["P0"]
            tauN = active_datagroup.datasets[tag].params_dict["Tau_N"]
            tauP = active_datagroup.datasets[tag].params_dict["Tau_P"]
            alpha = active_datagroup.datasets[tag].params_dict["Alpha"] if (active_datagroup.datasets[tag].params_dict["ignore_alpha"] == 0.0) else 0
            theta = active_datagroup.datasets[tag].params_dict["Theta"]
            delta = active_datagroup.datasets[tag].params_dict["Delta"]
            frac_emitted = active_datagroup.datasets[tag].params_dict["Frac-Emitted"]

            if self.PL_mode == "Current time step":
                show_index = active_datagroup.datasets[tag].show_index

            # Clean up any bounds that extend past the confines of the system

            for bounds in self.integration_bounds:
               
                if (bounds[1] > total_length):
                    bounds[1] = total_length

                if (bounds[0] < 0):
                    bounds[0] = 0

            print("Bounds after cleanup: {}".format(self.integration_bounds))

            #if (self.integration_lbound_entry.get() == "f"):
            #    self.PL = np.zeros((boundList.__len__(), int(n) + 1))
            #    for i in range(boundList.__len__()):
            #        self.PL[i] = finite.propagatingPL(data_filename, boundList[i], boundList[i], dx, 0, total_length - dx, B_param, n0, p0, alpha, theta, delta, frac_emitted)

            #elif (self.integration_lbound_entry.get() == "g"):
            #    self.PL = np.zeros((boundList.__len__(), int(n) + 1))
            #    for i in range(boundList.__len__()):
            #        self.PL[i] = finite.propagatingPL(data_filename, boundList[i] - 500, boundList[i] + 500, dx, 0, total_length - dx, B_param, n0, p0, alpha, theta, delta, frac_emitted)
            #        if boundList[i] == 0:
            #            self.PL[i] *= 2
            for bounds in self.integration_bounds:
                if (active_datagroup.datasets[tag].type == "ΔN"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-n.h5", mode='r') as ifstream_N:
                        data = ifstream_N.root.N
                        I_data = finite.integrate(data, bounds[0], bounds[1], dx, total_length)
            
                elif (active_datagroup.datasets[tag].type == "ΔP"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-p.h5", mode='r') as ifstream_P:
                        data = ifstream_P.root.P
                        I_data = finite.integrate(data, bounds[0], bounds[1], dx, total_length)

                elif (active_datagroup.datasets[tag].type == "E-field"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-E_field.h5", mode='r') as ifstream_E_field:
                        data = ifstream_E_field.root.E_field
                        I_data = finite.integrate(data, bounds[0], bounds[1], dx, total_length)

                elif (active_datagroup.datasets[tag].type == "RR"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-n.h5", mode='r') as ifstream_N, \
                        tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-p.h5", mode='r') as ifstream_P:
                        temp_N = np.array(ifstream_N.root.N)
                        temp_P = np.array(ifstream_P.root.P)

                        data = B_param * (temp_N + n0) * (temp_P + p0) - n0 * p0
                        I_data = finite.integrate(data, bounds[0], bounds[1], dx, total_length)

                elif (active_datagroup.datasets[tag].type == "NRR"):
                    with tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-n.h5", mode='r') as ifstream_N, \
                        tables.open_file(self.default_dirs["Data"] + "\\" + data_filename + "\\" + data_filename + "-p.h5", mode='r') as ifstream_P:
                        temp_N = np.array(ifstream_N.root.N)
                        temp_P = np.array(ifstream_P.root.P)
                        data = ((temp_N + n0) * (temp_P + p0) - n0 * p0) / (tauN * (temp_P + p0) + tauP * (temp_N + n0))
                        I_data = finite.integrate(data, bounds[0], bounds[1], dx, total_length)

                else:
                    I_data = finite.propagatingPL(data_filename, bounds[0], bounds[1], dx, 0, total_length, B_param, n0, p0, alpha, theta, delta, frac_emitted)
            

                if self.PL_mode == "Current time step":
                    # FIXME: We don't need to integrate everything just to extract a single time step
                    # Change the integration procedure above to work only with the needed data
                    I_data = I_data[show_index]

                    # Don't forget to change out of TEDs units, or the x axis won't match the parameters the user typed in
                    grid_xaxis = float(active_datagroup.datasets[tag].params_dict[self.xaxis_param] * self.convert_out_dict[self.xaxis_param])

                    xaxis_label = self.xaxis_param + " [WIP]"

                elif self.PL_mode == "All time steps":
                    self.I_plot.global_gridx = np.linspace(0, total_time, n + 1)
                    grid_xaxis = -1 # A dummy value for the I_Set constructor
                    xaxis_label = "Time [ns]"

                self.I_plot.add(I_Set(I_data, grid_xaxis, active_datagroup.datasets[tag].params_dict, active_datagroup.datasets[tag].type, tips(data_filename, 4) + "__" + str(bounds[0]) + "_to_" + str(bounds[1])))

                counter += 1
                print("Integration: {} of {} complete".format(counter, active_datagroup.size() * self.integration_bounds.__len__()))

            
        plot.figure(9)
        plot.clf()
        
        max = self.I_plot.get_maxval()
        
        self.I_plot.xaxis_type = 'linear'
        self.I_plot.yaxis_type = 'log'
        self.I_plot.ylim = max * 1e-12, max * 10

        plot.yscale(self.I_plot.yaxis_type)
        plot.ylim(self.I_plot.ylim)
        plot.xlabel(xaxis_label)
        plot.ylabel(self.I_plot.type)
        plot.title("Total {} from {} nm to {} nm".format(self.I_plot.type, self.integration_lbound, self.integration_ubound))

        for key in self.I_plot.I_sets:

            if self.PL_mode == "Current time step":
                plot.scatter(self.I_plot.I_sets[key].grid_x, self.I_plot.I_sets[key].I_data, label=self.I_plot.I_sets[key].tag())

            elif self.PL_mode == "All time steps":
                plot.plot(self.I_plot.global_gridx, self.I_plot.I_sets[key].I_data, label=self.I_plot.I_sets[key].tag())
                self.I_plot.xlim = (0, np.amax(self.I_plot.global_gridx))
                
        plot.legend()
        plot.tight_layout()
        self.main_fig3.canvas.draw()
        
        self.write(self.analysis_status, "Integration complete")

        return

    ## Initial Condition Managers

    def reset_IC(self, force=False):
        # Clears and uninitializes all fields and saved data on IC tab
        cleared_items = ""
        if self.check_reset_params.get() or force:
            for key in self.sys_param_entryboxes_dict:
                self.enter(self.sys_param_entryboxes_dict[key], "")

            cleared_items += " Params,"

        if self.check_reset_inits.get() or force:
            self.deleteall_HIC()
            self.IC_is_AIC = False
            self.thickness = None
            self.dx = None
            self.init_N = None
            self.init_P = None
            self.init_Ec = None
            self.init_Chi = None
            self.set_thickness_and_dx_entryboxes(state='unlock')
            plot.figure(1)
            plot.clf()
            self.init_canvas_NP.draw()
            plot.figure(2)
            plot.clf()
            self.init_fig_ec.canvas.draw()
            cleared_items += " Inits"

        self.write(self.ICtab_status, "Cleared:{}".format(cleared_items))
        return

    def check_IC_initialized(self):
        # Helper to make sure IC arrays are initialized to some dummy values at least—allows plotter to plot some IC even if not all IC variables (N, P, Ec, Chi) are assigned yet
        # This is mainly used when the program is first started up and the IC arrays don't have values yet
        if self.init_N is None:
            self.init_N = np.zeros(int(0.5 + self.thickness / self.dx))
        if self.init_P is None:
            self.init_P = np.zeros(int(0.5 + self.thickness / self.dx))
        if self.init_Ec is None:
            self.init_Ec = np.zeros(int(0.5 + self.thickness / self.dx) + 1)
        if self.init_Chi is None:
            self.init_Chi = np.zeros(int(0.5 + self.thickness / self.dx) + 1)

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
        #if (float(self.thickness_entry.get()) != self.thickness or float(self.dx_entry.get()) != self.dx):
        #    raise Exception("Error: Thickness or space step size has been altered - reset the initial condition to use new space mesh")

        self.thickness = float(self.thickness_entry.get())
        self.dx = float(self.dx_entry.get())

        if (self.thickness <= 0 or self.dx <= 0): raise ValueError

        if not finite.check_valid_dx(self.thickness, self.dx):
            raise Exception("Error: space step size larger than thickness")

        # Upper limit on number of space steps
        if (int(0.5 + self.thickness / self.dx) > 1e6): 
            raise Exception("Error: too many space steps")

            
        self.init_x = np.linspace(self.dx / 2,self.thickness - self.dx / 2, int(0.5 + self.thickness / self.dx))
        self.init_x_edges = np.linspace(0, self.thickness, int(0.5 + self.thickness / self.dx) + 1)
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
        AIC_options = {"long_expfactor":self.calculate_init_material_expfactor.get(), 
                     "incidence":self.AIC_stim_mode.get(),
                     "power_mode":self.AIC_gen_power_mode.get()}
        try:
            if AIC_options["long_expfactor"] == '' or AIC_options["power_mode"] == '':
                raise ValueError("Error: select material param and power generation options ")
        except ValueError as oops:
            self.write(self.ICtab_status, oops)
            return

        # Flush HICs, as AIC will overwrite the entire spatial mesh
        self.deleteall_HIC(False)

        # Establish constants; calculate alpha
        # FIXME: Units
        h = 6.626e-34   # [J*s]
        c = 2.997e8     # [m/s]
        hc_evnm = h * c * 6.241e18 * 1e9    # [J*m] to [eV*nm]
        hc_nm = h * c * 1e9     # [J*m] to [J*nm] 
        try: wavelength = float(self.pulse_wavelength_entry.get())              # [nm]
        except ValueError:
            self.write(self.ICtab_status, "Error: missing or invalid pulsed laser wavelength")
            return

        if (AIC_options["long_expfactor"]):
            try: A0 = float(self.A0_entry.get())         # [cm^-1 eV^-1/2] or [cm^-1 eV^-2]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid A0")
                return

            try: Eg = float(self.Eg_entry.get())                  # [eV]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing or invalid Eg")
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

            if (self.pulse_freq_entry.get() == "cw"):
                freq = 1
            else:
                try:
                    freq = float(self.pulse_freq_entry.get()) * 1e3    # [kHz] to [1/s]

                except ValueError:
                    self.write(self.ICtab_status, "Error: missing or invalid pulse frequency")
                    return

            self.init_N = finite.pulse_laser_power_spotsize(power, spotsize, freq, wavelength, alpha_nm, self.init_x, hc=hc_nm)
        
        elif (AIC_options["power_mode"] == "density"):
            try: power_density = float(self.power_density_entry.get()) * 1e-6 * ((1e-7) ** 2)  # [uW / cm^2] to [J/s nm^2]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing power density")
                return

            if (self.pulse_freq_entry.get() == "cw"):
                freq = 1
            else:
                try:
                    freq = float(self.pulse_freq_entry.get()) * 1e3    # [kHz] to [1/s]

                except ValueError:
                    self.write(self.ICtab_status, "Error: missing or invalid pulse frequency")
                    return

            self.init_N = finite.pulse_laser_powerdensity(power_density, freq, wavelength, alpha_nm, self.init_x, hc=hc_nm)
        
        elif (AIC_options["power_mode"] == "max-gen"):
            try: max_gen = float(self.max_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing max gen")
                return

            self.init_N = finite.pulse_laser_maxgen(max_gen, alpha_nm, self.init_x)
        

        elif (AIC_options["power_mode"] == "total-gen"):
            try: total_gen = float(self.total_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing total gen")
                return

            self.init_N = finite.pulse_laser_totalgen(total_gen, self.thickness, alpha_nm, self.init_x)
        
        else:
            self.write(self.ICtab_status, "An unexpected error occurred while calculating the power generation params")
            return

        ## Assuming that the initial distributions of holes and electrons are identical
        self.init_P = self.init_N
        self.update_IC_plot()
        self.IC_is_AIC = True
        return

    ## Special functions for Heuristic Init:
    def calcHeuristic(self, condition, initArray):
        # Translates the parameters of a HIC into IC arrays
        if (condition.type == "POINT"):
            initArray[finite.toIndex(condition.l_bound, self.dx, self.thickness, condition.is_edge())] = condition.l_boundval

        elif (condition.type == "FILL"):
            i = finite.toIndex(condition.l_bound, self.dx, self.thickness, condition.is_edge())
            j = finite.toIndex(condition.r_bound, self.dx, self.thickness, condition.is_edge())
            initArray[i:j+1] = condition.l_boundval

        elif (condition.type == "LINE"):
            slope = (condition.r_boundval - condition.l_boundval) / (condition.r_bound - condition.l_bound)
            i = finite.toIndex(condition.l_bound, self.dx, self.thickness, condition.is_edge())
            j = finite.toIndex(condition.r_bound, self.dx, self.thickness, condition.is_edge())

            ndx = np.linspace(0, self.dx * (j - i), j - i + 1)
            initArray[i:j+1] = condition.l_boundval + ndx * slope

        elif (condition.type == "EXP"):
            i = finite.toIndex(condition.l_bound, self.dx, self.thickness, condition.is_edge())
            j = finite.toIndex(condition.r_bound, self.dx, self.thickness, condition.is_edge())

            ndx = np.linspace(0, j - i, j - i + 1)
            try:
                initArray[i:j+1] = condition.l_boundval * np.power(condition.r_boundval / condition.l_boundval, ndx / (j - i))
            except FloatingPointError:
                print("Warning: Step size too large to resolve initial condition accurately")

        return initArray

    def recalc_HIC(self):
        # Recalculate IC arrays from every condition, sequentially, in the HIC list
        # Used when HIC list is updated
        try:
            self.set_init_x()

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        self.check_IC_initialized()
        for condition in self.HIC_list:
            if (condition.variable == "ΔN"):
                self.init_N = self.calcHeuristic(condition, self.init_N)

            elif (condition.variable == "ΔP"):
                self.init_P = self.calcHeuristic(condition, self.init_P)

            elif (condition.variable == "dEc"):
                self.init_Ec = self.calcHeuristic(condition, self.init_Ec)

            elif (condition.variable == "chi"):
                self.init_Chi = self.calcHeuristic(condition, self.init_Chi)
        return

    def uncalc_HIC(self, IC_var, lbound, rbound, on_point, is_edge=False):
        # (Attempts to) undo effect of adding a particular HIC
        # Only used when user deletes HIC
        if (IC_var == "ΔN"):
            if on_point:
                self.init_N[finite.toIndex(lbound, self.dx, self.thickness, is_edge)] = 0
            else:
                self.init_N[finite.toIndex(lbound, self.dx, self.thickness, is_edge):finite.toIndex(rbound, self.dx, self.thickness, is_edge) + 1] = 0
        elif (IC_var == "ΔP"):
            if on_point:
                self.init_P[finite.toIndex(lbound, self.dx, self.thickness, is_edge)] = 0
            else:
                self.init_P[finite.toIndex(lbound, self.dx, self.thickness, is_edge):finite.toIndex(rbound, self.dx, self.thickness, is_edge) + 1] = 0

        elif (IC_var == "dEc"):
            if on_point:
                self.init_Ec[finite.toIndex(lbound, self.dx, self.thickness, is_edge)] = 0
            else:
                self.init_Ec[finite.toIndex(lbound, self.dx, self.thickness, is_edge):finite.toIndex(rbound, self.dx, self.thickness, is_edge) + 1] = 0
        elif (IC_var == "chi"):
            if on_point:
                self.init_Chi[finite.toIndex(lbound, self.dx, self.thickness, is_edge)] = 0
            else:
                self.init_Chi[finite.toIndex(lbound, self.dx, self.thickness, is_edge):finite.toIndex(rbound, self.dx, self.thickness, is_edge) + 1] = 0

        return

    def add_HIC(self):
        # Create new Initial_Condition object and add to list
        if (self.HIC_list.__len__() > 0 and isinstance(self.HIC_list[0], str)):
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
            if (self.init_shape_selection.get() == "POINT"):

                if (float(self.HIC_lbound_entry.get()) < 0):
                    raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                newInitCond = Initial_Condition(self.init_var_selection.get(), "POINT", float(self.HIC_lbound_entry.get()), -1, float(self.HIC_lvalue_entry.get()), -1)

            elif (self.init_shape_selection.get() == "FILL"):
                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) > self.thickness):
                	raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                newInitCond = Initial_Condition(self.init_var_selection.get(), "FILL", float(self.HIC_lbound_entry.get()), float(self.HIC_rbound_entry.get()), float(self.HIC_lvalue_entry.get()), -1)

            elif (self.init_shape_selection.get() == "LINE"):
                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) > self.thickness):
                	raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) > float(self.HIC_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                newInitCond = Initial_Condition(self.init_var_selection.get(), "LINE", float(self.HIC_lbound_entry.get()), float(self.HIC_rbound_entry.get()), float(self.HIC_lvalue_entry.get()), float(self.HIC_rvalue_entry.get()))

            elif (self.init_shape_selection.get() == "EXP"):
                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) > self.thickness):
                    raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.HIC_lbound_entry.get()) > float(self.HIC_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.HIC_lbound_entry.get()) < 0 or float(self.HIC_rbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                newInitCond = Initial_Condition(self.init_var_selection.get(), "EXP", float(self.HIC_lbound_entry.get()), float(self.HIC_rbound_entry.get()), float(self.HIC_lvalue_entry.get()), float(self.HIC_rvalue_entry.get()))

            else:
                raise Exception("Error: No init. type selected")

        except ValueError:
            self.write(self.ICtab_status, "Error: Missing Parameters")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        self.HIC_list.append(newInitCond)
        self.HIC_listbox.insert(self.HIC_list.__len__() - 1, newInitCond.get())
        self.recalc_HIC()
        self.IC_is_AIC = False
        self.update_IC_plot()
        return

	# Repositions HICs in their list
	# HICs are applied sequentially and do overwrite each other, so this CAN affect what the initial condition looks like
    def moveup_HIC(self):
        currentSelectionIndex = self.HIC_listbox.curselection()[0]
        
        if (currentSelectionIndex > 0):
            self.HIC_list[currentSelectionIndex], self.HIC_list[currentSelectionIndex - 1] = self.HIC_list[currentSelectionIndex - 1], self.IC_list[currentSelectionIndex]
            self.HIC_listbox.delete(currentSelectionIndex)
            self.HIC_listbox.insert(currentSelectionIndex - 1, self.HIC_list[currentSelectionIndex - 1].get())
            self.HIC_listbox.selection_set(currentSelectionIndex - 1)
            self.recalc_HIC()
            self.update_IC_plot()
        return

    def movedown_HIC(self):
        currentSelectionIndex = self.HIC_listbox.curselection()[0] + 1
        
        if (currentSelectionIndex < self.HIC_list.__len__()):
            self.HIC_list[currentSelectionIndex], self.HIC_list[currentSelectionIndex - 1] = self.HIC_list[currentSelectionIndex - 1], self.IC_list[currentSelectionIndex]
            self.HIC_listbox.delete(currentSelectionIndex)
            self.HIC_listbox.insert(currentSelectionIndex - 1, self.HIC_list[currentSelectionIndex - 1].get())
            self.HIC_listbox.selection_set(currentSelectionIndex)
            self.recalc_HIC()
            self.update_IC_plot()
        return

    # Wrapper - Call delete_HIC until HIC list box is empty
    def deleteall_HIC(self, doPlotUpdate=True):
        while (self.HIC_list.__len__() > 0):
            # These first two lines mimic user repeatedly selecting topmost HIC in listbox
            self.HIC_listbox.select_set(0)
            self.HIC_listbox.event_generate("<<ListboxSelect>>")

            self.delete_HIC()
        return

    # Remove user-selected HIC from box and scrub its calculated values from stored IC arrays
    def delete_HIC(self):
        if (self.HIC_list.__len__() > 0):
            try:
                print("Deleted init. cond. #" + str(self.HIC_listbox.curselection()[0]))
                condition_to_delete = self.HIC_list[self.HIC_listbox.curselection()[0]]
                self.uncalc_HIC(condition_to_delete.variable, condition_to_delete.l_bound, condition_to_delete.r_bound, condition_to_delete.type == "POINT", condition_to_delete.is_edge())
                self.HIC_list.pop(self.HIC_listbox.curselection()[0])
                self.HIC_listbox.delete(self.HIC_listbox.curselection()[0])
                self.recalc_HIC()
                self.update_IC_plot()
            except IndexError:
                self.write(self.ICtab_status, "No HIC selected")
                return
        return

    # Fill IC arrays using list from .txt file
    def add_EIC(self):
        warning_flag = False
        IC_type = self.EIC_var_selection.get()
        is_edge = (IC_type == "dEc" or IC_type == "chi")
        valuelist_filename = tk.filedialog.askopenfilename(title="Select Values from text file", filetypes=[("Text files","*.txt")])
        if valuelist_filename == "": # If no file selected
            return

        IC_values_list = []
        with open(valuelist_filename, 'r') as ifstream:
            for line in ifstream:
                if (line == "" or "#" in line): continue

                else: IC_values_list.append(line.strip('\n'))

        try:
            self.set_init_x()

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return
           
        temp_IC_values = np.zeros(int(0.5 + self.thickness / self.dx)) if not is_edge else np.zeros(int(0.5 + self.thickness / self.dx) + 1)

        try:
            IC_values_list.sort(key = lambda x:float(x[0:x.find('\t')]))
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
            if (first_valueset[0] <= self.dx / 2):
                intermediate_x_indices = np.arange(finite.toIndex(first_valueset[0], self.dx, self.thickness, is_edge), finite.toIndex(second_valueset[0], self.dx, self.thickness, is_edge) + 1, 1)

            else:
                intermediate_x_indices = np.arange(finite.toIndex(first_valueset[0], self.dx, self.thickness, is_edge) + 1, finite.toIndex(second_valueset[0], self.dx, self.thickness, is_edge) + 1, 1)

            for j in intermediate_x_indices: # y-y0 = (y1-y0)/(x1-x0) * (x-x0)
                try:
                    temp_IC_values[j] = first_valueset[1] + (finite.toCoord(j, self.dx) - first_valueset[0]) * (second_valueset[1] - first_valueset[1]) / (second_valueset[0] - first_valueset[0])
                except IndexError:
                    self.write(self.ICtab_status, "Warning: some points out of bounds")
                    warning_flag = True
                except:
                    temp_IC_values[j] = 0
                    warning_flag = True
                

        if (IC_type == "ΔN"):
            self.init_N = np.copy(temp_IC_values)
        elif (IC_type == "ΔP"):
            self.init_P = np.copy(temp_IC_values)
        elif (IC_type == "dEc"):
            self.init_Ec = np.copy(temp_IC_values)
        elif (IC_type == "chi"):
            self.init_Chi = np.copy(temp_IC_values)

        self.IC_is_AIC = False
        self.update_IC_plot(warn=warning_flag)
        return

    ## Replot IC plots with updated IC arrays
    def update_IC_plot(self, warn=False):
        self.check_IC_initialized()
        plot.figure(1)
        plot.clf()
        plot.yscale('log')

        max_N = np.amax(self.init_N) * ((1e7) ** 3)
        max_P = np.amax(self.init_P) * ((1e7) ** 3)
        largest_initValue = max(max_N, max_P)
        plot.ylim((largest_initValue + 1e-30) * 1e-12, (largest_initValue + 1e-30) * 1e4)

        plot.plot(self.init_x, self.init_N * ((1e7) ** 3), label="delta_N") # [per nm^3] to [per cm^3]
        plot.plot(self.init_x, self.init_P * ((1e7) ** 3), label="delta_P")

        plot.xlabel("x [nm]")
        plot.ylabel("ΔN, ΔP [per cm^-3]")
        plot.title("Initial ΔN, ΔP Distribution")
        plot.legend()
        plot.tight_layout()
        self.init_fig_NP.canvas.draw()

        plot.figure(2)
        plot.clf()
        plot.yscale('log')

        max_Ec = np.amax(self.init_Ec)
        max_Chi = np.amax(self.init_Chi)
        largest_initValue = max(max_Ec, max_Chi)
        plot.ylim((largest_initValue + 1e-30) * 1e-12, (largest_initValue + 1e-30) * 1e4)

        plot.plot(self.init_x_edges, self.init_Ec, label="Eg")
        plot.plot(self.init_x_edges, self.init_Chi, label="e- aff.")

        plot.xlabel("x [nm]")
        plot.ylabel("Eg, e- aff. [TODO]")
        plot.title("Initial Eg, Electron Affinity Distribution")
        plot.legend()
        plot.tight_layout()
        self.init_fig_ec.canvas.draw()

        if not warn: self.write(self.ICtab_status, "Initial Condition Updated")
        return

    ## Initial Condition I/O

    def test_entryboxes_valid(self, entrybox_list):
        # Check if all system parameters are filled in
        self.missing_params = []
        for key in entrybox_list:
            try:
                test_var = float(entrybox_list[key].get())
                if (test_var == ""): raise TypeError
            except:
                self.missing_params.append(key)

        return (self.missing_params.__len__() == 0)

    def create_batch_init(self):
        try:
            batch_values = extract_values(self.batch_param_entry.get(), ' ')
        except ValueError:
            self.write(self.batch_status, "Error: Invalid batch values")
            return

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

        appended_sys_param_entryboxes_dict = dict(self.sys_param_entryboxes_dict)

        if self.IC_is_AIC:
            # dict.update(dict2) adds the contents of dict2 to dict, overwriting values in dict if needed
            appended_sys_param_entryboxes_dict.update(dict(self.analytical_entryboxes_dict))

        try:
            for batch_value in batch_values:
                self.enter(appended_sys_param_entryboxes_dict[self.batch_param.get()], str(batch_value))
                
                # It turns out that the previous batch tool did not actually support differing IC distributions, only
                # different parameter sets with the same IC distribution. To make IC distributions batchable, we
                # simply call a function to calculate the new IC every time a new parameter that changes the IC is entered

                # So far this only applies to the AIC
                if self.IC_is_AIC:
                    self.add_AIC()
                # For whatever reason, tk.filedialog.asksaveasfilename will automatically append .txt to the file name before passing to self.write_init_file(). 
                # This behavior is NOT done by default, so here the .txt is manually appended to the file name.
                self.write_init_file("{}\\{}\\{}{:.0e}.txt".format(self.default_dirs["Initial"], batch_dir_name, self.batch_param.get(),batch_value))
        except KeyError:
            self.write(self.batch_status, "Select a parameter from the drop-down menu")
            return

        self.write(self.batch_status, "Batch {} created successfully".format(batch_dir_name))

        return

	# Wrapper for write_init_file() - this one is for IC files user saves from the Initial tab and is called when the Save button is clicked
    def save_ICfile(self):
        try:
            # Check that user has filled in all parameters
            if not (self.test_entryboxes_valid(self.sys_param_entryboxes_dict)):
                self.write(self.ICtab_status, "Error: Missing or invalid parameter: {}".format(self.missing_params[0]))
                return

            new_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["Initial"], title="Save IC text file", filetypes=[("Text files","*.txt")])
            
            if new_filename == "": return

            if new_filename.endswith(".txt"): new_filename = new_filename[:-4]
            self.write_init_file(new_filename + ".txt")

        except ValueError as uh_Oh:
            print(uh_Oh)
            
        return

    def write_init_file(self, newFileName, dir_name=""):
        self.update_IC_plot()

        try:
            with open(newFileName, "w+") as ofstream:
                print(dir_name + newFileName + " opened successfully")

                # We don't really need to note down the time of creation, but it could be useful for interaction with other programs.
                ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
                ofstream.write("$ System Parameters:\n")
                for key in self.sys_param_entryboxes_dict:
                    ofstream.write(key + ": " + str(self.sys_param_entryboxes_dict[key].get()) + "\n")

                ofstream.write("$ System Flags:\n")
                for key in self.sys_flag_dict:
                    ofstream.write(key + ": " + str(self.sys_flag_dict[key].tk_var.get()) + "\n")

                ofstream.write("$ Initial Conditions: (Nodes) x, N, P\n")
                for i in range(self.init_x.__len__()):
                    ofstream.write("{:.8e}\t{:.8e}\t{:.8e}\n".format(self.init_x[i], self.init_N[i], self.init_P[i]))

                ofstream.write("\n$ Initial Conditions: (Edges) x, E-field, Eg, Chi\n")
                for i in range(self.init_x_edges.__len__()):
                    ofstream.write("{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\n".format(self.init_x_edges[i], 0, self.init_Ec[i], self.init_Chi[i]))

        except OSError as oops:
            self.write(self.ICtab_status, "IC file not created")
            return

        self.write(self.ICtab_status, "IC file generated")
        return

    # Wrapper for load_ICfile with user selection from IC tab
    def select_init_file(self):
        self.IC_file_name = tk.filedialog.askopenfilename(initialdir = self.default_dirs["Initial"], title="Select IC text files", filetypes=[("Text files","*.txt")])
        if self.IC_file_name == "": return # If user closes dialog box without selecting a file

        self.load_ICfile()
        return

    def load_ICfile(self):
        warning_flag = False

        try:
            print("Poked file: {}".format(self.IC_file_name))
            with open(self.IC_file_name, 'r') as ifstream:
                init_param_values_dict = {"Mu_N":0, "Mu_P":0, "N0":0, "P0":0, 
                                          "Thickness":0, "B":0, "Tau_N":0, "Tau_P":0,
                                          "Sf":0, "Sb":0, "Temperature":0, "Rel-Permitivity":0, "Ext_E-Field":0,
                                          "Theta":0, "Alpha":0, "Delta":0, "Frac-Emitted":0, "dx":0}

                flag_values_dict = {"ignore_alpha":0}
                node_init_list = []
                edge_init_list = []
                initFlag = 0
                file_is_valid = False
                # Extract parameters, ICs
                for line in ifstream:
                    # Attempt to determine if loaded file is valid initial condition.
                    if not(file_is_valid) and ("$$ INITIAL CONDITION FILE CREATED ON" in line):
                        file_is_valid = True
                        continue

                    if ("#" in line) or (line.strip('\n').__len__() == 0):
                        continue

                    elif "$ System Parameters" in line:
                        continue

                    # There are four "$" flags in an IC file: "System Parameters", "System Flags", "Node Initial Conditions", and "Edge Initial Conditions"
                    # each corresponding to a different part of the initial state of the system
                    elif "$ Initial Conditions:" in line or "$ System Flags:" in line:
                        initFlag += 1

                    elif (initFlag == 0):
                        line = line.strip('\n')
                        init_param_values_dict[line[0:line.find(':')]] = (line[line.find(' ') + 1:])

                    elif (initFlag == 1):
                        line = line.strip('\n')
                        flag_values_dict[line[0:line.find(':')]] = (line[line.find(' ') + 1:])

                    elif (initFlag == 2):
                        if ('j' in line): raise OSError("Error: unable to read complex value in {}".format(self.IC_file_name)) # If complex arg detected
                        node_init_list.append(line.strip('\n'))

                    elif (initFlag == 3):
                        if ('j' in line): raise OSError("Error: unable to read complex value in {}".format(self.IC_file_name)) # If complex arg detected
                        edge_init_list.append(line.strip('\n'))

                if not(file_is_valid):
                    raise OSError("Error: unable to read {}".format(self.IC_file_name))

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        # Clear values in any IC generation areas; this is done to minimize ambiguity between IC's that came from the recently loaded file and ICs that were generated using the Initial tab
        for key in self.analytical_entryboxes_dict:
            self.enter(self.analytical_entryboxes_dict[key], "")

        self.reset_IC(force=True)

        # Enter saved params into boxes
        for key in init_param_values_dict:
            self.enter(self.sys_param_entryboxes_dict[key], init_param_values_dict[key])

        for key in flag_values_dict:
            self.sys_flag_dict[key].value = int(flag_values_dict[key])
            if (self.sys_flag_dict[key].value): self.sys_flag_dict[key].tk_var.set(1)
            else: self.sys_flag_dict[key].tk_var.set(0)

        try:
            self.set_init_x()
            
        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        self.init_N = np.zeros(int(0.5 + self.thickness / self.dx))
        self.init_P = np.zeros(int(0.5 + self.thickness / self.dx))
        self.init_E_field = np.zeros(int(0.5 + self.thickness / self.dx) + 1)
        self.init_Ec = np.zeros(int(0.5 + self.thickness / self.dx) + 1)
        self.init_Chi = np.zeros(int(0.5 + self.thickness / self.dx) + 1)

        # Though a rare case, the file reader is able to interpret out-of-order IC values and sort by ascending x coord
        node_init_list.sort(key = lambda x:float(x[0:x.find('\t')]))
        edge_init_list.sort(key = lambda x:float(x[0:x.find('\t')]))

        # Process the sorted node_init_list entries TWO at a time, filling all relevant IC arrays with linear-interpolated values.
        # We could use a much less complex one-value-at-a-time reader, but those methods usually lose out on linear interpolation ability
        try:
            for i in range(0, node_init_list.__len__() - 1):
                try:
                    first_valueset = extract_values(node_init_list[i], '\t') #[x1, N(x1), P(x1)]
                    second_valueset = extract_values(node_init_list[i+1], '\t') #[x2, N(x2), P(x2)]
                except ValueError:
                    self.write(self.ICtab_status, "Warning: Unusual IC File content")
                    warning_flag = True

                intermediate_x_indices = np.arange(finite.toIndex(first_valueset[0], self.dx, self.thickness, is_edge=False), finite.toIndex(second_valueset[0], self.dx, self.thickness, is_edge=False) + 1, 1)

                for j in intermediate_x_indices: # y-y0 = (y1-y0)/(x1-x0) * (x-x0)
                    try:
                        self.init_N[j] = first_valueset[1] + (finite.toCoord(j, self.dx) - first_valueset[0]) * (second_valueset[1] - first_valueset[1]) / (second_valueset[0] - first_valueset[0])
                    except:
                        self.init_N[j] = 0
                        
                    try:
                        self.init_P[j] = first_valueset[2] + (finite.toCoord(j, self.dx) - first_valueset[0]) * (second_valueset[2] - first_valueset[2]) / (second_valueset[0] - first_valueset[0])
                    except:
                        self.init_P[j] = 0

        
             #END LOOP
        #END LOOP
            
        # Now do the same for edge_init_list
            for i in range(0, edge_init_list.__len__() - 1):
                try:
                    first_valueset = extract_values(edge_init_list[i], '\t') #[x1, E(x1), Eg(x1), Chi(x1)]
                    second_valueset = extract_values(edge_init_list[i+1], '\t') #[x2, E(x2), Eg(x2), Chi(x2)]
                except ValueError:
                    self.write(self.ICtab_status, "Warning: Unusual IC File content")
                    warning_flag = True

                intermediate_x_indices = np.arange(finite.toIndex(first_valueset[0], self.dx, self.thickness, is_edge=True), finite.toIndex(second_valueset[0], self.dx, self.thickness, is_edge=True) + 1, 1)

                for j in intermediate_x_indices: # y-y0 = (y1-y0)/(x1-x0) * (x-x0)
                    try:
                        self.init_E_field[j] = first_valueset[1] + (finite.toCoord(j, self.dx, True) - first_valueset[0]) * (second_valueset[1] - first_valueset[1]) / (second_valueset[0] - first_valueset[0])
                    except:
                        self.init_E_field[j] = 0

                    try:
                        self.init_Ec[j] = first_valueset[2] + (finite.toCoord(j, self.dx, True) - first_valueset[0]) * (second_valueset[2] - first_valueset[2]) / (second_valueset[0] - first_valueset[0])
                    except:
                        self.init_Ec[j] = 0
                        
                    try:
                        self.init_Chi[j] = first_valueset[3] + (finite.toCoord(j, self.dx, True) - first_valueset[0]) * (second_valueset[3] - first_valueset[3]) / (second_valueset[0] - first_valueset[0])
                    except:
                        self.init_Chi[j] = 0

        except IndexError:
            self.write(self.ICtab_status, "Warning: could not fit IC distribution into system with thickness {}".format(self.thickness))
            warning_flag = True

        self.update_IC_plot(warn=warning_flag)

        if not warning_flag: self.write(self.ICtab_status, "IC file loaded successfully")
        return

    # Data I/O

    def export_plot(self, plot_ID):

        if plot_ID == -1:
            if self.I_plot.size() == 0: return
            if self.I_plot.mode == "Current time step": 
                paired_data = [[self.I_plot.I_sets[key].grid_x, self.I_plot.I_sets[key].I_data] for key in self.I_plot.I_sets]

                # TODO: Write both of these values with their units
                header = "{}, {}".format(self.I_plot.x_param, self.I_plot.type)

            else: # if self.I_plot.mode == "All time steps"
                raw_data = np.array([self.I_plot.I_sets[key].I_data for key in self.I_plot.I_sets])
                grid_x = np.reshape(self.I_plot.global_gridx, (1,self.I_plot.global_gridx.__len__()))
                paired_data = np.concatenate((grid_x, raw_data), axis=0).T
                header = "Time [ns],"
                for key in self.I_plot.I_sets:
                    header += self.I_plot.I_sets[key].tag().replace("Δ", "") + ","

        else:
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
        if self.I_plot.size() == 0: return
            
        if (self.I_plot.mode == "All time steps"):
            if self.bay_mode.get() == "obs":
                for key in self.I_plot.I_sets:  # For each curve on the integration plot
                    raw_data = self.I_plot.I_sets[key].I_data
                    grid_x = self.I_plot.global_gridx   # grid_x refers to what is on the x-axis, which in this case is technically 'time'
                    unc = raw_data * 0.1
                    full_data = np.vstack((grid_x, raw_data, unc)).T
                    full_data = pd.DataFrame.from_records(data=full_data,columns=['time', self.I_plot.type, 'uncertainty'])
                    
                    #FIXME: dd.save has no visible file overwrite handler
                    # If the file name already exists, dd.save will simply not save anything
                    dd.save("{}//{}.h5".format(self.default_dirs["PL"], self.I_plot.I_sets[key].tag()), full_data)

            elif self.bay_mode.get() == "model":
                active_bay_params = []
                for key in self.bay_params:
                    if self.bay_params[key].get(): active_bay_params.append(key)

                is_first = True
                for key in self.I_plot.I_sets:
                    raw_data = self.I_plot.I_sets[key].I_data
                    grid_x = self.I_plot.global_gridx
                    paired_data = np.vstack((grid_x, raw_data))

                    for param in active_bay_params:
                        param_column = np.ones((1,raw_data.__len__())) * self.I_plot.I_sets[key].params_dict[param] * self.convert_out_dict[param]
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
                panda_columns.append(self.I_plot.type)

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

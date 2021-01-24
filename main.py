#################################################
# Transient Electron Dynamics Simulator
# Model photoluminescent behavior in one-dimensional nanowire
# Last modified: Nov 25, 2020
# Author: Calvin Fai, Charles Hages
# Contact:
################################################# 

import numpy as np
import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure

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
import modules
from utils import extract_values, u_read, check_valid_filename, autoscale

import pandas as pd # For bayesim compatibility
import bayesim.hdf5io as dd

np.seterr(divide='raise', over='warn', under='warn', invalid='raise')
class Param_Rule:
    # The Parameter Toolkit uses these to build Parameter()'s values
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

class Flag:
    # This class exists to solve a little problem involving tkinter checkbuttons: we get the value of a checkbutton using its tk.IntVar() 
    # but we interact with the checkbutton using the actual tk.CheckButton() element
    # So wrap both of those together in a single object and call it a day
    def __init__(self, master, display_name):
        self.tk_var = tk.IntVar()
        self.tk_element = tk.ttk.Checkbutton(master=master, text=display_name, variable=self.tk_var, onvalue=1, offvalue=0)
        return
    
    def value(self):
        return self.tk_var.get()

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
    def __init__(self, data, grid_x, node_x, params_dict, type, filename, show_index):
        super().__init__(data, grid_x, params_dict, type, filename)
        self.node_x = node_x        # Array of x-coordinates corresponding to system nodes - needed to generate initial condition from data

        # node_x and grid_x will usually be identical, unless the data is a type (like E-field) that exists on edges
        # There's a little optimization that can be made here because grid_x will either be identical to node_x or not, but that makes the code harder to follow

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

    def build(self, convert_out_dict):
        result = []
        for key in self.datasets:
            result.append(self.datasets[key].grid_x)
            result.append(self.datasets[key].data * convert_out_dict[self.type])
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

        
class Notebook:
	# This is somewhat Java-like: everything about the GUI exists inside a class
    # A goal is to achieve total separation between this class (i.e. the GUI) and all mathematical operations, which makes this GUI reusable for different problems

    def __init__(self, title):
        self.nanowire = modules.Nanowire()
        #self.nanowire = modules.HeatPlate()
        self.nanowire.verify()
        
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

        # Tkinter checkboxes and radiobuttons require special variables to extract user input
        # IntVars or BooleanVars are sufficient for binary choices e.g. whether a checkbox is checked
        # while StringVars are more suitable for open-ended choices e.g. selecting one mode from a list
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
        self.paramtoolkit_viewer_selection = tk.StringVar()
        self.listupload_var_selection = tk.StringVar()
        self.display_selection = tk.StringVar()

        # Flags and containers for IC arrays
        
        self.convert_in_dict = self.nanowire.convert_in_dict
        self.convert_out_dict = self.nanowire.convert_out_dict

        self.active_paramrule_list = []
        self.paramtoolkit_currentparam = ""
        self.IC_file_list = None
        self.IC_file_name = ""
        
        # FIXME: This is a somewhat clumsy way to introduce a Nanowire module-specific functionality
        if self.nanowire.system_ID == "Nanowire":
            self.using_AIC = False

        self.carry_include_flags = {}
        for var in self.nanowire.simulation_outputs_dict:
            self.carry_include_flags[var] = tk.IntVar()
            
        self.check_bay_params = {}
        for param in self.nanowire.param_dict:
            self.check_bay_params[param] = tk.IntVar()
            
        self.bay_mode = tk.StringVar(value="model")

        # Helpers, flags, and containers for analysis plots
        self.analysis_plots = [Analysis_Plot_State(), Analysis_Plot_State(), Analysis_Plot_State(), Analysis_Plot_State()]
        self.integration_plots = [Integration_Plot_State()]
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
        self.sys_printsummary_popup_isopen = False
        self.sys_plotsummary_popup_isopen = False
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
        #self.tab_inputs.bind("<<NotebookTabChanged>>", self.on_input_subtab_selected)

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
            
        print("Checking whether the current system class ({}) has a dedicated data subdirectory...".format(self.nanowire.system_ID))
        try:
            os.mkdir(self.default_dirs["Data"] + "\\" + self.nanowire.system_ID)
            print("No such subdirectory detected; automatically creating...")
        except FileExistsError:
            print("Subdirectory detected")

        return

    def run(self):
        self.notebook.pack(expand=1, fill="both")
        width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()

        self.root.geometry('%dx%d+0+0' % (width,height))
        self.root.mainloop()
        print("Closed TEDs")
        matplotlib.use(starting_backend)
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
        paramtoolkit_method_dropdown_list = ["POINT", "FILL", "LINE", "EXP"]
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

        self.steps_head = tk.ttk.Label(self.spacegrid_frame, text="Space Grid - Start Here", style="Header.TLabel")
        self.steps_head.grid(row=0,column=0,columnspan=2)

        self.thickness_label = tk.ttk.Label(self.spacegrid_frame, text="Thickness " + self.nanowire.length_unit)
        self.thickness_label.grid(row=1,column=0)

        self.thickness_entry = tk.ttk.Entry(self.spacegrid_frame, width=9)
        self.thickness_entry.grid(row=1,column=1)

        self.dx_label = tk.ttk.Label(self.spacegrid_frame, text="Node width " + self.nanowire.length_unit)
        self.dx_label.grid(row=2,column=0)

        self.dx_entry = tk.ttk.Entry(self.spacegrid_frame, width=9)
        self.dx_entry.grid(row=2,column=1)

        self.params_frame = tk.ttk.Frame(self.tab_inputs)
        self.params_frame.grid(row=2,column=0,columnspan=2, rowspan=4)

        self.system_params_head = tk.ttk.Label(self.params_frame, text="Constant-value Parameters",style="Header.TLabel")
        self.system_params_head.grid(row=0, column=0,columnspan=2)
        
        self.system_params_shortcut_button = tk.ttk.Button(self.params_frame, text="Fast Param Entry Tool", command=self.do_sys_param_shortcut_popup)
        self.system_params_shortcut_button.grid(row=1,column=0,columnspan=2)

        self.flags_frame = tk.ttk.Frame(self.tab_inputs)
        self.flags_frame.grid(row=6,column=0,columnspan=2)

        self.flags_head = tk.ttk.Label(self.flags_frame, text="Flags", style="Header.TLabel")
        self.flags_head.grid(row=0,column=0,columnspan=2)
        
        # Procedurally generated elements for flags
        i = 1
        self.sys_flag_dict = {}
        for flag in self.nanowire.flags_dict:
            self.sys_flag_dict[flag] = Flag(self.flags_frame, self.nanowire.flags_dict[flag])
            self.sys_flag_dict[flag].tk_element.grid(row=i,column=0)
            i += 1
            
        self.ICtab_status = tk.Text(self.tab_inputs, width=20,height=8)
        self.ICtab_status.grid(row=7, column=0, columnspan=2)
        self.ICtab_status.configure(state='disabled')
        
        self.system_printout_button = tk.ttk.Button(self.tab_inputs, text="Print Init. State Summary", command=self.do_sys_printsummary_popup)
        self.system_printout_button.grid(row=8,column=0,columnspan=2)
        
        self.system_plotout_button = tk.ttk.Button(self.tab_inputs, text="Show Init. State Plots", command=self.do_sys_plotsummary_popup)
        self.system_plotout_button.grid(row=9,column=0,columnspan=2)

        self.line1_separator = tk.ttk.Separator(self.tab_inputs, orient="vertical", style="Grey Bar.TSeparator")
        self.line1_separator.grid(row=0,rowspan=30,column=2,pady=(24,0),sticky="ns")
             
        ## Parameter Toolkit:

        self.param_rules_frame = tk.ttk.Frame(self.tab_rules_init)
        self.param_rules_frame.grid(row=0,column=0,padx=(370,0))

        self.active_paramrule_list_title = tk.ttk.Label(self.param_rules_frame, text="Add/Edit/Remove Space-Dependent Parameters", style="Header.TLabel")
        self.active_paramrule_list_title.grid(row=0,column=0,columnspan=3)

        self.active_paramrule_listbox = tk.Listbox(self.param_rules_frame, width=86,height=8)
        self.active_paramrule_listbox.grid(row=1,rowspan=3,column=0,columnspan=3, padx=(32,32))

        self.paramrule_var_label = tk.ttk.Label(self.param_rules_frame, text="Select parameter to edit:")
        self.paramrule_var_label.grid(row=4,column=0)
        
        self.paramrule_var_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, self.init_var_selection, var_dropdown_list[0], *var_dropdown_list)
        self.paramrule_var_dropdown.grid(row=4,column=1)

        self.paramrule_method_label = tk.ttk.Label(self.param_rules_frame, text="Select calculation method:")
        self.paramrule_method_label.grid(row=5,column=0)

        self.paramrule_method_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, self.init_shape_selection, paramtoolkit_method_dropdown_list[0], *paramtoolkit_method_dropdown_list)
        self.paramrule_method_dropdown.grid(row=5, column=1)

        self.paramrule_lbound_label = tk.ttk.Label(self.param_rules_frame, text="Left bound coordinate:")
        self.paramrule_lbound_label.grid(row=6, column=0)

        self.paramrule_lbound_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_lbound_entry.grid(row=6,column=1)

        self.paramrule_rbound_label = tk.ttk.Label(self.param_rules_frame, text="Right bound coordinate:")
        self.paramrule_rbound_label.grid(row=7, column=0)

        self.paramrule_rbound_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_rbound_entry.grid(row=7,column=1)

        self.paramrule_lvalue_label = tk.ttk.Label(self.param_rules_frame, text="Left bound value:")
        self.paramrule_lvalue_label.grid(row=8, column=0)

        self.paramrule_lvalue_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_lvalue_entry.grid(row=8,column=1)

        self.paramrule_rvalue_label = tk.ttk.Label(self.param_rules_frame, text="Right bound value:")
        self.paramrule_rvalue_label.grid(row=9, column=0)

        self.paramrule_rvalue_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_rvalue_entry.grid(row=9,column=1)

        self.add_paramrule_button = tk.ttk.Button(self.param_rules_frame, text="Add new parameter rule", command=self.add_paramrule)
        self.add_paramrule_button.grid(row=10,column=0,columnspan=2)

        self.delete_paramrule_button = tk.ttk.Button(self.param_rules_frame, text="Delete highlighted rule", command=self.delete_paramrule)
        self.delete_paramrule_button.grid(row=4,column=2)

        self.deleteall_paramrule_button = tk.ttk.Button(self.param_rules_frame, text="Delete all rules for this parameter", command=self.deleteall_paramrule)
        self.deleteall_paramrule_button.grid(row=5,column=2)

        self.paramtoolkit_description = tk.Message(self.param_rules_frame, text="The Parameter Toolkit uses a series of rules and patterns to build a spatially dependent distribution for any parameter.", width=250)
        self.paramtoolkit_description.grid(row=6,rowspan=3,column=2,columnspan=2)

        self.paramtoolkit_description2 = tk.Message(self.param_rules_frame, text="Warning: Rules are applied from top to bottom. Order matters!", width=250)
        self.paramtoolkit_description2.grid(row=9,rowspan=3,column=2,columnspan=2)
        
        # These plots were previously attached to self.tab_inputs so that it was visible on all three IC tabs,
        # but it was hard to position them correctly.
        # Attaching to the Parameter Toolkit makes them easier to position
        self.custom_param_fig = Figure(figsize=(5,3.1))
        self.custom_param_subplot = self.custom_param_fig.add_subplot(111)
        # Prevent coordinate values from appearing in the toolbar; this would sometimes jostle GUI elements around
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

        self.moveup_paramrule_button = tk.ttk.Button(self.param_rules_frame, text="⇧", command=self.moveup_paramrule)
        self.moveup_paramrule_button.grid(row=1,column=4)

        self.paramrule_viewer_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, self.paramtoolkit_viewer_selection, unitless_dropdown_list[0], *unitless_dropdown_list)
        self.paramrule_viewer_dropdown.grid(row=2,column=4)

        self.paramrule_view_button = tk.ttk.Button(self.param_rules_frame, text="Change view", command=self.refresh_paramrule_listbox)
        self.paramrule_view_button.grid(row=2,column=5)

        self.movedown_paramrule_button = tk.ttk.Button(self.param_rules_frame, text="⇩", command=self.movedown_paramrule)
        self.movedown_paramrule_button.grid(row=3,column=4)

        ## Param List Upload:

        self.listupload_frame = tk.ttk.Frame(self.tab_explicit_init)
        self.listupload_frame.grid(row=0,column=0,padx=(440,0))

        self.listupload_description = tk.Message(self.listupload_frame, text="This tab provides an option to directly import a list of data points, on which the TED will do linear interpolation to fit to the specified space grid.", width=360)
        self.listupload_description.grid(row=0,column=0)
        
        self.listupload_dropdown = tk.ttk.OptionMenu(self.listupload_frame, self.listupload_var_selection, unitless_dropdown_list[0], *unitless_dropdown_list)
        self.listupload_dropdown.grid(row=1,column=0)

        self.add_listupload_button = tk.ttk.Button(self.listupload_frame, text="Import", command=self.add_listupload)
        self.add_listupload_button.grid(row=2,column=0)
        
        self.listupload_fig = Figure(figsize=(6,3.8))
        self.listupload_subplot = self.listupload_fig.add_subplot(111)
        self.listupload_canvas = tkagg.FigureCanvasTkAgg(self.listupload_fig, master=self.listupload_frame)
        self.listupload_plotwidget = self.listupload_canvas.get_tk_widget()
        self.listupload_plotwidget.grid(row=0, rowspan=3,column=1)
        
        self.listupload_toolbar_frame = tk.ttk.Frame(master=self.listupload_frame)
        self.listupload_toolbar_frame.grid(row=3,column=1)
        self.listupload_toolbar = tkagg.NavigationToolbar2Tk(self.listupload_canvas, self.listupload_toolbar_frame)


        # Attach sub-frames to input tab and input tab to overall notebook
        self.tab_inputs.add(self.tab_rules_init, text="Parameter Toolkit")
        
        ## Analytical Initial Condition (AIC): extra input mtd for specific system type
        if self.nanowire.system_ID == "Nanowire":
            self.create_AIC_frame()
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

        self.do_ss_checkbutton = tk.ttk.Checkbutton(self.tab_simulate, text="Inject init. conds. as generation?", variable=self.check_do_ss, onvalue=1, offvalue=0)
        self.do_ss_checkbutton.grid(row=5,column=0)

        self.calculate_NP = tk.ttk.Button(self.tab_simulate, text="Start Simulation(s)", command=self.do_Batch)
        self.calculate_NP.grid(row=6,column=0,columnspan=2,padx=(9,12))

        self.status_label = tk.ttk.Label(self.tab_simulate, text="Status")
        self.status_label.grid(row=7, column=0, columnspan=2)

        self.status = tk.Text(self.tab_simulate, width=28,height=4)
        self.status.grid(row=8, rowspan=2, column=0, columnspan=2)
        self.status.configure(state='disabled')

        self.line3_separator = tk.ttk.Separator(self.tab_simulate, orient="vertical", style="Grey Bar.TSeparator")
        self.line3_separator.grid(row=0,rowspan=30,column=2,sticky="ns")

        self.subtitle = tk.ttk.Label(self.tab_simulate, text="Simulation - {}".format(self.nanowire.system_ID))
        self.subtitle.grid(row=0,column=3,columnspan=3)
        
        self.sim_fig = Figure(figsize=(12, 8))
        count = 1
        cdim = np.ceil(np.sqrt(self.nanowire.simulation_outputs_count))
        
        rdim = np.ceil(self.nanowire.simulation_outputs_count / cdim)
        self.sim_subplots = {}
        for variable in self.nanowire.simulation_outputs_dict:
            self.sim_subplots[variable] = self.sim_fig.add_subplot(rdim, cdim, count)
            self.sim_subplots[variable].set_title(variable)
            count += 1

        self.sim_canvas = tkagg.FigureCanvasTkAgg(self.sim_fig, master=self.tab_simulate)
        self.sim_plot_widget = self.sim_canvas.get_tk_widget()
        self.sim_plot_widget.grid(row=1,column=3,rowspan=12,columnspan=2)
        
        self.simfig_toolbar_frame = tk.ttk.Frame(master=self.tab_simulate)
        self.simfig_toolbar_frame.grid(row=13,column=3,columnspan=2)
        self.simfig_toolbar = tkagg.NavigationToolbar2Tk(self.sim_canvas, self.simfig_toolbar_frame)

        self.notebook.add(self.tab_simulate, text="Simulate")
        return

    def add_tab_analyze(self):
        self.tab_analyze = tk.ttk.Notebook(self.notebook)
        self.tab_overview_analysis = tk.ttk.Frame(self.tab_analyze)
        self.tab_detailed_analysis = tk.ttk.Frame(self.tab_analyze)
        
        self.analyze_overview_fig = Figure(figsize=(15,8))
        self.overview_subplots = {}
        count = 1
        rdim = np.floor(np.sqrt(self.nanowire.total_outputs_count))
        cdim = np.ceil(self.nanowire.total_outputs_count / rdim)
        for output in self.nanowire.simulation_outputs_dict:
            self.overview_subplots[output] = self.analyze_overview_fig.add_subplot(rdim, cdim, count)
            count += 1
            
        for output in self.nanowire.calculated_outputs_dict:
            self.overview_subplots[output] = self.analyze_overview_fig.add_subplot(rdim, cdim, count)
            count += 1
        
        self.analyze_overview_button = tk.ttk.Button(master=self.tab_overview_analysis, text="Select Dataset", command=self.plot_overview_analysis)
        self.analyze_overview_button.grid(row=0,column=0)
        
        self.analyze_overview_canvas = tkagg.FigureCanvasTkAgg(self.analyze_overview_fig, master=self.tab_overview_analysis)
        self.analyze_overview_widget = self.analyze_overview_canvas.get_tk_widget()
        self.analyze_overview_widget.grid(row=1,column=0)

        self.overview_toolbar_frame = tk.ttk.Frame(self.tab_overview_analysis)
        self.overview_toolbar_frame.grid(row=2,column=0)
        self.overview_toolbar = tkagg.NavigationToolbar2Tk(self.analyze_overview_canvas, self.overview_toolbar_frame)
        self.overview_toolbar.grid(row=0,column=0)
        
        self.analysis_title = tk.ttk.Label(self.tab_detailed_analysis, text="Plot and Integrate Saved Datasets", style="Header.TLabel")
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
        
        self.analyze_canvas = tkagg.FigureCanvasTkAgg(self.analyze_fig, master=self.tab_detailed_analysis)
        self.analyze_widget = self.analyze_canvas.get_tk_widget()
        self.analyze_widget.grid(row=1,column=0,rowspan=1,columnspan=4, padx=(12,0))

        self.analyze_plotselector_frame = tk.ttk.Frame(master=self.tab_detailed_analysis)
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
        
        self.analyze_toolbar_frame = tk.ttk.Frame(master=self.tab_detailed_analysis)
        self.analyze_toolbar_frame.grid(row=4,column=0,rowspan=4,columnspan=4)
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
        self.integration_subplot = self.integration_fig.add_subplot(1,1,1)
        self.integration_plots[0].plot_obj = self.integration_subplot

        self.integration_canvas = tkagg.FigureCanvasTkAgg(self.integration_fig, master=self.tab_detailed_analysis)
        self.integration_widget = self.integration_canvas.get_tk_widget()
        self.integration_widget.grid(row=1,column=5,rowspan=1,columnspan=1, padx=(20,0))

        self.integration_plotselector_frame = tk.ttk.Frame(master=self.tab_detailed_analysis)
        self.integration_plotselector_frame.grid(row=2,column=5)
        
        self.integrationplot_left = tk.ttk.Radiobutton(self.integration_plotselector_frame, variable=self.active_integrationplot_ID, value=0)
        self.integrationplot_left.grid(row=0,column=0)

        self.integrationplot_topleft_label = tk.ttk.Label(self.integration_plotselector_frame, text="Use: Integration")
        self.integrationplot_topleft_label.grid(row=0,column=1)
        
        self.integrationplot_topright = tk.ttk.Radiobutton(self.integration_plotselector_frame, variable=self.active_integrationplot_ID, value=1)
        self.integrationplot_topright.grid(row=0,column=2)

        self.integrationplot_topright_label = tk.ttk.Label(self.integration_plotselector_frame, text="Use: Tau_Diff")
        self.integrationplot_topright_label.grid(row=0,column=3)

        self.integration_toolbar_frame = tk.ttk.Frame(master=self.tab_detailed_analysis)
        self.integration_toolbar_frame.grid(row=3,column=5, rowspan=2,columnspan=1)
        self.integration_toolbar = tkagg.NavigationToolbar2Tk(self.integration_canvas, self.integration_toolbar_frame)
        self.integration_toolbar.grid(row=0,column=0,columnspan=5)

        self.integration_axis_button = tk.ttk.Button(self.integration_toolbar_frame, text="Axis Settings", command=partial(self.do_change_axis_popup, from_integration=1))
        self.integration_axis_button.grid(row=1,column=0)

        self.integration_export_button = tk.ttk.Button(self.integration_toolbar_frame, text="Export", command=partial(self.export_plot, from_integration=1))
        self.integration_export_button.grid(row=1,column=1)

        self.integration_bayesim_button = tk.ttk.Button(self.integration_toolbar_frame, text="Bayesim", command=partial(self.do_bayesim_popup))
        self.integration_bayesim_button.grid(row=1,column=2)

        self.analysis_status = tk.Text(self.tab_detailed_analysis, width=28,height=3)
        self.analysis_status.grid(row=5,rowspan=3,column=5,columnspan=1)
        self.analysis_status.configure(state="disabled")

        self.tab_analyze.add(self.tab_overview_analysis, text="Overview")
        self.tab_analyze.add(self.tab_detailed_analysis, text="Detailed Analysis")
        self.notebook.add(self.tab_analyze, text="Analyze")
        return
    
    def create_AIC_frame(self):
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
        
        self.analytical_entryboxes_dict = {"A0":self.A0_entry, "Eg":self.Eg_entry, "AIC_expfactor":self.AIC_expfactor_entry, "Pulse_Freq":self.pulse_freq_entry, 
                                           "Pulse_Wavelength":self.pulse_wavelength_entry, "Power":self.power_entry, "Spotsize":self.spotsize_entry, "Power_Density":self.power_density_entry,
                                           "Max_Gen":self.max_gen_entry, "Total_Gen":self.total_gen_entry}
        return

    def DEBUG(self):
        print(self.nanowire.DEBUG_print())
        return

    def update_system_summary(self):
        if self.sys_printsummary_popup_isopen:
            self.write(self.printsummary_textbox, self.nanowire.DEBUG_print())
            
        if self.sys_plotsummary_popup_isopen:
            for param_name in self.nanowire.param_dict:
                param = self.nanowire.param_dict[param_name]
                val = finite.toArray(param.value, len(self.nanowire.grid_x_nodes), param.is_edge)
                grid_x = self.nanowire.grid_x_nodes if not param.is_edge else self.nanowire.grid_x_edges
                self.sys_param_summaryplots[param_name].plot(grid_x, val)
                self.sys_param_summaryplots[param_name].set_yscale(autoscale(val))
                
            self.plotsummary_fig.tight_layout()
            self.plotsummary_fig.canvas.draw()

        return
    ## Functions to create popups and manage
    
    def do_sys_printsummary_popup(self):
        if not self.sys_printsummary_popup_isopen: # Don't open more than one of this window at a time
            self.sys_printsummary_popup = tk.Toplevel(self.root)
            
            self.printsummary_textbox = tkscrolledtext.ScrolledText(self.sys_printsummary_popup, width=100,height=30)
            self.printsummary_textbox.grid(row=0,column=0,padx=(20,0), pady=(20,20))
            
            self.sys_printsummary_popup_isopen = True
            
            self.update_system_summary()
            
            self.sys_printsummary_popup.protocol("WM_DELETE_WINDOW", self.on_sys_printsummary_popup_close)
            return
        
    def on_sys_printsummary_popup_close(self):
        try:
            self.sys_printsummary_popup.destroy()
            self.sys_printsummary_popup_isopen = False
        except:
            print("Error #2022: Failed to close shortcut popup.")
        return
    
    def do_sys_plotsummary_popup(self):
        if not self.nanowire.spacegrid_is_set: return
        
        if not self.sys_plotsummary_popup_isopen:
            self.sys_plotsummary_popup = tk.Toplevel(self.root)

            count = 1
            rdim = np.floor(np.sqrt(self.nanowire.param_count))
            #rdim = 4
            cdim = np.ceil(self.nanowire.param_count / rdim)
            
            if self.sys_flag_dict['symmetric_system'].value():
                self.plotsummary_symmetriclabel = tk.Label(self.sys_plotsummary_popup, text="Note: All distributions are symmetric about x=0")
                self.plotsummary_symmetriclabel.grid(row=0,column=0)

            self.plotsummary_fig = Figure(figsize=(20,10))
            self.sys_param_summaryplots = {}
            for param_name in self.nanowire.param_dict:
                
                self.sys_param_summaryplots[param_name] = self.plotsummary_fig.add_subplot(rdim, cdim, count)
                self.sys_param_summaryplots[param_name].set_title(param_name)
                count += 1
            
            self.plotsummary_canvas = tkagg.FigureCanvasTkAgg(self.plotsummary_fig, master=self.sys_plotsummary_popup)
            self.plotsummary_plotwidget = self.plotsummary_canvas.get_tk_widget()
            self.plotsummary_plotwidget.grid(row=1,column=0)
            
            self.sys_plotsummary_popup_isopen = True
            self.update_system_summary()
            
            self.sys_plotsummary_popup.protocol("WM_DELETE_WINDOW", self.on_sys_plotsummary_popup_close)
            ## Temporarily disable the main window while this popup is active
            self.sys_plotsummary_popup.grab_set()
            
            return
        
    def on_sys_plotsummary_popup_close(self):
        try:
            self.sys_plotsummary_popup.destroy()
            self.sys_plotsummary_popup_isopen = False
        except:
            print("Error #2023: Failed to close plotsummary popup.")
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
                
                if isinstance(self.nanowire.param_dict[param].value, (float, int)):
                    self.enter(self.sys_param_entryboxes_dict[param], str(self.nanowire.param_dict[param].value))
                
                else:
                    self.enter(self.sys_param_entryboxes_dict[param], "[list]")
                row_count += 1
                if row_count == max_per_col:
                    row_count = 0
                    col_count += 2
                    
            self.shortcut_continue_button = tk.Button(self.sys_param_shortcut_popup, text="Continue", command=partial(self.on_sys_param_shortcut_popup_close, True))
            self.shortcut_continue_button.grid(row=2,column=1)
                    
            self.sys_param_shortcut_popup.protocol("WM_DELETE_WINDOW", self.on_sys_param_shortcut_popup_close)
            self.sys_param_shortcut_popup_isopen = True
            ## Temporarily disable the main window while this popup is active
            self.sys_param_shortcut_popup.grab_set()
            
            return
                    
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
                    
                    self.paramtoolkit_currentparam = param
                    self.deleteall_paramrule()
                    self.nanowire.param_dict[param].value = val
                    changed_params.append(param)
                    
                if changed_params.__len__() > 0:
                    self.update_IC_plot(plot_ID="recent")
                    self.write(self.ICtab_status, "Updated: {}".format(changed_params))
                    
                else:
                    self.write(self.ICtab_status, "")
                    
            self.sys_param_shortcut_popup.destroy()
            self.sys_param_shortcut_popup_isopen = False
        except:
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
            batchable_params = [param for param in self.nanowire.param_dict if not (self.nanowire.system_ID == "Nanowire" and self.using_AIC and (param == "deltaN" or param == "deltaP"))]
            
            if self.nanowire.system_ID == "Nanowire" and self.using_AIC:
                
                self.AIC_instruction1 = tk.Message(self.batch_popup, text="Additional options for generating deltaN and deltaP batches " +
                                                  "are available when using the Analytical Initial Condition tool.", width=300)
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
                if self.nanowire.system_ID == "Nanowire" and self.using_AIC:
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

            self.var_select_menu = tk.OptionMenu(self.plotter_popup, self.data_var, *(output for output in self.nanowire.outputs_dict if self.nanowire.outputs_dict[output].analysis_plotable))
            self.var_select_menu.grid(row=1,column=0)

            self.autointegrate_checkbutton = tk.Checkbutton(self.plotter_popup, text="Auto integrate all space and time steps?", variable=self.check_autointegrate, onvalue=1, offvalue=0)
            self.autointegrate_checkbutton.grid(row=1,column=1)
            
            self.plotter_continue_button = tk.Button(self.plotter_popup, text="Continue", command=partial(self.on_plotter_popup_close, plot_ID, continue_=True))
            self.plotter_continue_button.grid(row=2,column=1)

            self.data_listbox = tk.Listbox(self.plotter_popup, width=20, height=20, selectmode="extended")
            self.data_listbox.grid(row=2,rowspan=13,column=0)
            self.data_listbox.delete(0,tk.END)
            self.data_list = [file for file in os.listdir(self.default_dirs["Data"] + "\\" + self.nanowire.system_ID) if not file.endswith(".txt")]
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

            self.integration_getbounds_title_label = tk.Label(self.integration_getbounds_popup, text="Enter bounds of integration " + self.nanowire.length_unit)
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

            self.integration_center_label = tk.Label(self.integration_getbounds_popup, text="Enter space-separated e.g. (100 200 300...) Centers {}: ".format(self.nanowire.length_unit))
            self.integration_center_label.grid(row=5,column=2)

            self.integration_center_entry = tk.Entry(self.integration_getbounds_popup, width=30)
            self.integration_center_entry.grid(row=5,column=3,columnspan=3)

            self.integration_width_label = tk.Label(self.integration_getbounds_popup, text="Width {}: +/- ".format(self.nanowire.length_unit))
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
                        
                    if (lbound < 0 and ubound < 0):
                        raise KeyError("Error: bounds out of range")

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
                        if center < 0:
                            raise KeyError("Error: center {} is out of range".format(center))
                        else:
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
            self.enter(self.ylbound, "{:.2e}".format(active_plot.ylim[0]))
            self.enter(self.yubound, "{:.2e}".format(active_plot.ylim[1]))
            self.xaxis_type.set(active_plot.xaxis_type)
            self.yaxis_type.set(active_plot.yaxis_type)
            self.check_display_legend.set(active_plot.display_legend)

            self.change_axis_popup.protocol("WM_DELETE_WINDOW", partial(self.on_change_axis_popup_close, from_integration, continue_=False))
            self.change_axis_popup.grab_set()
            self.change_axis_popup_isopen = True
        else:
            print("Error #440: Opened more than one change axis popup at a time")
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

            self.change_axis_popup_isopen = False

        except ValueError as oops:
            self.write(self.change_axis_status, oops)
            return
        except:
            print("Error #441: Failed to close change axis popup.")

        return

    def do_IC_carry_popup(self):
        plot_ID = self.active_analysisplot_ID.get()
        # Don't open if no data plotted
        if self.analysis_plots[plot_ID].datagroup.size() == 0: return

        if not self.IC_carry_popup_isopen:
            self.IC_carry_popup = tk.Toplevel(self.root)

            self.IC_carry_title_label = tk.ttk.Label(self.IC_carry_popup, text="Select data to include in new IC", style="Header.TLabel")
            self.IC_carry_title_label.grid(row=0,column=0,columnspan=2)
            
            self.carry_checkbuttons = {}
            rcount = 1
            for var in self.nanowire.simulation_outputs_dict:
                self.carry_checkbuttons[var] = tk.Checkbutton(self.IC_carry_popup, text=var, variable=self.carry_include_flags[var])
                self.carry_checkbuttons[var].grid(row=rcount, column=0)
                rcount += 1

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
            print("Error #510: Opened more than one IC carryover popup at a time")
        return

    def on_IC_carry_popup_close(self, continue_=False):
        try:
            if continue_:
                plot_ID = self.active_analysisplot_ID.get()
                active_sets = self.analysis_plots[plot_ID].datagroup.datasets
                datasets = [self.carry_IC_listbox.get(i) for i in self.carry_IC_listbox.curselection()]
                
                include_flags = {}
                for iflag in self.carry_include_flags:
                    include_flags[iflag] = self.carry_include_flags[iflag].get()
                    
                for key in datasets:
                    new_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["Initial"], title="Save IC text file for {}".format(key), filetypes=[("Text files","*.txt")])
                    if new_filename == "": continue

                    if new_filename.endswith(".txt"): new_filename = new_filename[:-4]
                    
                    param_dict_copy = dict(active_sets[key].params_dict)

                    node_x = active_sets[key].node_x
                    
                    filename = active_sets[key].filename
                    sim_data = {}
                    for var in self.nanowire.simulation_outputs_dict:
                        path_name = "{}\\{}\\{}\\{}-{}.h5".format(self.default_dirs["Data"], self.nanowire.system_ID, filename, filename, var)
                        sim_data[var] = u_read(path_name, t0=active_sets[key].show_index, single_tstep=True)

                    self.nanowire.get_IC_carry(sim_data, param_dict_copy, include_flags, node_x)

                    with open(new_filename + ".txt", "w+") as ofstream:
                        ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
                        ofstream.write("System_class: {}\n".format(self.nanowire.system_ID))
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
                                ofstream.write("{}: {}\n".format(param, int(param_dict_copy[param])))

                self.write(self.analysis_status, "IC file generated")

            self.IC_carry_popup.destroy()

            self.IC_carry_popup_isopen = False

        except OSError:
            self.write(self.analysis_status, "Error: failed to regenerate IC file")
            
        except:
            print("Error #511: Failed to close IC carry popup.")

        return

    def do_bayesim_popup(self, plot_ID=0):
        if self.integration_plots[plot_ID].datagroup.size() == 0: return

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
            self.bayesim_popup_isopen = False

        except FloatingPointError:
            print("Error #601: Failed to close Bayesim popup")
        return

    ## Plotter for simulation tab    
    def update_sim_plots(self, index, do_clear_plots=True):
        ## V2: Update plots on Simulate tab
        
        for variable, output_obj in self.nanowire.simulation_outputs_dict.items():
            
            plot = self.sim_subplots[variable]
            
            if do_clear_plots: 
                plot.cla()

                ymin = np.amin(self.sim_data[variable]) * output_obj.yfactors[0]
                ymax = np.amax(self.sim_data[variable]) * output_obj.yfactors[1]
                plot.set_ylim(ymin * self.convert_out_dict[variable], ymax * self.convert_out_dict[variable])

            plot.set_yscale(output_obj.yscale)
            
            grid_x = self.nanowire.grid_x_nodes if not output_obj.is_edge else self.nanowire.grid_x_edges
            plot.plot(grid_x, self.sim_data[variable] * self.convert_out_dict[variable])

            plot.set_xlabel("x {}".format(self.nanowire.length_unit))
            plot.set_ylabel("{} {}".format(variable, output_obj.units))

            plot.set_title("Time: {} ns".format(self.simtime * index / self.n))
            
        self.sim_fig.tight_layout()
        self.sim_fig.canvas.draw()
        return
    
    ## Func for overview analyze tab
    def fetch_metadata(self, data_filename):
        with open(self.default_dirs["Data"] + "\\" + self.nanowire.system_ID + "\\" + data_filename + "\\" + "metadata.txt", "r") as ifstream:
            param_values_dict = {}
            for line in ifstream:
                if "$" in line: continue

                elif "#" in line: continue
            
                elif "System_class" in line:
                    system_class = line[line.find(' ') + 1:].strip('\n')
                    if not (self.nanowire.system_ID == system_class):
                        raise ValueError

                else:
                    param = line[0:line.find(':')]
                    new_value = line[line.find(' ') + 1:].strip('\n')

                    if '\t' in new_value:
                        param_values_dict[param] = np.array(extract_values(new_value, '\t'))
                    else: param_values_dict[param] = float(new_value)
                    
        # Convert from cm, V, s to nm, V, ns
        for param in param_values_dict:
            if param in self.convert_in_dict:
                param_values_dict[param] *= self.convert_in_dict[param]
                    
        return param_values_dict
    
    def plot_overview_analysis(self):
        data_dirname = tk.filedialog.askdirectory(title="Select a dataset", initialdir=self.default_dirs["Data"])
        if not data_dirname:
            print("No data set selected :(")
            return

        data_filename = data_dirname[data_dirname.rfind('/')+1:]
        
        try:
            param_values_dict = self.fetch_metadata(data_filename)
                 
            data_n = int(0.5 + param_values_dict["Total-Time"] / param_values_dict["dt"])
            data_m = int(0.5 + param_values_dict["Total_length"] / param_values_dict["Node_width"])
            data_edge_x = np.linspace(0, param_values_dict["Total_length"],data_m+1)
            data_node_x = np.linspace(param_values_dict["Node_width"] / 2, param_values_dict["Total_length"] - param_values_dict["Node_width"] / 2, data_m)
            data_node_t = np.linspace(0, param_values_dict["Total-Time"], data_n + 1)
            tstep_list = np.append([0], np.geomspace(1, data_n, num=5, dtype=int))
        except:
            self.write(self.analysis_status, "Error: {} is missing or has unusual metadata.txt".format(data_filename))
            return
        
        for subplot in self.overview_subplots:
            plot_obj = self.overview_subplots[subplot]
            output_info_obj = self.nanowire.outputs_dict[subplot]
            plot_obj.cla()
            plot_obj.set_yscale(output_info_obj.yscale)
            plot_obj.set_xlabel(output_info_obj.xlabel)
            plot_obj.set_title("{} {}".format(output_info_obj.display_name, output_info_obj.units))
            
            
        data_dict = self.nanowire.get_overview_analysis(param_values_dict, tstep_list, data_dirname, data_filename)
        
        for output_name, output_info in self.nanowire.outputs_dict.items():
            try:
                values = data_dict[output_name]
                
                if not isinstance(values, np.ndarray): raise KeyError
            except KeyError:
                print("Warning: {}'s get_overview_analysis() did not return data for {}".format(self.nanowire.system_ID, output_name))
                continue

            if output_info.xvar == "time":
                grid_x = data_node_t
                
            elif output_info.xvar == "position":
                grid_x = data_node_x if not output_info.is_edge else data_edge_x
                
            else:
                print("Warning: invalid xvar {} in system class definition for output {}".format(output_info.xvar, output_name))
                continue
            
            if values.ndim == 2: # time/space variant outputs
                for i in range(len(values)):
                    self.overview_subplots[output_name].plot(grid_x, values[i], label="{:.3f} ns".format(tstep_list[i] * param_values_dict["dt"]))
                    
            else: # Time variant only
                self.overview_subplots[output_name].plot(grid_x, values)

        for output_name in self.nanowire.simulation_outputs_dict:
            self.overview_subplots[output_name].legend().set_draggable(True)
            break
        
        self.analyze_overview_fig.tight_layout()
        self.analyze_overview_fig.canvas.draw()
        return

    ## Funcs for detailed analyze tab

    def plot_analyze(self, plot_ID, clear_plot=True):
        # Draw on analysis tab
        try:
            active_plot_data = self.analysis_plots[plot_ID]
            subplot = active_plot_data.plot_obj
            
            if clear_plot: subplot.cla()

            subplot.set_yscale(active_plot_data.yaxis_type)
            subplot.set_xscale(active_plot_data.xaxis_type)
            active_datagroup = active_plot_data.datagroup

            subplot.set_ylim(*active_plot_data.ylim)
            subplot.set_xlim(*active_plot_data.xlim)

            # This data is in TEDs units since we just used it in a calculation - convert back to common units first
            for dataset in active_datagroup.datasets.values():
                label = dataset.tag(for_matplotlib=True) + "*" if dataset.params_dict["symmetric_system"] else dataset.tag(for_matplotlib=True)
                subplot.plot(dataset.grid_x, dataset.data * self.convert_out_dict[active_datagroup.type], label=label)

            subplot.set_xlabel("x {}".format(self.nanowire.length_unit))
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

    def read_data(self, data_filename, plot_ID, datatype):
        # Create a dataset object and prepare to plot on analysis tab
        # Select data type of incoming dataset from existing datasets
        active_plot = self.analysis_plots[plot_ID]

        try:
            param_values_dict = self.fetch_metadata(data_filename)
                    
            data_n = int(0.5 + param_values_dict["Total-Time"] / param_values_dict["dt"])
            data_m = int(0.5 + param_values_dict["Total_length"] / param_values_dict["Node_width"])
            data_edge_x = np.linspace(0, param_values_dict["Total_length"],data_m+1)
            data_node_x = np.linspace(param_values_dict["Node_width"] / 2, param_values_dict["Total_length"] - param_values_dict["Node_width"] / 2, data_m)

        except:
            self.write(self.analysis_status, "Error: {} is missing or has unusual metadata.txt".format(data_filename))
            return

		# Now that we have the parameters from metadata, fetch the data itself
        sim_data = {}
        for sim_datatype in self.nanowire.simulation_outputs_dict:
            path_name = "{}\\{}\\{}\\{}-{}.h5".format(self.default_dirs["Data"], self.nanowire.system_ID, data_filename, data_filename, sim_datatype)
            sim_data[sim_datatype] = u_read(path_name, t0=active_plot.time_index, single_tstep=True)
        
        try:
            values = self.nanowire.prep_dataset(datatype, sim_data, param_values_dict)
            if self.nanowire.outputs_dict[datatype].is_edge: 
                new_data = Raw_Data_Set(values, data_edge_x, data_node_x, param_values_dict, datatype, data_filename, active_plot.time_index)
            else:
                new_data = Raw_Data_Set(values, data_node_x, data_node_x, param_values_dict, datatype, data_filename, active_plot.time_index)
    
        except:
            self.write(self.analysis_status, "Error: Unable to calculate {}".format(datatype))
            return

        try:
            active_plot.datagroup.add(new_data, new_data.tag())

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

        try:
            datatype = self.data_var.get()
            if (datatype == ""): raise ValueError("Select a data type from the drop-down menu")
        except ValueError as oops:
            self.write(self.analysis_status, oops)
            return

        active_plot.time_index = 0
        active_plot.datagroup.clear()
        
        for i in range(0, active_plot.data_filenames.__len__()):
            data_filename = active_plot.data_filenames[i]
            short_filename = data_filename[data_filename.rfind('/') + 1:]
            self.read_data(short_filename, plot_ID, datatype)

        # TODO: Better y-axis autoscaling
        active_plot.xlim = (0, active_plot.datagroup.get_max_x())
        active_plot.xaxis_type = 'linear'
        max_val = active_plot.datagroup.get_maxval() * self.convert_out_dict[active_plot.datagroup.type]
        active_plot.ylim = (max_val * 1e-11, max_val * 10)
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

        active_datagroup = active_plot.datagroup

        # Search data files for data at new time step
        for tag, dataset in active_datagroup.datasets.items():
            sim_data = {}
            for sim_datatype in self.nanowire.simulation_outputs_dict:
                path_name = "{}\\{}\\{}\\{}-{}.h5".format(self.default_dirs["Data"], self.nanowire.system_ID, dataset.filename, dataset.filename, sim_datatype)
                sim_data[sim_datatype] = u_read(path_name, t0=active_plot.time_index, single_tstep=True)
        
            dataset.data = self.nanowire.prep_dataset(active_datagroup.type, sim_data, dataset.params_dict)
            dataset.show_index = active_plot.time_index
            
        # except:
        #     self.write(self.analysis_status, "Error #107: Data group has an invalid datatype")

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
        self.sim_warning_msg = ""
        
        for IC in IC_files:
            batch_num += 1
            self.IC_file_name = IC
            self.load_ICfile()
            self.write(self.status, "Now calculating {} : ({} of {})".format(self.IC_file_name[self.IC_file_name.rfind("/") + 1:self.IC_file_name.rfind(".txt")], str(batch_num), str(IC_files.__len__())))
            self.do_Calculate()
            
        self.write(self.status, "Simulations complete")

        if not self.sim_warning_msg == "":
            sim_warning_popup = tk.Toplevel(self.root)
            sim_warning_textbox = tk.ttk.Label(sim_warning_popup, text=self.sim_warning_msg)
            sim_warning_textbox.grid(row=0,column=0)
        return

	# The big function that does all the simulating
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

            init_conditions = self.nanowire.calc_inits()
            
            for variable in self.nanowire.simulation_outputs_dict:
                if not variable in init_conditions:
                    raise KeyError
            
        except ValueError:
            self.sim_warning_msg += "Error: Invalid parameters for {}\n".format(data_file_name)

            return
        
        except KeyError:
            self.sim_warning_msg += "Error: Module calc_inits() did not return values for all simulation output variables\n"
            return

        except Exception as oops:
            self.sim_warning_msg += "Error: \"{}\" reported while setting up {}\n".format(oops, data_file_name)
            return
    
        try:
            print("Attempting to create {} data folder".format(data_file_name))
            full_path_name = "{}\\{}\\{}".format(self.default_dirs["Data"], self.nanowire.system_ID, data_file_name)
            # Append a number to the end of the new directory's name if an overwrite would occur
            # This is what happens if you download my_file.txt twice and the second copy is saved as my_file(1).txt, for example
            
            if os.path.isdir(full_path_name):
                print("{} folder already exists; trying alternate name".format(data_file_name))
                append = 1
                while (os.path.isdir("{}({})".format(full_path_name, append))):
                    append += 1

                full_path_name = "{}({})".format(full_path_name, append)
                
                
                self.sim_warning_msg += "Overwrite warning - {} already exists in Data directory\nSaving as {} instead\n".format(data_file_name, full_path_name)
                
                data_file_name = "{}({})".format(data_file_name, append)
                
            os.mkdir("{}".format(full_path_name))

        except:
            self.sim_warning_msg += "Error: unable to create directory for results of simulation {}\n".format(shortened_IC_name)
            return


        ## Calculate!
        atom = tables.Float64Atom()

        ## Create data files
        for variable in self.nanowire.simulation_outputs_dict:
            with tables.open_file("{}\\{}-{}.h5".format(full_path_name, data_file_name, variable), mode='w') as ofstream:
                length = self.m if not self.nanowire.simulation_outputs_dict[variable].is_edge else self.m + 1

                # Important - "data" must be used as the array name here, as pytables will use the string "data" 
                # to name the attribute earray.data, which is then used to access the array
                earray = ofstream.create_earray(ofstream.root, "data", atom, (0, length))
                earray.append(np.reshape(init_conditions[variable], (1, length)))
        
        ## Setup simulation plots and plot initial
        
        self.sim_data = dict(init_conditions)
        self.update_sim_plots(0)

        # numTimeStepsDone = 0

        # WIP: Option for staggered calculate/plot: In this mode the program calculates a block of time steps, plots intermediate (N, P, E), and calculates the next block 
        # using the final time step from the previous block as the initial condition.
        # This mode can be disabled by inputting numPartitions = 1.
        #for i in range(1, self.numPartitions):
            
        #    #finite.simulate_nanowire(self.IC_file_name,self.m,int(self.n / self.numPartitions),self.dx,self.dt, *(boundaryParams), *(systemParams), False, self.alphaCof, self.thetaCof, self.fracEmitted, self.max_iter, init_conditions["N"], init_conditions["P"], init_conditions["E_field"])
        #    finite.ode_nanowire(self.IC_file_name,self.m,int(self.n / self.numPartitions),self.dx,self.dt, *(boundaryParams), *(systemParams), False, self.alphaCof, self.thetaCof, self.fracEmitted, init_conditions["N"], init_conditions["P"], init_conditions["E_field"])

        #    numTimeStepsDone += int(self.n / self.numPartitions)
        #    self.write(self.status, "Calculations {:.1f}% complete".format(100 * i / self.numPartitions))
            

        #    with tables.open_file("Data\\" + self.IC_file_name + "\\" + self.IC_file_name + "-n.h5", mode='r') as ifstream_N, \
        #        tables.open_file("Data\\" + self.IC_file_name + "\\" + self.IC_file_name + "-p.h5", mode='r') as ifstream_P, \
        #        tables.open_file("Data\\" + self.IC_file_name + "\\" + self.IC_file_name + "-E_field.h5", mode='r') as ifstream_E_field:
        #        init_conditions["N"] = ifstream_N.root.data[-1]
        #        init_conditions["P"] = ifstream_P.root.data[-1]
        #        init_conditions["E_field"] = ifstream_E_field.root.data[-1]


        #    self.update_sim_plots(int(self.n * i / self.numPartitions), self.numPartitions > 20)
            #self.update_err_plots()

        write_output = True

        try:
            error_dict = self.nanowire.simulate("{}\\{}".format(full_path_name,data_file_name), self.m, self.n, self.dt, 
                                                temp_sim_dict, self.sys_flag_dict, self.check_do_ss.get(), init_conditions, write_output)
            
        except FloatingPointError:
            self.sim_warning_msg += ("Error: an unusual value occurred while simulating {}\n".format(data_file_name))
            return
        except Exception as oops:
            self.sim_warning_msg += ("Error \"{}\" occurred while simulating {}\n".format(oops, data_file_name))
            return
            
        grid_t = np.linspace(self.dt, self.simtime, self.n)

        try:
            np.savetxt(full_path_name + "\\convergence.csv", np.vstack((grid_t, error_dict['hu'], error_dict['tcur'],\
                error_dict['tolsf'], error_dict['tsw'], error_dict['nst'], error_dict['nfe'], error_dict['nje'], error_dict['nqu'],\
                error_dict['mused'])).transpose(), fmt='%.4e', delimiter=',', header="t, hu, tcur, tolsf, tsw, nst, nfe, nje, nqu, mused")
        except PermissionError:
            print("Error: unable to access convergence data export destination")
        
        self.write(self.status, "Finalizing...")

        for i in range(1,6):
            for var in self.sim_data:
                path_name = "{}\\{}\\{}\\{}-{}.h5".format(self.default_dirs["Data"], self.nanowire.system_ID, data_file_name, data_file_name, var)
                self.sim_data[var] = u_read(path_name, t0=int(self.n * i / 5), single_tstep=True)
            self.update_sim_plots(self.n, do_clear_plots=False)

        # Save metadata: list of param values used for the simulation
        # Inverting the unit conversion between the inputted params and the calculation engine is also necessary to regain the originally inputted param values

        with open(full_path_name + "\\metadata.txt", "w+") as ofstream:
            ofstream.write("$$ METADATA FOR CALCULATIONS PERFORMED ON {} AT {}\n".format(datetime.datetime.now().date(),datetime.datetime.now().time()))
            ofstream.write("System_class: {}\n".format(self.nanowire.system_ID))
            ofstream.write("Total_length: {}\n".format(self.nanowire.total_length))
            ofstream.write("Node_width: {}\n".format(self.nanowire.dx))
            
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
            ofstream.write("Total-Time: {}\n".format(self.simtime))
            ofstream.write("dt: {}\n".format(self.dt))
            for flag in self.sys_flag_dict:
                ofstream.write("{}: {}\n".format(flag, self.sys_flag_dict[flag].value()))
            
            ofstream.write("steady_state_exc: {}\n".format(self.check_do_ss.get()))

        return

    def do_Integrate(self, bypass_inputs=False):
        plot_ID = self.active_analysisplot_ID.get()
        
        # Replace this with an appropriate getter function if more integration plots are added
        ip_ID = 0
        
        
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
                self.integration_plots[ip_ID].x_param = self.xaxis_param
    
            else:
                self.integration_plots[ip_ID].x_param = "Time"
                
        else:
            # A "default integration behavior": integrate the present data over all time and space steps
            self.PL_mode = "All time steps"
            self.integration_plots[ip_ID].x_param = "Time"
            self.integration_bounds = [[0,active_datagroup.get_max_x()]]
            
        # Clean up the I_plot and prepare to integrate given selections
        # A lot of the following is a data transfer between the sending active_datagroup and the receiving I_plot
        self.integration_plots[ip_ID].datagroup.clear()
        self.integration_plots[ip_ID].mode = self.PL_mode
        self.integration_plots[ip_ID].global_gridx = None

        
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
            symmetric_flag = active_datagroup.datasets[tag].params_dict["symmetric_system"]

            if self.PL_mode == "Current time step":
                show_index = active_datagroup.datasets[tag].show_index
            else:
                show_index = None

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

                j = finite.toIndex(u_bound, dx, total_length)
                if include_negative:
                    i = finite.toIndex(-l_bound, dx, total_length)
                    nen = [-l_bound > finite.toCoord(i, dx) + dx / 2,
                                       u_bound > finite.toCoord(j, dx) + dx / 2]
                else:
                    i = finite.toIndex(l_bound, dx, total_length)
                    nen = u_bound > finite.toCoord(j, dx) + dx / 2 or l_bound == u_bound
                
                m = int(total_length / dx)
            
                do_curr_t = self.PL_mode == "Current time step"
                
                pathname = self.default_dirs["Data"] + "\\" + self.nanowire.system_ID + "\\" + data_filename + "\\" + data_filename
                
                if include_negative:
                    sim_data = {}
                    extra_data = {}
                    
                    for sim_datatype in self.nanowire.simulation_outputs_dict:
                        sim_data[sim_datatype] = u_read("{}-{}.h5".format(pathname, sim_datatype), t0=show_index, l=0, r=i+1, single_tstep=do_curr_t, need_extra_node=nen[0]) 
                        extra_data[sim_datatype] = u_read("{}-{}.h5".format(pathname, sim_datatype), t0=show_index, single_tstep=do_curr_t)
            
                    data = self.nanowire.prep_dataset(datatype, sim_data, active_datagroup.datasets[tag].params_dict, False, 0, i, nen[0], extra_data)
                    I_data = finite.new_integrate(data, 0, -l_bound, dx, total_length, nen[0])
                    sim_data = {}
                    
                    for sim_datatype in self.nanowire.simulation_outputs_dict:
                        sim_data[sim_datatype] = u_read("{}-{}.h5".format(pathname, sim_datatype), t0=show_index, l=0, r=j+1, single_tstep=do_curr_t, need_extra_node=nen[1]) 
            
                    data = self.nanowire.prep_dataset(datatype, sim_data, active_datagroup.datasets[tag].params_dict, False, 0, j, nen[1], extra_data)
                    I_data += finite.new_integrate(data, 0, u_bound, dx, total_length, nen[1])
                    
                else:
                    sim_data = {}
                    extra_data = {}
                    for sim_datatype in self.nanowire.simulation_outputs_dict:
                        sim_data[sim_datatype] = u_read("{}-{}.h5".format(pathname, sim_datatype), t0=show_index, l=i, r=j+1, single_tstep=do_curr_t, need_extra_node=nen) 
                        extra_data[sim_datatype] = u_read("{}-{}.h5".format(pathname, sim_datatype), t0=show_index, single_tstep=do_curr_t) 
            
                    data = self.nanowire.prep_dataset(datatype, sim_data, active_datagroup.datasets[tag].params_dict, False, i, j, nen, extra_data)
                    
                    I_data = finite.new_integrate(data, l_bound, u_bound, dx, total_length, nen)

                            
                if self.PL_mode == "Current time step":
                    # Don't forget to change out of TEDs units, or the x axis won't match the parameters the user typed in
                    grid_xaxis = float(active_datagroup.datasets[tag].params_dict[self.xaxis_param] * self.convert_out_dict[self.xaxis_param])

                    xaxis_label = self.xaxis_param + " [WIP]"

                elif self.PL_mode == "All time steps":
                    self.integration_plots[ip_ID].global_gridx = np.linspace(0, total_time, n + 1)
                    grid_xaxis = -1 # A dummy value for the I_Set constructor
                    xaxis_label = "Time [ns]"

                self.integration_plots[ip_ID].datagroup.add(Integrated_Data_Set(I_data, grid_xaxis, active_datagroup.datasets[tag].params_dict, active_datagroup.datasets[tag].type, data_filename + "__" + str(l_bound) + "_to_" + str(u_bound)))
            
                counter += 1
                print("Integration: {} of {} complete".format(counter, active_datagroup.size() * self.integration_bounds.__len__()))

        subplot = self.integration_plots[ip_ID].plot_obj
        datagroup = self.integration_plots[ip_ID].datagroup
        subplot.cla()
        
        #max = datagroup.get_maxval() * self.convert_out_dict[datagroup.type]
        

        self.integration_plots[ip_ID].xaxis_type = 'linear'
        self.integration_plots[ip_ID].yaxis_type = 'log'
        #self.integration_plots[ip_ID].ylim = max * 1e-12, max * 10

        subplot.set_yscale(self.integration_plots[ip_ID].yaxis_type)
        #subplot.set_ylim(self.integration_plots[ip_ID].ylim)
        subplot.set_xlabel(xaxis_label)
        subplot.set_ylabel(datagroup.type)
        subplot.set_title("Integrated {}".format(datagroup.type))

        for key in datagroup.datasets:

            if self.PL_mode == "Current time step":
                subplot.scatter(datagroup.datasets[key].grid_x, datagroup.datasets[key].data * self.convert_out_dict[datagroup.type], label=datagroup.datasets[key].tag(for_matplotlib=True))

            elif self.PL_mode == "All time steps":
                subplot.plot(self.integration_plots[ip_ID].global_gridx, datagroup.datasets[key].data * self.convert_out_dict[datagroup.type], label=datagroup.datasets[key].tag(for_matplotlib=True))
                
        self.integration_plots[ip_ID].xlim = subplot.get_xlim()
        self.integration_plots[ip_ID].ylim = subplot.get_ylim()
                
        subplot.legend().set_draggable(True)

        self.integration_fig.tight_layout()
        self.integration_fig.canvas.draw()
        
        self.write(self.analysis_status, "Integration complete")

        if (self.nanowire.system_ID == "Nanowire" and self.PL_mode == "All time steps" and datatype == "PL"):
            # Calculate tau_D
            if self.integration_plots[ip_ID].datagroup.size(): # If has tau_diff data to plot
                td_gridt = {}
                td = {}
                
                for tag, dataset in self.integration_plots[ip_ID].datagroup.datasets.items():
                    total_time = dataset.params_dict["Total-Time"]
                    dt = dataset.params_dict["dt"]
                    td_gridt[tag] = np.linspace(0, total_time, n + 1)
                    td[tag] = finite.tau_diff(dataset.data, dt)
                    
                td_popup = tk.Toplevel(self.root)
                td_fig = Figure(figsize=(6,4))
                td_subplot = td_fig.add_subplot(1, 1, 1)
                
                td_canvas = tkagg.FigureCanvasTkAgg(td_fig, master=td_popup)
                td_plotwidget = td_canvas.get_tk_widget()
                td_plotwidget.grid(row=0,column=0)
                
                
                
                td_subplot.set_ylabel("tau_diff")
                td_subplot.set_xlabel("Time [ns]")
                td_subplot.set_title("-(dln(PL)/dt)^(-1)")
                for tag in td:
                    td_subplot.plot(td_gridt[tag], td[tag], label=tag.strip('_'))
            
                td_subplot.legend().set_draggable(True)
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
            self.paramtoolkit_currentparam = param
            
            # These two lines changes the text displayed in the param_rule display box's menu and is for cosmetic purposes only
            self.update_paramrule_listbox(param)
            self.paramtoolkit_viewer_selection.set(param)
            
            self.deleteall_paramrule()
            
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

        # Remove all param_rules for deltaN and deltaP, as we will be reassigning them shortly.
        self.paramtoolkit_currentparam = "deltaN"
        self.deleteall_paramrule()
        self.paramtoolkit_currentparam = "deltaP"
        self.deleteall_paramrule()

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
            self.nanowire.param_dict["deltaN"].value = finite.pulse_laser_power_spotsize(power, spotsize, freq, wavelength, alpha_nm, self.nanowire.grid_x_nodes, hc=hc_nm)
        
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

            self.nanowire.param_dict["deltaN"].value = finite.pulse_laser_powerdensity(power_density, freq, wavelength, alpha_nm, self.nanowire.grid_x_nodes, hc=hc_nm)
        
        elif (AIC_options["power_mode"] == "max-gen"):
            try: max_gen = float(self.max_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing max gen")
                return

            self.nanowire.param_dict["deltaN"].value = finite.pulse_laser_maxgen(max_gen, alpha_nm, self.nanowire.grid_x_nodes)
        

        elif (AIC_options["power_mode"] == "total-gen"):
            try: total_gen = float(self.total_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except ValueError:
                self.write(self.ICtab_status, "Error: missing total gen")
                return

            self.nanowire.param_dict["deltaN"].value = finite.pulse_laser_totalgen(total_gen, self.nanowire.total_length, alpha_nm, self.nanowire.grid_x_nodes)
        
        else:
            self.write(self.ICtab_status, "An unexpected error occurred while calculating the power generation params")
            return
        
        
        ## TODO: Make AIC deposit in common units, so this patch isn't required
        self.nanowire.param_dict["deltaN"].value *= self.convert_out_dict["deltaN"]
        ## Assuming that the initial distributions of holes and electrons are identical
        self.nanowire.param_dict["deltaP"].value = self.nanowire.param_dict["deltaN"].value

        self.update_IC_plot(plot_ID="AIC")
        self.paramtoolkit_currentparam = "deltaN"
        self.update_IC_plot(plot_ID="custom")
        self.paramtoolkit_currentparam = "deltaP"
        self.update_IC_plot(plot_ID="recent")
        self.using_AIC = True
        return

    ## Special functions for Parameter Toolkit:
    def add_paramrule(self):
        # V2 update
        # Set the value of one of Nanowire's Parameters

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

                if (float(self.paramrule_lbound_entry.get()) < 0):
                    raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.paramrule_lbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "POINT", float(self.paramrule_lbound_entry.get()), -1, float(self.paramrule_lvalue_entry.get()), -1)

            elif (self.init_shape_selection.get() == "FILL"):
                if (float(self.paramrule_lbound_entry.get()) < 0 or float(self.paramrule_rbound_entry.get()) > self.nanowire.total_length):
                	raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.paramrule_lbound_entry.get()) > float(self.paramrule_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.paramrule_lbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "FILL", float(self.paramrule_lbound_entry.get()), float(self.paramrule_rbound_entry.get()), float(self.paramrule_lvalue_entry.get()), -1)

            elif (self.init_shape_selection.get() == "LINE"):
                if (float(self.paramrule_lbound_entry.get()) < 0 or float(self.paramrule_rbound_entry.get()) > self.nanowire.total_length):
                	raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.paramrule_lbound_entry.get()) > float(self.paramrule_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.paramrule_lbound_entry.get()) < 0 or float(self.paramrule_rbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "LINE", float(self.paramrule_lbound_entry.get()), float(self.paramrule_rbound_entry.get()), 
                                            float(self.paramrule_lvalue_entry.get()), float(self.paramrule_rvalue_entry.get()))

            elif (self.init_shape_selection.get() == "EXP"):
                if (float(self.paramrule_lbound_entry.get()) < 0 or float(self.paramrule_rbound_entry.get()) > self.nanowire.total_length):
                    raise Exception("Error: Bound coordinates exceed system thickness specifications")

                if (float(self.paramrule_lbound_entry.get()) > float(self.paramrule_rbound_entry.get())):
                	raise Exception("Error: Left bound coordinate is larger than right bound coordinate")

                if (float(self.paramrule_lbound_entry.get()) < 0 or float(self.paramrule_rbound_entry.get()) < 0):
                	self.write(self.ICtab_status, "Warning: negative initial condition value")

                new_param_rule = Param_Rule(new_param_name, "EXP", float(self.paramrule_lbound_entry.get()), float(self.paramrule_rbound_entry.get()), 
                                            float(self.paramrule_lvalue_entry.get()), float(self.paramrule_rvalue_entry.get()))

            else:
                raise Exception("Error: No init. type selected")

        except ValueError:
            self.write(self.ICtab_status, "Error: Missing Parameters")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        self.nanowire.add_param_rule(new_param_name, new_param_rule)

        self.paramtoolkit_viewer_selection.set(new_param_name)
        self.update_paramrule_listbox(new_param_name)

        if self.nanowire.system_ID == "Nanowire":
            if new_param_name == "deltaN" or new_param_name == "deltaP": self.using_AIC = False
        self.update_IC_plot(plot_ID="recent")
        return

    def refresh_paramrule_listbox(self):
        # The View button has two jobs: change the listbox to the new param and display a snapshot of it
        self.update_paramrule_listbox(self.paramtoolkit_viewer_selection.get())
        self.update_IC_plot(plot_ID="custom")
        return
    
    def update_paramrule_listbox(self, param_name):
        # Grab current param's rules from Nanowire and show them in the param_rule listbox
        if param_name == "":
            self.write(self.ICtab_status, "Select a parameter")
            return

        # 1. Clear the viewer
        self.hideall_paramrules()

        # 2. Write in the new rules
        current_param_rules = self.nanowire.param_dict[param_name].param_rules
        self.paramtoolkit_currentparam = param_name

        for param_rule in current_param_rules:
            self.active_paramrule_list.append(param_rule)
            self.active_paramrule_listbox.insert(self.active_paramrule_list.__len__() - 1, param_rule.get())

        
        self.write(self.ICtab_status, "")

        return

    # These two reposition the order of param_rules
    def moveup_paramrule(self):
        try:
            currentSelectionIndex = self.active_paramrule_listbox.curselection()[0]
        except IndexError:
            return
        
        if (currentSelectionIndex > 0):
            # Two things must be done here for a complete swap:
            # 1. Change the order param rules appear in the box
            self.active_paramrule_list[currentSelectionIndex], self.active_paramrule_list[currentSelectionIndex - 1] = self.active_paramrule_list[currentSelectionIndex - 1], self.active_paramrule_list[currentSelectionIndex]
            self.active_paramrule_listbox.delete(currentSelectionIndex)
            self.active_paramrule_listbox.insert(currentSelectionIndex - 1, self.active_paramrule_list[currentSelectionIndex - 1].get())
            self.active_paramrule_listbox.selection_set(currentSelectionIndex - 1)

            # 2. Change the order param rules are applied when calculating Parameter's values
            self.nanowire.swap_param_rules(self.paramtoolkit_currentparam, currentSelectionIndex)
            self.update_IC_plot(plot_ID="recent")
        return

    def movedown_paramrule(self):
        try:
            currentSelectionIndex = self.active_paramrule_listbox.curselection()[0] + 1
        except IndexError:
            return
        
        if (currentSelectionIndex < self.active_paramrule_list.__len__()):
            self.active_paramrule_list[currentSelectionIndex], self.active_paramrule_list[currentSelectionIndex - 1] = self.active_paramrule_list[currentSelectionIndex - 1], self.active_paramrule_list[currentSelectionIndex]
            self.active_paramrule_listbox.delete(currentSelectionIndex)
            self.active_paramrule_listbox.insert(currentSelectionIndex - 1, self.active_paramrule_list[currentSelectionIndex - 1].get())
            self.active_paramrule_listbox.selection_set(currentSelectionIndex)
            
            self.nanowire.swap_param_rules(self.paramtoolkit_currentparam, currentSelectionIndex)
            self.update_IC_plot(plot_ID="recent")
        return

    def hideall_paramrules(self, doPlotUpdate=True):
        # Wrapper - Call hide_paramrule() until listbox is empty
        while (self.active_paramrule_list.__len__() > 0):
            # These first two lines mimic user repeatedly selecting topmost paramrule in listbox
            self.active_paramrule_listbox.select_set(0)
            self.active_paramrule_listbox.event_generate("<<ListboxSelect>>")

            self.hide_paramrule()
        return

    def hide_paramrule(self):
        # Remove user-selected param rule from box (but don't touch Nanowire's saved info)
        self.active_paramrule_list.pop(self.active_paramrule_listbox.curselection()[0])
        self.active_paramrule_listbox.delete(self.active_paramrule_listbox.curselection()[0])
        return
    
    def deleteall_paramrule(self, doPlotUpdate=True):
        # Note: deletes all rules for currentparam
        # Use reset_IC instead to delete all rules for every param
        if (self.nanowire.param_dict[self.paramtoolkit_currentparam].param_rules.__len__() > 0):
            self.nanowire.removeall_param_rules(self.paramtoolkit_currentparam)
            self.hideall_paramrules()
            self.update_IC_plot(plot_ID="recent")
        return

    def delete_paramrule(self):
        # Remove user-selected param rule from box AND from Nanowire's list of param_rules
        if (self.nanowire.param_dict[self.paramtoolkit_currentparam].param_rules.__len__() > 0):
            try:
                self.nanowire.remove_param_rule(self.paramtoolkit_currentparam, self.active_paramrule_listbox.curselection()[0])
                self.hide_paramrule()
                self.update_IC_plot(plot_ID="recent")
            except IndexError:
                self.write(self.ICtab_status, "No rule selected")
                return
        return

    # Fill IC arrays using list from .txt file
    def add_listupload(self):
        try:
            self.set_init_x()

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return
        
        warning_flag = False
        var = self.listupload_var_selection.get()
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

            # Linear interpolate from provided param list to specified grid points
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
                
        if self.nanowire.system_ID == "Nanowire":
            if var == "deltaN" or var == "deltaP": self.using_AIC = False
        
        self.paramtoolkit_currentparam = var
        self.deleteall_paramrule()
        self.nanowire.param_dict[var].value = temp_IC_values
        self.update_IC_plot(plot_ID="listupload", warn=warning_flag)
        self.update_IC_plot(plot_ID="recent", warn=warning_flag)
        return

    def update_IC_plot(self, plot_ID, warn=False):
        # V2 update: can now plot any parameter
        # Plot 2 is for recently changed parameter while plot 1 is for user-selected views

        if plot_ID=="recent": plot = self.recent_param_subplot
        elif plot_ID=="custom": plot = self.custom_param_subplot
        elif plot_ID=="AIC": plot = self.AIC_subplot
        elif plot_ID=="listupload": plot = self.listupload_subplot
        plot.cla()

        if plot_ID=="AIC": param_name="deltaN"
        else: param_name = self.paramtoolkit_currentparam
        
        param_obj = self.nanowire.param_dict[param_name]
        grid_x = self.nanowire.grid_x_edges if param_obj.is_edge else self.nanowire.grid_x_nodes
        # Support for constant value shortcut: temporarily create distribution
        # simulating filling across nanowire with that value
        val_array = finite.toArray(param_obj.value, len(self.nanowire.grid_x_nodes), param_obj.is_edge)

        plot.set_yscale(autoscale(val_array))

        if self.sys_flag_dict['symmetric_system'].value():
            plot.plot(np.concatenate((-np.flip(grid_x), grid_x), axis=0), np.concatenate((np.flip(val_array), val_array), axis=0), label=param_name)

            ymin, ymax = plot.get_ylim()
            plot.fill([-grid_x[-1], 0, 0, -grid_x[-1]], [ymin, ymin, ymax, ymax], 'b', alpha=0.1, edgecolor='r')
        else:
            plot.plot(grid_x, val_array, label=param_name)

        plot.set_xlabel("x {}".format(self.nanowire.length_unit))
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
        elif plot_ID=="listupload": 
            plot.set_title("Recent list upload")
            self.listupload_fig.tight_layout()
            self.listupload_fig.canvas.draw()

        if not warn: self.write(self.ICtab_status, "Initial Condition Updated")
        return

    ## Initial Condition I/O

    def create_batch_init(self):
        warning_flag = 0
        try:
            batch_values = {}

            for batchable in self.batchables_array:
                if batchable.param_name.get():
                    batch_values[batchable.param_name.get()] = extract_values(batchable.tk_entrybox.get(), ' ')
            
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
        batch_combinations = finite.get_all_combinations(batch_values)        
                
        # Apply each combination to Nanowire, going through AIC if necessary
        for batch_set in batch_combinations:
            filename = ""
            for param in batch_set:
                filename += str("__{}_{:.4e}".format(param, batch_set[param]))
                
                if self.nanowire.system_ID == "Nanowire" and (param in self.analytical_entryboxes_dict):
                    self.enter(self.analytical_entryboxes_dict[param], str(batch_set[param]))
                    
                else:
                    self.nanowire.param_dict[param].value = batch_set[param]

                
            if self.nanowire.system_ID == "Nanowire" and self.using_AIC: self.add_AIC()
                
            try:
                self.write_init_file("{}\\{}\\{}.txt".format(self.default_dirs["Initial"], batch_dir_name, filename))
            except:
                print("Error: failed to create batch file {}".format(filename))
                warning_flag += 1
                
        # Restore the original values of Nanowire
        for param in self.nanowire.param_dict:
            self.nanowire.param_dict[param].value = original_param_values[param]
        
        if not warning_flag:
            self.write(self.batch_status, "Batch \"{}\" created successfully".format(batch_dir_name))
        else:
            self.write(self.batch_status, "Warning: failed to create some batch files - see console")
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
                ofstream.write("System_class: {}\n".format(self.nanowire.system_ID))
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
                    ofstream.write("{}: {}\n".format(flag, self.sys_flag_dict[flag].value()))
                
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
                
                if not ("$$ INITIAL CONDITION FILE CREATED ON") in next(ifstream):
                    raise OSError("Error: this file is not a valid TEDs file")
                
                system_class = next(ifstream).strip('\n')
                system_class = system_class[system_class.find(' ') + 1:]
                if not system_class == self.nanowire.system_ID:
                    raise ValueError("Error: selected file is not a {}".format(self.nanowire.system_ID))
                                
                # Extract parameters, ICs
                for line in ifstream:
                    
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
        if self.nanowire.system_ID == "Nanowire":
            for key in self.analytical_entryboxes_dict:
                self.enter(self.analytical_entryboxes_dict[key], "")
            
        for param in self.nanowire.param_dict:
            self.paramtoolkit_currentparam = param
            
            self.update_paramrule_listbox(param)            
            self.deleteall_paramrule()
            
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
            # All we need to do here is mark the appropriate GUI elements as selected
            try:
                self.sys_flag_dict[flag].tk_var.set(flag_values_dict[flag])
            except:
                print("Warning: could not apply value for flag: {}".format(flag))
                print("Flags must have integer value 1 or 0")
                warning_flag += 1
            
        for param in self.nanowire.param_dict:
            new_value = init_param_values_dict[param]
            try:
                if '\t' in new_value:
                    self.nanowire.param_dict[param].value = np.array(extract_values(new_value, '\t'))
                else: self.nanowire.param_dict[param].value = float(new_value)
                
                self.paramtoolkit_currentparam = param
                if cycle_through_IC_plots: self.update_IC_plot(plot_ID="recent")
            except:
                print("Warning: could not apply value for param: {}".format(param))
                warning_flag += 1
                
        if self.nanowire.system_ID == "Nanowire": self.using_AIC = False
        
        if not warning_flag: self.write(self.ICtab_status, "IC file loaded successfully")
        else: self.write(self.ICtab_status, "IC file loaded with {} issue(s); see console".format(warning_flag))
        return

    # Data I/O

    def export_plot(self, from_integration):

        if from_integration:
            plot_ID = self.active_integrationplot_ID.get()
            datagroup = self.integration_plots[plot_ID].datagroup
            plot_info = self.integration_plots[plot_ID]
            if datagroup.size() == 0: return
            
            if plot_info.mode == "Current time step": 
                paired_data = [[datagroup.datasets[key].grid_x, datagroup.datasets[key].data * self.convert_out_dict[datagroup.type]] for key in datagroup.datasets]

                header = "{} {}, {}".format(plot_info.x_param, self.nanowire.param_dict[plot_info.x_param].units, datagroup.type)

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
            paired_data = self.analysis_plots[plot_ID].datagroup.build(self.convert_out_dict)
            # We need some fancy footwork using itertools to transpose a non-rectangular array
            paired_data = np.array(list(map(list, itertools.zip_longest(*paired_data, fillvalue=-1))))
            header = "".join(["x {},".format(self.nanowire.length_unit) + self.analysis_plots[plot_ID].datagroup.datasets[key].filename + "," for key in self.analysis_plots[plot_ID].datagroup.datasets])

        export_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["PL"], title="Save data", filetypes=[("csv (comma-separated-values)","*.csv")])
        
        # Export to .csv
        if not (export_filename == ""):
            try:
                if export_filename.endswith(".csv"): export_filename = export_filename[:-4]
                np.savetxt("{}.csv".format(export_filename), paired_data, fmt='%.4e', delimiter=',', header=header)
                self.write(self.analysis_status, "Export complete")
            except PermissionError:
                self.write(self.analysis_status, "Error: unable to access PL export destination")
        
        return

    def export_for_bayesim(self, ip_ID=0):
        # Note: DO NOT convert_out any of these values - bayesim models are created in TEDs units.
        datagroup = self.integration_plots[ip_ID].datagroup
        plot_info = self.integration_plots[ip_ID]
        if datagroup.size() == 0: 
            self.write(self.analysis_status, "No datasets loaded for bayesim")
            return
            
        if (plot_info.mode == "All time steps"):
            if self.bay_mode.get() == "obs":
                for key in datagroup.datasets:  # For each curve on the integration plot
                    raw_data = datagroup.datasets[key].data
                    grid_x = plot_info.global_gridx   # grid_x refers to what is on the x-axis, which in this case is technically 'time'
                    unc = raw_data * 0.1
                    full_data = np.vstack((grid_x, raw_data, unc)).T
                    full_data = pd.DataFrame.from_records(data=full_data,columns=['time', datagroup.type, 'uncertainty'])
                    
                    filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["PL"], title="Save Bayesim Model", filetypes=[("HDF5 Archive","*.h5")])
                    dd.save("{}.h5".format(filename.strip('.h5')), full_data)

            elif self.bay_mode.get() == "model":
                active_bay_params = []
                for param in self.check_bay_params:
                    if self.check_bay_params[param].get(): active_bay_params.append(param)

                is_first = True
                for key in datagroup.datasets:
                    raw_data = datagroup.datasets[key].data
                    grid_x = plot_info.global_gridx
                    paired_data = np.vstack((grid_x, raw_data))

                    try:
                        for param in active_bay_params:
                            if not isinstance(datagroup.datasets[key].params_dict[param], (int, float)):
                                raise ValueError
                            param_column = np.ones((1,raw_data.__len__())) * datagroup.datasets[key].params_dict[param] * self.convert_out_dict[param]
                            paired_data = np.concatenate((param_column, paired_data), axis=0)
                    except ValueError:
                        self.write(self.analysis_status, "WIP: Bayesim with space-dependent params not yet supported")
                        return
                    
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
                if new_filename == "": 
                    self.write(self.analysis_status, "Bayesim export cancelled")
                    return

                if not new_filename.endswith(".h5"): new_filename += ".h5"

                dd.save(new_filename, full_data)
        else:
            print("WIP =(")

        self.write(self.analysis_status, "Bayesim export complete")
        return

nb = Notebook("ted")
nb.run()

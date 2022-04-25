# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:54:47 2022

@author: cfai2
"""

import numpy as np
import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import tkinter as tk

# This lets us pass params to functions called by tkinter buttons
from functools import partial 
from GUI_structs import Flag
from GUI_structs import Analysis_Plot_State
from GUI_structs import Integration_Plot_State



np.seterr(divide='raise', over='warn', under='warn', invalid='raise')


class BaseNotebook:
    """Base class with all the necessary methods
    for the following Notebook class."""
    
    def prepare_main_canvas(self):
        self.main_canvas = tk.Canvas(self.root)
        self.main_canvas.grid(row=0,column=0, sticky='nswe')
        
        self.notebook = tk.ttk.Notebook(self.main_canvas)
        
        # Allocate room for and add scrollbars to overall notebook
        self.main_scroll_y = tk.ttk.Scrollbar(self.root, orient="vertical", 
                                              command=self.main_canvas.yview)
        self.main_scroll_y.grid(row=0,column=1, sticky='ns')
        self.main_scroll_x = tk.ttk.Scrollbar(self.root, orient="horizontal", 
                                              command=self.main_canvas.xview)
        self.main_scroll_x.grid(row=1,column=0,sticky='ew')
        self.main_canvas.configure(yscrollcommand=self.main_scroll_y.set, 
                                   xscrollcommand=self.main_scroll_x.set)
        # Make area for scrollbars as narrow as possible without cutting off
        self.root.rowconfigure(0,weight=100)
        self.root.rowconfigure(1,weight=1, minsize=20) 
        self.root.columnconfigure(0,weight=100)
        self.root.columnconfigure(1,weight=1, minsize=20)
        
        self.main_canvas.create_window((0,0), window=self.notebook, anchor="nw")
        self.notebook.bind('<Configure>', 
                           lambda e:self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all')))
        return
    
    def prepare_radiobuttons_and_checkboxes(self):
        # Tkinter checkboxes and radiobuttons require special variables 
        # to extract user input.
        # IntVars or BooleanVars are sufficient for binary choices 
        # e.g. whether a checkbox is checked
        # while StringVars are more suitable for open-ended choices 
        # e.g. selecting one mode from a list
        self.check_reset_params = tk.IntVar()
        self.check_reset_inits = tk.IntVar()
        self.check_display_legend = tk.IntVar()
        self.check_freeze_axes = tk.IntVar()
        self.check_autointegrate = tk.IntVar(value=0)
        
        self.active_analysisplot_ID = tk.IntVar()
        self.active_integrationplot_ID = tk.IntVar()
        return
    
    def prepare_initial_things(self):
        self.init_shape_selection = tk.StringVar()
        self.init_var_selection = tk.StringVar()
        self.paramtoolkit_viewer_selection = tk.StringVar()
        self.listupload_var_selection = tk.StringVar()
        self.display_selection = tk.StringVar()

        # Stores whatever layer is being displayed in the layer selection box
        self.current_layer_selection = tk.StringVar()
        # Stores whatever layer TEDs is currently operating on
        self.current_layer_name = ""
        
        # Pressing the layer change button updates self.current_layer_name to the
        # value of self.current_layer_selection
        
        # Flags and containers for IC arrays
        self.active_paramrule_list = []
        self.paramtoolkit_currentparam = ""
        self.IC_file_list = None
        self.IC_file_name = ""
        
        self.carryover_include_flags = {}
        for layer_name, layer in self.module.layers.items():
            self.carryover_include_flags[layer_name] = {}
            for var in layer.s_outputs:
                self.carryover_include_flags[layer_name][var] = tk.IntVar()
            
        # Helpers, flags, and containers for analysis plots
        self.analysis_plots = [Analysis_Plot_State(), Analysis_Plot_State(), 
                               Analysis_Plot_State(), Analysis_Plot_State()]
        self.integration_plots = [Integration_Plot_State()]
        self.data_var = tk.StringVar()
        self.fetch_PLmode = tk.StringVar()
        self.fetch_intg_mode = tk.StringVar()
        self.yaxis_type = tk.StringVar()
        self.xaxis_type = tk.StringVar()

        # Track which timeseries popups are open
        self.active_timeseries = {}
        return
    
    def prepare_eligible_modules(self):
        # Add (e.g. for Nanowire) module-specific functionality
        # TODO: abstract the choices away from this code
        self.LGC_eligible_modules = ("Nanowire", "OneLayer", "MAPI_Rubrene")
        if self.module.system_ID in self.LGC_eligible_modules:
            self.using_LGC = {}
            self.LGC_options = {}
            self.LGC_values = {}
            
        return
    
    def reset_popup_flags(self):
        # Record when popup menus open and close
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
        return
    
    # Create GUI elements for each tab
    # Tkinter works a bit like a bulletin board:
    # we declare an overall frame and
    # pin things to it at specified locations
    # This includes other frames, which is evident
    # in how the tab_inputs has three sub-tabs pinned to itself
    
    def add_tab_inputs(self):
        """ Add the menu tab 'Inputs' to the main notebook"""
        self.tab_inputs = tk.ttk.Notebook(self.notebook)
        self.tab_generation_init = tk.ttk.Frame(self.tab_inputs)
        self.tab_rules_init = tk.ttk.Frame(self.tab_inputs)
        self.tab_explicit_init = tk.ttk.Frame(self.tab_inputs)
        
        first_layer = next(iter(self.module.layers))

        var_dropdown_list = ["{} {}".format(param_name, param.units) 
                             for param_name, param in self.module.layers[first_layer].params.items()
                             if param.is_space_dependent]
        paramtoolkit_method_dropdown_list = ["POINT", "FILL", "LINE", "EXP"]
        unitless_dropdown_list = [param_name 
                                  for param_name, param in self.module.layers[first_layer].params.items()
                                  if param.is_space_dependent]
        
        self.line_sep_style = tk.ttk.Style()
        self.line_sep_style.configure("Grey Bar.TSeparator", background='#000000', 
                                      padding=160)

        self.header_style = tk.ttk.Style()
        self.header_style.configure("Header.TLabel", background='#D0FFFF',
                                    highlightbackground='#000000')

		# We use the grid location specifier for general placement and padx/pady for fine-tuning
		# The other two options are the pack specifier, which doesn't really provide enough versatility,
		# and absolute coordinates, which maximize versatility but are a pain to adjust manually.
        self.IO_frame = tk.ttk.Frame(self.tab_inputs)
        self.IO_frame.grid(row=0,column=0,columnspan=2, pady=(25,0))
        
        tk.ttk.Button(self.IO_frame, text="Load", 
                      command=self.select_init_file).grid(row=0,column=0)

        tk.ttk.Button(self.IO_frame, text="debug", 
                      command=self.DEBUG).grid(row=0,column=1)

        tk.ttk.Button(self.IO_frame, text="Save", 
                      command=self.save_ICfile).grid(row=0,column=2)

        tk.ttk.Button(self.IO_frame, text="Reset", 
                      command=self.reset_IC).grid(row=0, column=3)

        self.spacegrid_frame = tk.ttk.Frame(self.tab_inputs)
        self.spacegrid_frame.grid(row=1,column=0,columnspan=2, pady=(10,10))

        tk.ttk.Label(self.spacegrid_frame, 
                    text="Space Grid - Start Here", 
                    style="Header.TLabel").grid(row=0,column=0,columnspan=2)

        tk.ttk.Label(self.spacegrid_frame, 
                    text="Thickness " 
                    + self.module.layers[first_layer].length_unit).grid(row=1,column=0)

        self.thickness_entry = tk.ttk.Entry(self.spacegrid_frame, width=9)
        self.thickness_entry.grid(row=1,column=1)

        tk.ttk.Label(self.spacegrid_frame, 
                    text="Node width " 
                    + self.module.layers[first_layer].length_unit).grid(row=2,column=0)

        self.dx_entry = tk.ttk.Entry(self.spacegrid_frame, width=9)
        self.dx_entry.grid(row=2,column=1)

        self.params_frame = tk.ttk.Frame(self.tab_inputs)
        self.params_frame.grid(row=2,column=0,columnspan=2, rowspan=4, pady=(10,10))

        tk.ttk.Label(self.params_frame, 
                    text="Constant-value Parameters",
                    style="Header.TLabel").grid(row=0, column=0,columnspan=2)
        
        tk.ttk.Button(self.params_frame, 
                      text="Fast Param Entry Tool", 
                      command=self.do_sys_param_shortcut_popup).grid(row=1,column=0,columnspan=2)
        
        self.flags_frame = tk.ttk.Frame(self.tab_inputs)
        self.flags_frame.grid(row=6,column=0,columnspan=2, pady=(10,10))

        tk.ttk.Label(self.flags_frame, text="Flags", 
                     style="Header.TLabel").grid(row=0,column=0,columnspan=2)
        
        # Procedurally generated elements for flags
        i = 1
        self.sys_flag_dict = {}
        for flag in self.module.flags_dict:
            self.sys_flag_dict[flag] = Flag(self.flags_frame, 
                                            self.module.flags_dict[flag][0])
            self.sys_flag_dict[flag].set(self.module.flags_dict[flag][2])
            
            if not self.module.flags_dict[flag][1]:
                continue
            else:
                self.sys_flag_dict[flag].tk_element.grid(row=i,column=0)
                i += 1
                
        self.ICtab_status = tk.Text(self.tab_inputs, width=24,height=8)
        self.ICtab_status.grid(row=7, column=0, columnspan=2)
        self.ICtab_status.configure(state='disabled')
        
        tk.ttk.Button(self.tab_inputs, 
                      text="Print Init. State Summary", 
                      command=self.do_sys_printsummary_popup).grid(row=8,column=0,columnspan=2)
        
        tk.ttk.Button(self.tab_inputs, 
                      text="Show Init. State Plots", 
                      command=self.do_sys_plotsummary_popup).grid(row=9,column=0,columnspan=2)
        
        tk.ttk.Separator(self.tab_inputs, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=10,column=0,columnspan=2, pady=(10,10), sticky="ew")
        
        self.layer_statusbox = tk.Text(self.tab_inputs, width=24, height=1)
        self.layer_statusbox.grid(row=11,column=0,columnspan=2)
        
        # Init this dropdown with some default layer
        tk.ttk.OptionMenu(self.tab_inputs, self.current_layer_selection,
                          first_layer, *self.module.layers).grid(row=12,column=0)
        
        tk.ttk.Button(self.tab_inputs, text="Change to Layer",
                      command=self.change_layer).grid(row=12,column=1)
        
        tk.ttk.Separator(self.tab_inputs, orient="vertical", 
                         style="Grey Bar.TSeparator").grid(row=0,rowspan=30,column=2,pady=(24,0),sticky="ns")
             
        ## Parameter Toolkit:

        self.param_rules_frame = tk.ttk.Frame(self.tab_rules_init)
        self.param_rules_frame.grid(row=0,column=0,padx=(370,0))

        tk.ttk.Label(self.param_rules_frame, 
                     text="Add/Edit/Remove Space-Dependent Parameters", 
                     style="Header.TLabel").grid(row=0,column=0,columnspan=3)

        self.active_paramrule_listbox = tk.Listbox(self.param_rules_frame, width=86,
                                                   height=8)
        self.active_paramrule_listbox.grid(row=1,rowspan=3,column=0,columnspan=3, 
                                           padx=(32,32))

        tk.ttk.Label(self.param_rules_frame, 
                     text="Select parameter to edit:").grid(row=4,column=0)
        
        tk.ttk.OptionMenu(self.param_rules_frame, 
                          self.init_var_selection, 
                          var_dropdown_list[0], 
                          *var_dropdown_list).grid(row=4,column=1)

        tk.ttk.Label(self.param_rules_frame, 
                     text="Select calculation method:").grid(row=5,column=0)

        tk.ttk.OptionMenu(self.param_rules_frame, 
                          self.init_shape_selection,
                          paramtoolkit_method_dropdown_list[0], 
                          *paramtoolkit_method_dropdown_list).grid(row=5, column=1)

        tk.ttk.Label(self.param_rules_frame, 
                     text="Left bound coordinate:").grid(row=6, column=0)

        self.paramrule_lbound_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_lbound_entry.grid(row=6,column=1)

        tk.ttk.Label(self.param_rules_frame, 
                     text="Right bound coordinate:").grid(row=7, column=0)

        self.paramrule_rbound_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_rbound_entry.grid(row=7,column=1)

        tk.ttk.Label(self.param_rules_frame, 
                     text="Left bound value:").grid(row=8, column=0)

        self.paramrule_lvalue_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_lvalue_entry.grid(row=8,column=1)

        tk.ttk.Label(self.param_rules_frame, 
                     text="Right bound value:").grid(row=9, column=0)

        self.paramrule_rvalue_entry = tk.ttk.Entry(self.param_rules_frame, width=8)
        self.paramrule_rvalue_entry.grid(row=9,column=1)

        tk.ttk.Button(self.param_rules_frame, 
                      text="Add new parameter rule", 
                      command=self.add_paramrule).grid(row=10,column=0,columnspan=2)

        tk.ttk.Button(self.param_rules_frame, 
                      text="Delete highlighted rule", 
                      command=self.delete_paramrule).grid(row=4,column=2)

        tk.ttk.Button(self.param_rules_frame, 
                      text="Delete all rules for this parameter", 
                      command=self.deleteall_paramrule).grid(row=5,column=2)

        tk.Message(self.param_rules_frame, 
                   text="The Parameter Toolkit uses a series "
                   "of rules and patterns to build a spatially "
                   "dependent distribution for any parameter.", 
                   width=250).grid(row=6,rowspan=3,column=2,columnspan=2)

        tk.Message(self.param_rules_frame, 
                   text="Warning: Rules are applied "
                   "from top to bottom. Order matters!", 
                   width=250).grid(row=9,rowspan=3,column=2,columnspan=2)
        
        # These plots were previously attached to self.tab_inputs so that it was visible on all three IC tabs,
        # but it was hard to position them correctly.
        # Attaching to the Parameter Toolkit makes them easier to position
        self.custom_param_fig = Figure(figsize=(5,3.1))
        self.custom_param_subplot = self.custom_param_fig.add_subplot(111)
        # Prevent coordinate values from appearing in the toolbar; this would sometimes jostle GUI elements around
        self.custom_param_subplot.format_coord = lambda x, y: ""
        self.custom_param_canvas = tkagg.FigureCanvasTkAgg(self.custom_param_fig, 
                                                           master=self.param_rules_frame)
        self.custom_param_canvas.get_tk_widget().grid(row=12, column=0, columnspan=2)

        self.custom_param_toolbar_frame = tk.ttk.Frame(master=self.param_rules_frame)
        self.custom_param_toolbar_frame.grid(row=13,column=0,columnspan=2)
        tkagg.NavigationToolbar2Tk(self.custom_param_canvas, 
                                   self.custom_param_toolbar_frame)
        
        self.recent_param_fig = Figure(figsize=(5,3.1))
        self.recent_param_subplot = self.recent_param_fig.add_subplot(111)
        self.recent_param_subplot.format_coord = lambda x, y: ""
        self.recent_param_canvas = tkagg.FigureCanvasTkAgg(self.recent_param_fig, 
                                                           master=self.param_rules_frame)
        self.recent_param_canvas.get_tk_widget().grid(row=12,column=2,columnspan=2)

        self.recent_param_toolbar_frame = tk.ttk.Frame(master=self.param_rules_frame)
        self.recent_param_toolbar_frame.grid(row=13,column=2,columnspan=2)
        tkagg.NavigationToolbar2Tk(self.recent_param_canvas, 
                                   self.recent_param_toolbar_frame)

        tk.ttk.Button(self.param_rules_frame, text="⇧", 
                      command=self.moveup_paramrule).grid(row=1,column=4)

        tk.ttk.OptionMenu(self.param_rules_frame, 
                          self.paramtoolkit_viewer_selection, 
                          unitless_dropdown_list[0], 
                          *unitless_dropdown_list).grid(row=2,column=4)

        tk.ttk.Button(self.param_rules_frame, 
                      text="Change view", 
                      command=self.refresh_paramrule_listbox).grid(row=2,column=5)

        tk.ttk.Button(self.param_rules_frame, text="⇩", 
                      command=self.movedown_paramrule).grid(row=3,column=4)

        ## Param List Upload:

        self.listupload_frame = tk.ttk.Frame(self.tab_explicit_init)
        self.listupload_frame.grid(row=0,column=0,padx=(440,0))

        tk.Message(self.listupload_frame, 
                   text="This tab provides an option "
                   "to directly import a list of data points, "
                   "on which the TED will do linear interpolation "
                   "to fit to the specified space grid.", 
                   width=360).grid(row=0,column=0)
        
        tk.ttk.OptionMenu(self.listupload_frame, 
                          self.listupload_var_selection, 
                          unitless_dropdown_list[0], 
                          *unitless_dropdown_list).grid(row=1,column=0)

        tk.ttk.Button(self.listupload_frame, 
                      text="Import", 
                      command=self.add_listupload).grid(row=2,column=0)
        
        self.listupload_fig = Figure(figsize=(6,3.8))
        self.listupload_subplot = self.listupload_fig.add_subplot(111)
        self.listupload_canvas = tkagg.FigureCanvasTkAgg(self.listupload_fig, 
                                                         master=self.listupload_frame)
        self.listupload_canvas.get_tk_widget().grid(row=0, rowspan=3,column=1)
        
        self.listupload_toolbar_frame = tk.ttk.Frame(master=self.listupload_frame)
        self.listupload_toolbar_frame.grid(row=3,column=1)
        tkagg.NavigationToolbar2Tk(self.listupload_canvas, 
                                   self.listupload_toolbar_frame)

        ## Laser Generation Condition (LGC): extra input mtds for nanowire-specific applications
        if self.module.system_ID in self.LGC_eligible_modules:
            self.create_LGC_frame()
            self.tab_inputs.add(self.tab_generation_init, 
                                text="Laser Generation Conditions")
            
        # Attach sub-frames to input tab and input tab to overall notebook
        self.tab_inputs.add(self.tab_rules_init, text="Parameter Toolkit")
        self.tab_inputs.add(self.tab_explicit_init, text="Parameter List Upload")
        self.notebook.add(self.tab_inputs, text="Inputs")
        return
    
    def create_LGC_frame(self):
        """ Create a special frame for collecting laser initial contiions"""
        self.check_calculate_init_material_expfactor = tk.IntVar()
        self.LGC_layer = tk.StringVar()
        self.LGC_stim_mode = tk.StringVar()
        self.LGC_gen_power_mode = tk.StringVar()
        self.LGC_direction = tk.StringVar()
        
        self.LGC_frame = tk.ttk.Frame(self.tab_generation_init)
        self.LGC_frame.grid(row=0,column=0, padx=(360,0))

        tk.ttk.Label(self.LGC_frame, 
                     text="Generation from Laser Excitation", 
                     style="Header.TLabel").grid(row=0,column=0,columnspan=3)
        

        # A sub-frame attached to a sub-frame
        # With these we can group related elements into a common region
        self.material_param_frame = tk.Frame(self.LGC_frame, 
                                             highlightbackground="black", 
                                             highlightthickness=1)
        self.material_param_frame.grid(row=1,column=0)

        tk.Label(self.material_param_frame, 
                 text="Material Params - Select One").grid(row=0,column=0,columnspan=4)

        tk.ttk.Separator(self.material_param_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        tk.ttk.Radiobutton(self.material_param_frame, 
                           variable=self.check_calculate_init_material_expfactor, 
                           value=1).grid(row=2,column=0)

        tk.Label(self.material_param_frame, 
                 text="Option 1").grid(row=2,column=1)

        tk.Label(self.material_param_frame, text="A0 [cm^-1 eV^-γ]").grid(row=2,column=2)

        self.A0_entry = tk.ttk.Entry(self.material_param_frame, width=9)
        self.A0_entry.grid(row=2,column=3)

        tk.Label(self.material_param_frame, text="Eg [eV]").grid(row=3,column=2)

        self.Eg_entry = tk.ttk.Entry(self.material_param_frame, width=9)
        self.Eg_entry.grid(row=3,column=3)

        tk.ttk.Radiobutton(self.material_param_frame, 
                           variable=self.LGC_stim_mode, 
                           value="direct").grid(row=4,column=2)

        tk.Label(self.material_param_frame,
                 text="Direct (γ=1/2)").grid(row=4,column=3)

        tk.ttk.Radiobutton(self.material_param_frame, 
                           variable=self.LGC_stim_mode, 
                           value="indirect").grid(row=5,column=2)

        tk.Label(self.material_param_frame,
                 text="Indirect (γ=2)").grid(row=5,column=3)

        tk.ttk.Separator(self.material_param_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=6,column=0,columnspan=30, pady=(5,5), sticky="ew")

        tk.ttk.Radiobutton(self.material_param_frame, 
                           variable=self.check_calculate_init_material_expfactor, 
                           value=0).grid(row=7,column=0)

        tk.Label(self.material_param_frame, 
                 text="Option 2").grid(row=7,column=1)

        tk.Label(self.material_param_frame, 
                 text="α [cm^-1]").grid(row=8,column=2)

        self.LGC_absorption_cof_entry = tk.ttk.Entry(self.material_param_frame, 
                                                     width=9)
        self.LGC_absorption_cof_entry.grid(row=8,column=3)

        self.pulse_laser_frame = tk.Frame(self.LGC_frame, 
                                          highlightbackground="black", 
                                          highlightthickness=1)
        self.pulse_laser_frame.grid(row=1,column=1, padx=(20,0))

        tk.Label(self.pulse_laser_frame, 
                 text="Pulse Laser Params").grid(row=0,column=0,columnspan=4)

        tk.ttk.Separator(self.pulse_laser_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        tk.Label(self.pulse_laser_frame, 
                 text="Pulse frequency [kHz]").grid(row=2,column=2)

        self.pulse_freq_entry = tk.ttk.Entry(self.pulse_laser_frame, width=9)
        self.pulse_freq_entry.grid(row=2,column=3)

        tk.Label(self.pulse_laser_frame, 
                 text="Wavelength [nm]").grid(row=3,column=2)

        self.pulse_wavelength_entry = tk.ttk.Entry(self.pulse_laser_frame, width=9)
        self.pulse_wavelength_entry.grid(row=3,column=3)

        self.gen_power_param_frame = tk.Frame(self.LGC_frame, 
                                              highlightbackground="black",
                                              highlightthickness=1)
        self.gen_power_param_frame.grid(row=1,column=2, padx=(20,0))

        tk.Label(self.gen_power_param_frame, 
                 text="Generation/Power Params - Select One").grid(row=0,column=0,columnspan=4)

        tk.ttk.Separator(self.gen_power_param_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

        tk.ttk.Radiobutton(self.gen_power_param_frame, 
                           variable=self.LGC_gen_power_mode, 
                           value="power-spot").grid(row=2,column=0)

        tk.Label(self.gen_power_param_frame, text="Option 1").grid(row=2,column=1)

        tk.Label(self.gen_power_param_frame, text="Power [uW]").grid(row=2,column=2)

        self.power_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.power_entry.grid(row=2,column=3)

        tk.Label(self.gen_power_param_frame, text="Spot size [cm^2]").grid(row=3,column=2)

        self.spotsize_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.spotsize_entry.grid(row=3,column=3)

        tk.ttk.Separator(self.gen_power_param_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=4,column=0,columnspan=30, pady=(5,5), sticky="ew")

        tk.ttk.Radiobutton(self.gen_power_param_frame, 
                           variable=self.LGC_gen_power_mode, 
                           value="density").grid(row=5,column=0)

        tk.Label(self.gen_power_param_frame,text="Option 2").grid(row=5,column=1)

        tk.Label(self.gen_power_param_frame, 
                 text="Power Density [uW/cm^2]").grid(row=5,column=2)

        self.power_density_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.power_density_entry.grid(row=5,column=3)

        tk.ttk.Separator(self.gen_power_param_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=6,column=0,columnspan=30, pady=(5,5), sticky="ew")

        tk.ttk.Radiobutton(self.gen_power_param_frame, 
                           variable=self.LGC_gen_power_mode, 
                           value="max-gen").grid(row=7,column=0)

        tk.Label(self.gen_power_param_frame, text="Option 3").grid(row=7,column=1)

        tk.Label(self.gen_power_param_frame, 
                 text="Max Generation [carr/cm^3]").grid(row=7,column=2)

        self.max_gen_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.max_gen_entry.grid(row=7,column=3)

        tk.ttk.Separator(self.gen_power_param_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=8,column=0,columnspan=30, pady=(5,5), sticky="ew")

        tk.ttk.Radiobutton(self.gen_power_param_frame, 
                           variable=self.LGC_gen_power_mode, 
                           value="total-gen").grid(row=9,column=0)

        tk.Label(self.gen_power_param_frame, text="Option 4").grid(row=9,column=1)

        tk.Label(self.gen_power_param_frame, 
                 text="Average Generation [carr/cm^3]").grid(row=9,column=2)

        self.total_gen_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.total_gen_entry.grid(row=9,column=3)
        
        self.LGC_layer_frame = tk.ttk.Frame(self.LGC_frame)
        self.LGC_layer_frame.grid(row=2,column=1,padx=(20,0))
        
        LGC_eligible_layers = [layer_name for layer_name in self.module.layers
                               if "delta_N" in self.module.layers[layer_name].params
                               and "delta_P" in self.module.layers[layer_name].params]
        
        for layer_name in self.module.layers:
            self.using_LGC[layer_name] = False
            
        for layer_name in LGC_eligible_layers:
            self.LGC_options[layer_name] = {}
            self.LGC_values[layer_name] = {}
            
        
        self.LGC_layer_rbtns = {}
        self.LGC_layer_frame_title = tk.Label(self.LGC_layer_frame, text="Apply to layer: ")
        self.LGC_layer_frame_title.grid(row=0,column=0,columnspan=2)
        for i, layer_name in enumerate(LGC_eligible_layers):
            self.LGC_layer_rbtns[layer_name] = tk.ttk.Radiobutton(self.LGC_layer_frame, 
                                                      variable=self.LGC_layer, 
                                                      value=layer_name)
            self.LGC_layer_rbtns[layer_name].grid(row=i+1,column=0)
            layer_rbtn_label = tk.ttk.Label(self.LGC_layer_frame, text=layer_name)
            layer_rbtn_label.grid(row=i+1,column=1)

        self.LGC_direction_frame = tk.Frame(self.LGC_frame)
        self.LGC_direction_frame.grid(row=3,column=0,columnspan=3)
        
        tk.ttk.Radiobutton(self.LGC_direction_frame, variable=self.LGC_direction, 
                           value="fwd").grid(row=0,column=0)
        tk.Label(self.LGC_direction_frame, text="Forward").grid(row=0,column=1)
        
        tk.ttk.Radiobutton(self.LGC_direction_frame, variable=self.LGC_direction, 
                           value="reverse").grid(row=1,column=0)
        tk.Label(self.LGC_direction_frame, text="Reverse").grid(row=1,column=1)
        
        
        tk.ttk.Button(self.LGC_frame, 
                      text="Generate Initial Condition", 
                      command=self.add_LGC).grid(row=4,column=0,columnspan=3)

        tk.Message(self.LGC_frame, 
                   text="The Laser Generation Condition "
                   "uses the above numerical parameters "
                   "to generate an initial carrier "
                   "distribution based on an applied "
                   "laser excitation.", width=320).grid(row=5,column=0,columnspan=3)
        
        self.LGC_fig = Figure(figsize=(5,3.1))
        self.LGC_subplot = self.LGC_fig.add_subplot(111)
        self.LGC_canvas = tkagg.FigureCanvasTkAgg(self.LGC_fig, master=self.LGC_frame)
        self.LGC_canvas.get_tk_widget().grid(row=6, column=0, columnspan=3)
        
        self.LGC_toolbar_frame = tk.ttk.Frame(master=self.LGC_frame)
        self.LGC_toolbar_frame.grid(row=7,column=0,columnspan=3)
        tkagg.NavigationToolbar2Tk(self.LGC_canvas, 
                                   self.LGC_toolbar_frame)
        
        ## TODO: Assign these directly
        self.LGC_entryboxes_dict = {"A0":self.A0_entry, "Eg":self.Eg_entry, 
                                    "LGC_absorption_cof":self.LGC_absorption_cof_entry, 
                                    "Pulse_Freq":self.pulse_freq_entry, 
                                    "Pulse_Wavelength":self.pulse_wavelength_entry, 
                                    "Power":self.power_entry, 
                                    "Spotsize":self.spotsize_entry, 
                                    "Power_Density":self.power_density_entry,
                                    "Max_Gen":self.max_gen_entry, 
                                    "Total_Gen":self.total_gen_entry}
        self.enter(self.LGC_entryboxes_dict["A0"], "1240")
        self.LGC_optionboxes = {"long_expfactor":self.check_calculate_init_material_expfactor, 
                                "incidence":self.LGC_stim_mode,
                                "power_mode":self.LGC_gen_power_mode,
                                "direction":self.LGC_direction}
        return
    
    def add_tab_simulate(self):
        self.tab_simulate = tk.ttk.Frame(self.notebook)

        tk.ttk.Label(self.tab_simulate, 
                     text="Select Init. Cond.", 
                     style="Header.TLabel").grid(row=0,column=0,columnspan=2, padx=(9,12))

        tk.ttk.Label(self.tab_simulate, text="Simulation Time [ns]").grid(row=2,column=0)

        self.simtime_entry = tk.ttk.Entry(self.tab_simulate, width=9)
        self.simtime_entry.grid(row=2,column=1)

        tk.ttk.Label(self.tab_simulate, text="dt [ns]").grid(row=3,column=0)

        self.dt_entry = tk.ttk.Entry(self.tab_simulate, width=9)
        self.dt_entry.grid(row=3,column=1)
        
        tk.ttk.Label(self.tab_simulate, text="Max solver stepsize [ns]").grid(row=4,column=0)
        
        self.hmax_entry = tk.ttk.Entry(self.tab_simulate, width=9)
        self.hmax_entry.grid(row=4,column=1)

        self.enter(self.dt_entry, "0.5")
        self.enter(self.hmax_entry, "0.25")
        
        tk.ttk.Button(self.tab_simulate, text="Start Simulation(s)", 
                      command=self.do_Batch).grid(row=6,column=0,columnspan=2,padx=(9,12))

        tk.ttk.Label(self.tab_simulate, text="Status").grid(row=7, column=0, columnspan=2)

        self.status = tk.Text(self.tab_simulate, width=28,height=4)
        self.status.grid(row=8, rowspan=2, column=0, columnspan=2)
        self.status.configure(state='disabled')

        tk.ttk.Separator(self.tab_simulate, orient="vertical", 
                         style="Grey Bar.TSeparator").grid(row=0,rowspan=30,column=2,sticky="ns")

        tk.ttk.Label(self.tab_simulate, 
                     text="Simulation - {}".format(self.module.system_ID)).grid(row=0,column=3,columnspan=3)
        
        self.sim_fig = Figure(figsize=(14, 8))
        count = 1
        cdim = np.ceil(np.sqrt(self.module.count_s_outputs()))
        
        rdim = np.ceil(self.module.count_s_outputs() / cdim)
        self.sim_subplots = {}
        for layer_name, layer in self.module.layers.items():
            self.sim_subplots[layer_name] = {}
            for variable in layer.s_outputs:
                self.sim_subplots[layer_name][variable] = self.sim_fig.add_subplot(int(rdim), 
                                                                                   int(cdim), 
                                                                                   int(count))
                self.sim_subplots[layer_name][variable].set_title(variable)
                count += 1

        self.sim_canvas = tkagg.FigureCanvasTkAgg(self.sim_fig, master=self.tab_simulate)
        self.sim_canvas.get_tk_widget().grid(row=1,column=3,rowspan=12,columnspan=2)
        
        self.simfig_toolbar_frame = tk.ttk.Frame(master=self.tab_simulate)
        self.simfig_toolbar_frame.grid(row=13,column=3,columnspan=2)
        tkagg.NavigationToolbar2Tk(self.sim_canvas, self.simfig_toolbar_frame)

        self.notebook.add(self.tab_simulate, text="Simulate")
        return
    
    def add_tab_analyze(self):
        self.tab_analyze = tk.ttk.Notebook(self.notebook)
        self.tab_overview_analysis = tk.ttk.Frame(self.tab_analyze)
        self.tab_detailed_analysis = tk.ttk.Frame(self.tab_analyze)
        
        self.analyze_overview_fig = Figure(figsize=(21,8))
        self.overview_subplots = {}
        count = 1
        total_outputs_count = sum([self.module.layers[layer].outputs_count for layer in self.module.layers])
        rdim = np.floor(np.sqrt(total_outputs_count))
        cdim = np.ceil(total_outputs_count / rdim)
        
        all_outputs = []
        for layer_name in self.module.layers:
            self.overview_subplots[layer_name] = {}
            for output in self.module.layers[layer_name].outputs:
                self.overview_subplots[layer_name][output] = self.analyze_overview_fig.add_subplot(int(rdim), int(cdim), int(count))
                all_outputs.append("{}: {}".format(layer_name, output))
                count += 1
                    
        self.overview_setup_frame = tk.Frame(self.tab_overview_analysis)
        self.overview_setup_frame.grid(row=0,column=0, padx=(20,20))
        
        tk.Label(self.overview_setup_frame, text="No. samples").grid(row=0,column=0)
        
        self.overview_samplect_entry = tk.ttk.Entry(self.overview_setup_frame, width=8)
        self.overview_samplect_entry.grid(row=1,column=0)
        self.enter(self.overview_samplect_entry, "6")
        
        self.overview_sample_mode = tk.StringVar()
        self.overview_sample_mode.set("Log")
        tk.ttk.Radiobutton(self.overview_setup_frame, variable=self.overview_sample_mode,
                           value="Linear").grid(row=0,column=1)
        tk.Label(self.overview_setup_frame, text="Linear").grid(row=0,column=2)
        
        tk.ttk.Radiobutton(self.overview_setup_frame, variable=self.overview_sample_mode,
                           value="Log").grid(row=1,column=1)
        tk.Label(self.overview_setup_frame, text="Log").grid(row=1,column=2)
        
        tk.ttk.Radiobutton(self.overview_setup_frame, variable=self.overview_sample_mode,
                           value="Custom").grid(row=2,column=1)
        tk.Label(self.overview_setup_frame, text="Custom").grid(row=2,column=2)
        
        tk.Button(master=self.overview_setup_frame, text="Select Dataset", 
                      command=self.plot_overview_analysis).grid(row=0,rowspan=2,column=3)
        
        self.overview_var_selection = tk.StringVar()
        
        tk.ttk.OptionMenu(self.tab_overview_analysis, self.overview_var_selection, 
                          all_outputs[0], *all_outputs).grid(row=0,column=1)
        
        tk.ttk.Button(master=self.tab_overview_analysis, text="Export", 
                      command=self.export_overview).grid(row=0,column=2)
        
        
        self.analyze_overview_canvas = tkagg.FigureCanvasTkAgg(self.analyze_overview_fig, 
                                                               master=self.tab_overview_analysis)
        self.analyze_overview_canvas.get_tk_widget().grid(row=1,column=0,columnspan=99)

        self.overview_toolbar_frame = tk.ttk.Frame(self.tab_overview_analysis)
        self.overview_toolbar_frame.grid(row=2,column=0,columnspan=99)
        
        tkagg.NavigationToolbar2Tk(self.analyze_overview_canvas, 
                                   self.overview_toolbar_frame).grid(row=0,column=0)
        
        tk.ttk.Label(self.tab_detailed_analysis, 
                     text="Plot and Integrate Saved Datasets", 
                     style="Header.TLabel").grid(row=0,column=0,columnspan=8)
        
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
        
        self.analyze_canvas = tkagg.FigureCanvasTkAgg(self.analyze_fig, 
                                                      master=self.tab_detailed_analysis)
        self.analyze_canvas.get_tk_widget().grid(row=1,column=0,rowspan=1,columnspan=4, padx=(12,0))

        self.analyze_plotselector_frame = tk.ttk.Frame(master=self.tab_detailed_analysis)
        self.analyze_plotselector_frame.grid(row=2,rowspan=2,column=0,columnspan=4)
        
        tk.ttk.Radiobutton(self.analyze_plotselector_frame, 
                           variable=self.active_analysisplot_ID, 
                           value=0).grid(row=0,column=0)

        tk.ttk.Label(self.analyze_plotselector_frame, 
                     text="Use: Top Left").grid(row=0,column=1)
        
        tk.ttk.Radiobutton(self.analyze_plotselector_frame, 
                           variable=self.active_analysisplot_ID, 
                           value=1).grid(row=0,column=2)

        tk.ttk.Label(self.analyze_plotselector_frame, 
                     text="Use: Top Right").grid(row=0,column=3)
        
        tk.ttk.Radiobutton(self.analyze_plotselector_frame, 
                           variable=self.active_analysisplot_ID, 
                           value=2).grid(row=1,column=0)

        tk.ttk.Label(self.analyze_plotselector_frame, 
                     text="Use: Bottom Left").grid(row=1,column=1)
        
        tk.ttk.Radiobutton(self.analyze_plotselector_frame, 
                           variable=self.active_analysisplot_ID, 
                           value=3).grid(row=1,column=2)

        tk.ttk.Label(self.analyze_plotselector_frame, 
                     text="Use: Bottom Right").grid(row=1,column=3)
        
        self.analyze_toolbar_frame = tk.ttk.Frame(master=self.tab_detailed_analysis)
        self.analyze_toolbar_frame.grid(row=4,column=0,rowspan=4,columnspan=4)
        tkagg.NavigationToolbar2Tk(self.analyze_canvas, self.analyze_toolbar_frame).grid(row=0,column=0,columnspan=7)

        tk.ttk.Button(self.analyze_toolbar_frame, 
                      text="Plot", 
                      command=partial(self.load_datasets)).grid(row=1,column=0)
        
        self.analyze_tstep_entry = tk.ttk.Entry(self.analyze_toolbar_frame, width=9)
        self.analyze_tstep_entry.grid(row=1,column=1)

        tk.ttk.Button(self.analyze_toolbar_frame, 
                      text="Time >>", 
                      command=partial(self.plot_tstep)).grid(row=1,column=2)

        tk.ttk.Button(self.analyze_toolbar_frame, 
                      text=">> Integrate <<", 
                      command=partial(self.do_Integrate)).grid(row=1,column=3)

        tk.ttk.Button(self.analyze_toolbar_frame, 
                      text="Axis Settings", 
                      command=partial(self.do_change_axis_popup, 
                                      from_integration=0)).grid(row=1,column=4)

        tk.ttk.Button(self.analyze_toolbar_frame, 
                      text="Export", 
                      command=partial(self.export_plot, 
                                      from_integration=0)).grid(row=1,column=5)

        tk.ttk.Button(self.analyze_toolbar_frame, 
                      text="Generate IC", 
                      command=partial(self.do_IC_carry_popup)).grid(row=1,column=6)

        self.integration_fig = Figure(figsize=(9,5))
        self.integration_subplot = self.integration_fig.add_subplot(111)
        self.integration_plots[0].plot_obj = self.integration_subplot

        self.integration_canvas = tkagg.FigureCanvasTkAgg(self.integration_fig, 
                                                          master=self.tab_detailed_analysis)
        self.integration_canvas.get_tk_widget().grid(row=1,column=5,rowspan=1,columnspan=1, padx=(20,0))

        self.integration_toolbar_frame = tk.ttk.Frame(master=self.tab_detailed_analysis)
        self.integration_toolbar_frame.grid(row=3,column=5, rowspan=2,columnspan=1)
        tkagg.NavigationToolbar2Tk(self.integration_canvas, 
                                   self.integration_toolbar_frame).grid(row=0,column=0,columnspan=5)

        tk.ttk.Button(self.integration_toolbar_frame, 
                      text="Axis Settings", 
                      command=partial(self.do_change_axis_popup, 
                                      from_integration=1)).grid(row=1,column=0)

        tk.ttk.Button(self.integration_toolbar_frame, 
                      text="Export", 
                      command=partial(self.export_plot, 
                                      from_integration=1)).grid(row=1,column=1)

        # self.integration_bayesim_button = tk.ttk.Button(self.integration_toolbar_frame, text="Bayesim", command=partial(self.do_bayesim_popup))
        # self.integration_bayesim_button.grid(row=1,column=2)

        self.analysis_status = tk.Text(self.tab_detailed_analysis, width=28,height=3)
        self.analysis_status.grid(row=5,rowspan=3,column=5,columnspan=1)
        self.analysis_status.configure(state="disabled")

        self.tab_analyze.add(self.tab_overview_analysis, text="Overview")
        self.tab_analyze.add(self.tab_detailed_analysis, text="Detailed Analysis")
        self.notebook.add(self.tab_analyze, text="Analyze")
        return
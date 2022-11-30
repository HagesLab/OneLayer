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
        """Setup main canvas of Notebook."""

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

        self.default_dirs = {"Initial":"Initial", "Data":"Data", "PL":"Analysis"}


    def prepare_radiobuttons_and_checkboxes(self):
        """Tkinter checkboxes and radiobuttons require special variables 
        to extract user input.
        IntVars or BooleanVars are sufficient for binary choices 
        e.g. whether a checkbox is checked
        while StringVars are more suitable for open-ended choices 
        e.g. selecting one mode from a list."""

        self.check_reset_params = tk.IntVar()
        self.check_reset_inits = tk.IntVar()
        self.check_display_legend = tk.IntVar()
        self.check_freeze_axes = tk.IntVar()
        self.check_autointegrate = tk.IntVar(value=0)
        
        self.active_analysisplot_ID = tk.IntVar()
        self.active_integrationplot_ID = tk.IntVar()

       

    def prepare_initial_things(self):
        """Prepare inital empty variables and dictionaries"""
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
        self.ICregen_include_flags = {}
        
            
        # Helpers, flags, and containers for analysis plots
        self.analysis_plots = [
            Analysis_Plot_State(),
            Analysis_Plot_State(), 
            Analysis_Plot_State(),
            Analysis_Plot_State()]
        self.integration_plots = [Integration_Plot_State()]
        self.data_var = tk.StringVar()
        self.fetch_PLmode = tk.StringVar()
        self.fetch_intg_mode = tk.StringVar()
        self.yaxis_type = tk.StringVar()
        self.xaxis_type = tk.StringVar()

        # Track which timeseries popups are open
        self.active_timeseries = {}


    def prepare_eligible_modules(self):
        # Default LGC values
        self.using_LGC = {layer:False for layer in self.module.layers}
        self.LGC_options = {layer:{} for layer in self.module.layers}
        self.LGC_values = {layer:{} for layer in self.module.layers}


    def reset_popup_flags(self):
        """Flags to record when popup menus open and close."""
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
        self.IC_regen_popup_isopen = False
        self.bayesim_popup_isopen = False
    # Create GUI elements for each tab
    # Tkinter works a bit like a bulletin board:
    # we declare an overall frame and
    # pin things to it at specified locations
    # This includes other frames, which is evident
    # in how the tab_inputs has three sub-tabs pinned to itself
    
    


    def create_LGC_frame(self):
        """Method generating the LGC input frame."""
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
        
        
        
        tk.ttk.Separator(self.gen_power_param_frame, 
                         orient="horizontal", 
                         style="Grey Bar.TSeparator").grid(row=10,column=0,columnspan=30, pady=(5,5), sticky="ew")

        tk.ttk.Radiobutton(self.gen_power_param_frame, 
                           variable=self.LGC_gen_power_mode, 
                           value="fluence").grid(row=11,column=0)
        
        tk.Label(self.gen_power_param_frame, text="Option 5").grid(row=11,column=1)

        tk.Label(self.gen_power_param_frame, 
                 text="Fluence [phot/cm^2 pulse]").grid(row=11,column=2)

        self.fluence_entry = tk.ttk.Entry(self.gen_power_param_frame, width=9)
        self.fluence_entry.grid(row=11,column=3)

        
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
        self.LGC_layer_frame_title.grid(row=0,column=0,columnspan=99)
        for i, layer_name in enumerate(LGC_eligible_layers):
            self.LGC_layer_rbtns[layer_name] = tk.ttk.Radiobutton(self.LGC_layer_frame, 
                                                      variable=self.LGC_layer, 
                                                      value=layer_name)
            self.LGC_layer_rbtns[layer_name].grid(row=2,column=i, padx=(10,10))
            layer_rbtn_label = tk.ttk.Label(self.LGC_layer_frame, text=layer_name)
            layer_rbtn_label.grid(row=1,column=i, padx=(10,10))

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
                                    "Total_Gen":self.total_gen_entry,
                                    "Fluence":self.fluence_entry}
        self.enter(self.LGC_entryboxes_dict["A0"], "1240")
        self.LGC_optionboxes = {"long_expfactor":self.check_calculate_init_material_expfactor, 
                                "incidence":self.LGC_stim_mode,
                                "power_mode":self.LGC_gen_power_mode,
                                "direction":self.LGC_direction}

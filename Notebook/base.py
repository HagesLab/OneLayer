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

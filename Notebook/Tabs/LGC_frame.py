# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:21:31 2022

@author: cfai2
"""
import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk

def create_LGC_frame(nb):
    """Method generating the LGC input frame for a Notebook()."""
    nb.check_calculate_init_material_expfactor = tk.IntVar()
    nb.LGC_layer = tk.StringVar()
    nb.LGC_stim_mode = tk.StringVar()
    nb.LGC_gen_power_mode = tk.StringVar()
    nb.LGC_direction = tk.StringVar()
    
    nb.LGC_frame = tk.ttk.Frame(nb.tab_generation_init)
    nb.LGC_frame.grid(row=0,column=0, padx=(360,0))

    tk.ttk.Label(nb.LGC_frame, 
                 text="Generation from Laser Excitation", 
                 style="Header.TLabel").grid(row=0,column=0,columnspan=3)
    

    # A sub-frame attached to a sub-frame
    # With these we can group related elements into a common region
    nb.material_param_frame = tk.Frame(nb.LGC_frame, 
                                       highlightbackground="black", 
                                       highlightthickness=1)
    nb.material_param_frame.grid(row=1,column=0)

    tk.Label(nb.material_param_frame, 
             text="Material Params - Select One").grid(row=0,column=0,columnspan=4)

    tk.ttk.Separator(nb.material_param_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

    tk.ttk.Radiobutton(nb.material_param_frame, 
                       variable=nb.check_calculate_init_material_expfactor, 
                       value=1).grid(row=2,column=0)

    tk.Label(nb.material_param_frame, 
             text="Option 1").grid(row=2,column=1)

    tk.Label(nb.material_param_frame, text="A0 [cm^-1 eV^-γ]").grid(row=2,column=2)

    nb.A0_entry = tk.ttk.Entry(nb.material_param_frame, width=9)
    nb.A0_entry.grid(row=2,column=3)

    tk.Label(nb.material_param_frame, text="Eg [eV]").grid(row=3,column=2)

    nb.Eg_entry = tk.ttk.Entry(nb.material_param_frame, width=9)
    nb.Eg_entry.grid(row=3,column=3)

    tk.ttk.Radiobutton(nb.material_param_frame, 
                       variable=nb.LGC_stim_mode, 
                       value="direct").grid(row=4,column=2)

    tk.Label(nb.material_param_frame,
             text="Direct (γ=1/2)").grid(row=4,column=3)

    tk.ttk.Radiobutton(nb.material_param_frame, 
                       variable=nb.LGC_stim_mode, 
                       value="indirect").grid(row=5,column=2)

    tk.Label(nb.material_param_frame,
             text="Indirect (γ=2)").grid(row=5,column=3)

    tk.ttk.Separator(nb.material_param_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=6,column=0,columnspan=30, pady=(5,5), sticky="ew")

    tk.ttk.Radiobutton(nb.material_param_frame, 
                       variable=nb.check_calculate_init_material_expfactor, 
                       value=0).grid(row=7,column=0)

    tk.Label(nb.material_param_frame, 
             text="Option 2").grid(row=7,column=1)

    tk.Label(nb.material_param_frame, 
             text="α [cm^-1]").grid(row=8,column=2)

    nb.LGC_absorption_cof_entry = tk.ttk.Entry(nb.material_param_frame, 
                                                 width=9)
    nb.LGC_absorption_cof_entry.grid(row=8,column=3)

    nb.pulse_laser_frame = tk.Frame(nb.LGC_frame, 
                                      highlightbackground="black", 
                                      highlightthickness=1)
    nb.pulse_laser_frame.grid(row=1,column=1, padx=(20,0))

    tk.Label(nb.pulse_laser_frame, 
             text="Pulse Laser Params").grid(row=0,column=0,columnspan=4)

    tk.ttk.Separator(nb.pulse_laser_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

    tk.Label(nb.pulse_laser_frame, 
             text="Pulse frequency [kHz]").grid(row=2,column=2)

    nb.pulse_freq_entry = tk.ttk.Entry(nb.pulse_laser_frame, width=9)
    nb.pulse_freq_entry.grid(row=2,column=3)

    tk.Label(nb.pulse_laser_frame, 
             text="Wavelength [nm]").grid(row=3,column=2)

    nb.pulse_wavelength_entry = tk.ttk.Entry(nb.pulse_laser_frame, width=9)
    nb.pulse_wavelength_entry.grid(row=3,column=3)

    nb.gen_power_param_frame = tk.Frame(nb.LGC_frame, 
                                          highlightbackground="black",
                                          highlightthickness=1)
    nb.gen_power_param_frame.grid(row=1,column=2, padx=(20,0))

    tk.Label(nb.gen_power_param_frame, 
             text="Generation/Power Params - Select One").grid(row=0,column=0,columnspan=4)

    tk.ttk.Separator(nb.gen_power_param_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=1,column=0,columnspan=30, pady=(10,10), sticky="ew")

    tk.ttk.Radiobutton(nb.gen_power_param_frame, 
                       variable=nb.LGC_gen_power_mode, 
                       value="power-spot").grid(row=2,column=0)

    tk.Label(nb.gen_power_param_frame, text="Option 1").grid(row=2,column=1)

    tk.Label(nb.gen_power_param_frame, text="Power [uW]").grid(row=2,column=2)

    nb.power_entry = tk.ttk.Entry(nb.gen_power_param_frame, width=9)
    nb.power_entry.grid(row=2,column=3)

    tk.Label(nb.gen_power_param_frame, text="Spot size [cm^2]").grid(row=3,column=2)

    nb.spotsize_entry = tk.ttk.Entry(nb.gen_power_param_frame, width=9)
    nb.spotsize_entry.grid(row=3,column=3)

    tk.ttk.Separator(nb.gen_power_param_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=4,column=0,columnspan=30, pady=(5,5), sticky="ew")

    tk.ttk.Radiobutton(nb.gen_power_param_frame, 
                       variable=nb.LGC_gen_power_mode, 
                       value="density").grid(row=5,column=0)

    tk.Label(nb.gen_power_param_frame,text="Option 2").grid(row=5,column=1)

    tk.Label(nb.gen_power_param_frame, 
             text="Power Density [uW/cm^2]").grid(row=5,column=2)

    nb.power_density_entry = tk.ttk.Entry(nb.gen_power_param_frame, width=9)
    nb.power_density_entry.grid(row=5,column=3)

    tk.ttk.Separator(nb.gen_power_param_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=6,column=0,columnspan=30, pady=(5,5), sticky="ew")

    tk.ttk.Radiobutton(nb.gen_power_param_frame, 
                       variable=nb.LGC_gen_power_mode, 
                       value="max-gen").grid(row=7,column=0)

    tk.Label(nb.gen_power_param_frame, text="Option 3").grid(row=7,column=1)

    tk.Label(nb.gen_power_param_frame, 
             text="Max Generation [carr/cm^3]").grid(row=7,column=2)

    nb.max_gen_entry = tk.ttk.Entry(nb.gen_power_param_frame, width=9)
    nb.max_gen_entry.grid(row=7,column=3)

    tk.ttk.Separator(nb.gen_power_param_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=8,column=0,columnspan=30, pady=(5,5), sticky="ew")

    tk.ttk.Radiobutton(nb.gen_power_param_frame, 
                       variable=nb.LGC_gen_power_mode, 
                       value="total-gen").grid(row=9,column=0)

    tk.Label(nb.gen_power_param_frame, text="Option 4").grid(row=9,column=1)

    tk.Label(nb.gen_power_param_frame, 
             text="Average Generation [carr/cm^3]").grid(row=9,column=2)

    nb.total_gen_entry = tk.ttk.Entry(nb.gen_power_param_frame, width=9)
    nb.total_gen_entry.grid(row=9,column=3)
    
    
    
    tk.ttk.Separator(nb.gen_power_param_frame, 
                     orient="horizontal", 
                     style="Grey Bar.TSeparator").grid(row=10,column=0,columnspan=30, pady=(5,5), sticky="ew")

    tk.ttk.Radiobutton(nb.gen_power_param_frame, 
                       variable=nb.LGC_gen_power_mode, 
                       value="fluence").grid(row=11,column=0)
    
    tk.Label(nb.gen_power_param_frame, text="Option 5").grid(row=11,column=1)

    tk.Label(nb.gen_power_param_frame, 
             text="Fluence [phot/cm^2 pulse]").grid(row=11,column=2)

    nb.fluence_entry = tk.ttk.Entry(nb.gen_power_param_frame, width=9)
    nb.fluence_entry.grid(row=11,column=3)

    
    nb.LGC_layer_frame = tk.ttk.Frame(nb.LGC_frame)
    nb.LGC_layer_frame.grid(row=2,column=1,padx=(20,0))
    
    LGC_eligible_layers = [layer_name for layer_name in nb.module.layers
                           if "delta_N" in nb.module.layers[layer_name].params
                           and "delta_P" in nb.module.layers[layer_name].params]
    
    # for layer_name in nb.module.layers:
    #     nb.using_LGC[layer_name] = False
        
    # for layer_name in LGC_eligible_layers:
    #     nb.LGC_options[layer_name] = {}
    #     nb.LGC_values[layer_name] = {}
        
    
    nb.LGC_layer_rbtns = {}
    nb.LGC_layer_frame_title = tk.Label(nb.LGC_layer_frame, text="Apply to layer: ")
    nb.LGC_layer_frame_title.grid(row=0,column=0,columnspan=99)
    for i, layer_name in enumerate(LGC_eligible_layers):
        nb.LGC_layer_rbtns[layer_name] = tk.ttk.Radiobutton(nb.LGC_layer_frame, 
                                                  variable=nb.LGC_layer, 
                                                  value=layer_name)
        nb.LGC_layer_rbtns[layer_name].grid(row=2,column=i, padx=(10,10))
        layer_rbtn_label = tk.ttk.Label(nb.LGC_layer_frame, text=layer_name)
        layer_rbtn_label.grid(row=1,column=i, padx=(10,10))

    nb.LGC_direction_frame = tk.Frame(nb.LGC_frame)
    nb.LGC_direction_frame.grid(row=3,column=0,columnspan=3)
    
    tk.ttk.Radiobutton(nb.LGC_direction_frame, variable=nb.LGC_direction, 
                       value="fwd").grid(row=0,column=0)
    tk.Label(nb.LGC_direction_frame, text="Forward").grid(row=0,column=1)
    
    tk.ttk.Radiobutton(nb.LGC_direction_frame, variable=nb.LGC_direction, 
                       value="reverse").grid(row=1,column=0)
    tk.Label(nb.LGC_direction_frame, text="Reverse").grid(row=1,column=1)
    
    
    tk.ttk.Button(nb.LGC_frame, 
                  text="Generate Initial Condition", 
                  command=nb.add_LGC).grid(row=4,column=0,columnspan=3)

    tk.Message(nb.LGC_frame, 
               text="The Laser Generation Condition "
               "uses the above numerical parameters "
               "to generate an initial carrier "
               "distribution based on an applied "
               "laser excitation.", width=320).grid(row=5,column=0,columnspan=3)
    
    nb.LGC_fig = Figure(figsize=(5,3.1))
    nb.LGC_subplot = nb.LGC_fig.add_subplot(111)
    nb.LGC_canvas = tkagg.FigureCanvasTkAgg(nb.LGC_fig, master=nb.LGC_frame)
    nb.LGC_canvas.get_tk_widget().grid(row=6, column=0, columnspan=3)
    
    nb.LGC_toolbar_frame = tk.ttk.Frame(master=nb.LGC_frame)
    nb.LGC_toolbar_frame.grid(row=7,column=0,columnspan=3)
    tkagg.NavigationToolbar2Tk(nb.LGC_canvas, 
                               nb.LGC_toolbar_frame)
    
    nb.LGC_entryboxes_dict = {"A0":nb.A0_entry, "Eg":nb.Eg_entry, 
                                "LGC_absorption_cof":nb.LGC_absorption_cof_entry, 
                                "Pulse_Freq":nb.pulse_freq_entry, 
                                "Pulse_Wavelength":nb.pulse_wavelength_entry, 
                                "Power":nb.power_entry, 
                                "Spotsize":nb.spotsize_entry, 
                                "Power_Density":nb.power_density_entry,
                                "Max_Gen":nb.max_gen_entry, 
                                "Total_Gen":nb.total_gen_entry,
                                "Fluence":nb.fluence_entry}
    nb.enter(nb.LGC_entryboxes_dict["A0"], "1240")
    nb.LGC_optionboxes = {"long_expfactor":nb.check_calculate_init_material_expfactor, 
                            "incidence":nb.LGC_stim_mode,
                            "power_mode":nb.LGC_gen_power_mode,
                            "direction":nb.LGC_direction}
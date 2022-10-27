# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:37:11 2022

@author: cfai2
"""

import tkinter as tk

import os
from functools import partial

from io_utils import u_read, export_ICfile
from Notebook.Popups.base_popup import Popup

class ICRegenPopup(Popup):
    
    def __init__(self, plot_ID, nb, logger):
        super().__init__(nb, grab=True)
        

        tk.ttk.Label(self.toplevel, 
                     text="Select data to include in new IC",
                     style="Header.TLabel").grid(row=0,column=0,columnspan=2)
        
        self.regen_checkbuttons = {}
        rcount = 1

                
        if len(self.nb.module.layers) > 1:
            shared_outputs = self.nb.module.report_shared_s_outputs()
            self.regen_checkbuttons["__SHARED__"] = {}
            self.nb.ICregen_include_flags["__SHARED__"] = {}
        else:
            shared_outputs = {}
            
        for var in shared_outputs:
            self.nb.ICregen_include_flags["__SHARED__"][var] = tk.IntVar()
            self.nb.ICregen_include_flags["__SHARED__"][var].set(1)
            self.regen_checkbuttons["__SHARED__"][var] = \
                tk.Checkbutton(
                    self.toplevel,
                    text=var,
                    variable=self.nb.ICregen_include_flags["__SHARED__"][var])
            self.regen_checkbuttons["__SHARED__"][var].grid(row=rcount, column=0)
            rcount += 1
        
        for layer_name, layer in self.nb.module.layers.items():
            self.regen_checkbuttons[layer_name] = {}
            self.nb.ICregen_include_flags[layer_name] = {}
            for var in layer.s_outputs:
                if var in shared_outputs: continue
                self.nb.ICregen_include_flags[layer_name][var] = tk.IntVar()
                self.nb.ICregen_include_flags[layer_name][var].set(1)
                self.regen_checkbuttons[layer_name][var] = \
                    tk.Checkbutton(
                        self.toplevel,
                        text="{}: {}".format(layer_name, var),
                        variable=self.nb.ICregen_include_flags[layer_name][var],
                        onvalue=1,offvalue=0)
                self.regen_checkbuttons[layer_name][var].grid(row=rcount, column=0)
                rcount += 1

        self.regen_IC_listbox = \
            tk.Listbox(
                self.toplevel,
                width=30,
                height=10, 
                selectmode='extended')
        self.regen_IC_listbox.grid(row=rcount,column=0,columnspan=2)
        for key, dataset in self.nb.analysis_plots[plot_ID].datagroup.datasets.items():
            over_time = (self.nb.analysis_plots[plot_ID].time > dataset.total_time)
            if over_time: continue
            self.regen_IC_listbox.insert(tk.END, key)

        tk.Button(
            self.toplevel,
            text="Continue", 
            command=partial(self.close, logger, continue_=True)
            ).grid(row=rcount+1,column=0,columnspan=2)

        self.toplevel.protocol(
            "WM_DELETE_WINDOW", 
            partial(self.close, logger, continue_=False))
        
    def close(self, logger=None, continue_=False):
        
        try:
            if continue_:
                plot_ID = self.nb.active_analysisplot_ID.get()
                active_plot = self.nb.analysis_plots[plot_ID]
                active_sets = active_plot.datagroup.datasets
                datasets = [self.regen_IC_listbox.get(i) for i in self.regen_IC_listbox.curselection()]
                if not datasets: 
                    return
                
                include_flags = {}
                for layer_name in self.nb.ICregen_include_flags:
                    include_flags[layer_name] = {}
                    for iflag in self.nb.ICregen_include_flags[layer_name]:
                        include_flags[layer_name][iflag] = self.nb.ICregen_include_flags[layer_name][iflag].get()
                    
                status_msg = ["Files generated:"]
                for key in datasets:
                    new_filename = tk.filedialog.asksaveasfilename(initialdir = self.nb.default_dirs["Initial"], 
                                                                   title="Save IC text file for {}".format(key), 
                                                                   filetypes=[("Text files","*.txt")])
                    if not new_filename: 
                        continue

                    if not new_filename.endswith(".txt"): 
                        new_filename = new_filename + ".txt"
                    
                    param_dict_copy = dict(active_sets[key].params_dict)

                    grid_x = active_sets[key].grid_x
                    
                    filename = active_sets[key].filename
                    
                    shared_outputs = self.nb.module.report_shared_outputs()
                    shared_done = {}
                    sim_data = {}
                    sim_data["__SHARED__"] = {}
                    for layer_name, layer in self.nb.module.layers.items():
                        sim_data[layer_name] = {}
                        for var in layer.s_outputs:
                            is_shared = var in shared_outputs
                            # Don't repeat shared outputs
                            if is_shared and shared_done.get(var, False): continue
                        
                            path_name = os.path.join(self.nb.default_dirs["Data"], 
                                                        self.nb.module.system_ID,
                                                        filename,
                                                        "{}-{}.h5".format(filename, var))
                            floor_tstep = int(active_plot.time / active_sets[key].dt)
                            interpolated_step = u_read(path_name, t0=floor_tstep, t1=floor_tstep+2)
                            
                            if active_plot.time == active_sets[key].total_time:
                                pass
                            else:
                                slope = (interpolated_step[1] - interpolated_step[0]) / (active_sets[key].dt)
                                interpolated_step = interpolated_step[0] + slope * (active_plot.time - floor_tstep * active_sets[key].dt)
                            
                            L = "__SHARED__" if is_shared else layer_name
                            sim_data[L][var] = interpolated_step
                            if is_shared:
                                shared_done[var] = True

                    self.nb.module.get_IC_regen(sim_data, param_dict_copy, 
                                               include_flags, grid_x)
                    if "__SHARED__" in param_dict_copy:
                        param_dict_copy.pop("__SHARED__")
                        
                    for layer_name, layer_params in param_dict_copy.items():
                        for param in layer_params:
                            layer_params[param] *= self.nb.module.layers[layer_name].convert_out.get(param, 1)
                            
                    
                    export_ICfile(new_filename, self.nb, active_sets[key].flags, 
                                  param_dict_copy, allow_write_LGC=False)
                    
                    status_msg.append("{}-->{}".format(filename, new_filename))
                    
                # If NO new files saved
                if len(status_msg) == 1: 
                    status_msg.append("(none)")
                self.nb.do_confirmation_popup("\n".join(status_msg),hide_cancel=True)

            super().close()
            self.IC_regen_popup_isopen = False

        except OSError:
            self.nb.write(self.nb.analysis_status, "Error: failed to regenerate IC file")
            logger.error("Error: failed to regenerate IC file")
            
        except Exception:
            logger.error("Error #511: Failed to close IC regen popup.")
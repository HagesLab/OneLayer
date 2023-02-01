# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:51:26 2022

@author: cfai2
"""
import tkinter as tk
import tkfilebrowser

import os
from functools import partial

from Notebook.Popups.base_popup import Popup

class PlotterPopup(Popup):
    
    def __init__(self, plot_ID, nb, logger):
        super().__init__(nb, grab=True)
        
        tk.ttk.Label(self.toplevel, 
                     text="Select simulated data", 
                     style="Header.TLabel").grid(row=0,column=0,columnspan=2)
        
        self.data_list = []

        # def sims_sort_alpha():
        #     self.data_listbox.delete(0,tk.END)
        #     self.data_list = [file for file in os.listdir(os.path.join(self.nb.default_dirs["Data"], self.nb.module.system_ID)) 
        #                       if not file.endswith(".txt")]
            
        #     self.data_listbox.insert(0,*(self.data_list))
        #     return
            
        # def sims_sort_recent():
        #     self.data_listbox.delete(0,tk.END)
        #     self.data_list = [file for file in os.listdir(os.path.join(self.nb.default_dirs["Data"], self.nb.module.system_ID)) 
        #                       if not file.endswith(".txt")]
            
        #     self.data_list = sorted(self.data_list, key=lambda file: os.path.getmtime(os.path.join(self.nb.default_dirs["Data"], self.nb.module.system_ID, file)), reverse=True)
        #     self.data_listbox.insert(0,*(self.data_list))
        #     return
        
        sorting_frame = tk.Frame(self.toplevel)
        sorting_frame.grid(row=1,column=0)
        
        tk.Button(sorting_frame, text='Browse data',
                  command=self.stage_data).grid(row=0,column=0)
        
        tk.Button(sorting_frame, text='Remove selected data',
                  command=self.unstage_data).grid(row=0,column=1)
        
        
        data_listbox_frame = tk.Frame(self.toplevel)
        data_listbox_frame.grid(row=2,column=0)
        
        self.data_listbox = tk.Listbox(data_listbox_frame, width=80, 
                                       height=20, 
                                       selectmode="extended")
        self.data_listbox.grid(row=0,column=0)
        

        data_listbox_scrollbar = tk.ttk.Scrollbar(data_listbox_frame, orient="vertical",
                                                       command=self.data_listbox.yview)
        data_listbox_scrollbar.grid(row=0,column=1, sticky='ns')
        
        self.data_listbox.config(yscrollcommand=data_listbox_scrollbar.set)
        #sims_sort_alpha()
        
        plotter_options_frame = tk.Frame(self.toplevel)
        plotter_options_frame.grid(row=2,column=1)
        
        if len(self.nb.module.layers) > 1:
            shared_outputs = set.union(*[self.nb.module.report_shared_s_outputs(), self.nb.module.report_shared_c_outputs()])
        else:
            shared_outputs = {}
            
        all_outputs = []
        for shared_output in shared_outputs:
            if self.nb.module.shared_layer.outputs[shared_output].analysis_plotable:
                all_outputs.append("{}".format(shared_output))
            
        for layer_name, layer in self.nb.module.layers.items():
            for output in self.nb.module.layers[layer_name].outputs:
                if output not in shared_outputs and layer.outputs[output].analysis_plotable:
                    all_outputs.append("{}: {}".format(layer_name, output))
        
        # for layer_name, layer in self.nb.module.layers.items():
        #     all_outputs += [output for output in layer.outputs if layer.outputs[output].analysis_plotable]
        tk.OptionMenu(plotter_options_frame, self.nb.data_var, 
                      *all_outputs).grid(row=0,column=0)

        tk.Checkbutton(plotter_options_frame, text="Auto-integrate", 
                       variable=self.nb.check_autointegrate, 
                       onvalue=1, offvalue=0).grid(row=0,column=1)
        
        tk.Button(plotter_options_frame, text="Continue", 
                  command=partial(self.close, plot_ID, logger,
                                  continue_=True)).grid(row=1,column=0,columnspan=2)

        self.plotter_status = tk.Text(plotter_options_frame, width=24,height=2)
        self.plotter_status.grid(row=2,column=0,columnspan=2, padx=(20,20))
        self.plotter_status.configure(state="disabled")

        self.toplevel.protocol("WM_DELETE_WINDOW", partial(self.close, plot_ID, logger,
                                                           continue_=False))
        
        return
    
    def stage_data(self):

        dialog = tkfilebrowser.FileBrowser(parent=self.toplevel, mode="opendir", multiple_selection=True,
                                           title="Select datasets", 
                                           initialdir=os.path.join(self.nb.default_dirs["Data"],
                                                                   self.nb.module.system_ID),
                                           )
        
        dialog._sort_by_date(reverse=True)
        dialog.wait_window(dialog)
        data_dirnames = list(dialog.get_result())
        
        # Screen out dirs without data
        for data_dirname in list(data_dirnames):
            if any(map(lambda t: t.endswith(".h5"), os.listdir(data_dirname))):
                continue
            else:
                data_dirnames.remove(data_dirname)
            
        # Screen duplicate dirs
        for data_dirname in list(data_dirnames):
            if data_dirname in self.data_list:
                data_dirnames.remove(data_dirname)
        
        self.data_list += data_dirnames
        self.data_listbox.delete(0,tk.END)
        self.data_listbox.insert(0,*(self.data_list))
        return
        
    def unstage_data(self):
        remove_these = [i for i in self.data_listbox.curselection()]
        for i in reversed(remove_these):
            self.data_list.pop(i)
        
        self.data_listbox.delete(0,tk.END)
        self.data_listbox.insert(0,*(self.data_list))
        return
    
    def close(self, plot_ID, logger=None, continue_=False):
        try:
            #There are two ways for a popup to close: by the user pressing "Continue" or the user cancelling or pressing "X"
            #We only interpret the input on the popup if the user wants to continue
            self.nb.confirmed = continue_
            if continue_:
                if not self.nb.data_var.get():
                    raise ValueError("Select a data type from the drop-down menu")
                self.nb.analysis_plots[plot_ID].data_pathnames = []
                # A Christmas miracle - tk.askdirectories() has (sort of) been implemented!

                for next_dir in self.data_list:
                    self.nb.analysis_plots[plot_ID].data_pathnames.append(next_dir)

                #self.analysis_plots[plot_ID].remove_duplicate_filenames()
                
                if not self.nb.analysis_plots[plot_ID].data_pathnames:
                    raise ValueError("Select data files")

            super().close()
            self.plotter_popup_isopen = False
            
        except ValueError as oops:
            self.nb.write(self.plotter_status, str(oops))
            logger.error("Error: {}".format(oops))
        except Exception:
            logger.error("Error #502: Failed to close plotter popup.")
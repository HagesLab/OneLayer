# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:51:26 2022

@author: cfai2
"""
import tkinter as tk
import tkfilebrowser

import numpy as np
import os
from functools import partial

from Notebook.Popups.base_popup import Popup

class UploadsPopup(Popup):
    
    def __init__(self, nb, logger, plot_ID=0):
        super().__init__(nb, grab=True)

        self.fnames = nb.integration_plots[plot_ID].datagroup.uploaded_fnames
        self.uploads = nb.integration_plots[plot_ID].datagroup.uploaded_data
        
        tk.ttk.Label(self.toplevel, 
                     text="Uploaded Datasets", 
                     style="Header.TLabel").grid(row=0,column=0,columnspan=2)
        
        sorting_frame = tk.Frame(self.toplevel)
        sorting_frame.grid(row=1,column=0)
        
        tk.Button(sorting_frame, text='Browse external data',
                  command=partial(self.upload_to_integrate_plot,logger=logger)).grid(row=0,column=0)
        
        tk.Button(sorting_frame, text='Remove selected data',
                  command=self.unstage_data).grid(row=0,column=1)
        
        uploads_listbox_frame = tk.Frame(self.toplevel)
        uploads_listbox_frame.grid(row=2,column=0,rowspan=2)
        
        self.uploads_listbox = tk.Listbox(uploads_listbox_frame, width=80, 
                                           height=20, 
                                           selectmode="extended")
        self.uploads_listbox.grid(row=0,column=0)
        
        self.uploads_listbox.delete(0,tk.END)
        self.uploads_listbox.insert(0,*(self.fnames))

        uploads_listbox_scrollbar = tk.ttk.Scrollbar(uploads_listbox_frame, orient="vertical",
                                                       command=self.uploads_listbox.yview)
        uploads_listbox_scrollbar.grid(row=0,column=1, sticky='ns')
        
        self.uploads_listbox.config(yscrollcommand=uploads_listbox_scrollbar.set)
        
        uploads_options_frame = tk.Frame(self.toplevel)
        uploads_options_frame.grid(row=2,column=1)
        
        tk.Label(uploads_options_frame, text="Scale (Ord. Mag.)").grid(row=0,column=0)
        self.scale_entry = tk.ttk.Entry(uploads_options_frame, width=8)
        self.scale_entry.grid(row=0,column=1)
        
        # Refresh scale entry with selected upload's scale value
        def onselect(evt):
            # Note here that Tkinter passes an event object to onselect()
            w = evt.widget
            try:
                i = int(w.curselection()[0])
            except IndexError: # Nothing selected
                self.nb.enter(self.scale_entry, "")
                return
            
            self.nb.enter(self.scale_entry, self.uploads[i][2])
            return
        
        self.uploads_listbox.bind('<<ListboxSelect>>', onselect)
        
        tk.Button(uploads_options_frame, text="Rescale",
                  command=self.rescale).grid(row=0,column=2)

        plotter_options_frame = tk.Frame(self.toplevel)
        plotter_options_frame.grid(row=3,column=1)
        
        tk.Button(plotter_options_frame, text="Continue", 
                  command=partial(self.close, plot_ID, logger,
                                  continue_=True)).grid(row=1,column=0,columnspan=2)

        self.plotter_status = tk.Text(plotter_options_frame, width=24,height=2)
        self.plotter_status.grid(row=2,column=0,columnspan=2, padx=(20,20))
        self.plotter_status.configure(state="disabled")

        self.toplevel.protocol("WM_DELETE_WINDOW", partial(self.close, plot_ID, logger,
                                                           continue_=False))
        
        return
    
    def upload_to_integrate_plot(self, ip_ID=0, allow_clear=True, logger=None):
        # Deconstruct FileBrowser a little to have it launch with files already sorted by date
        dialog = tkfilebrowser.FileBrowser(parent=self.toplevel, mode="openfile", multiple_selection=True,
                                           title="Select other data to plot", 
                                           initialdir=os.path.join(self.nb.default_dirs["Analysis"]),
                                           filetypes=[("Experimental data", "\*.csv|\*.txt|\*.xlsx|\*.xlsm"), ("All files", "\*")])
        dialog._sort_by_date(reverse=True)
        dialog.wait_window(dialog)
        upload_these = dialog.get_result()
        if not upload_these:  # type consistency: always return a tuple
            upload_these = ()

        if len(upload_these) == 0:
            logger.info("No uploads")
            return
        
        for fname in upload_these:
            data = None
            for d in ["\t", ","]: # Try these delimiters
                try:
                    data = np.loadtxt(fname, delimiter=d)
                    break
                except ValueError as e:
                    logger.error(e)
                    continue
                except Exception as e:
                    logger.error(e)
                    break
            
            if data is None:
                logger.error(f"Read {fname} failed; skipping")
                continue
            
            if data.ndim != 2 or data.shape[1] != 2:
                logger.error(f"Read {fname} failed; data must be two-column and comma or tab separated")
                continue
            
            if fname in self.fnames:
                logger.error(f"{fname} already loaded; skipping")
                continue
            
            self.fnames.append(fname)
            self.uploads.append([data[:,0], data[:,1], 1])

        #self.nb.plot_integrate(ip_ID)
        self.uploads_listbox.delete(0,tk.END)
        self.uploads_listbox.insert(0,*(self.fnames))
        
        self.nb.plot_integrate(ip_ID)
        self.toplevel.grab_set()
        return
            
    def unstage_data(self, plot_ID=0):
        remove_these = [i for i in self.uploads_listbox.curselection()]
        for i in reversed(remove_these):
            self.fnames.pop(i)
            self.uploads.pop(i)
        
        self.uploads_listbox.delete(0,tk.END)
        self.uploads_listbox.insert(0,*(self.fnames))
        
        self.nb.plot_integrate(plot_ID)
        self.toplevel.grab_set()
        return
    
    def rescale(self, plot_ID=0):
        try:
            i = int(self.uploads_listbox.curselection()[0])
        except IndexError: # Nothing selected - do nothing
            return
        
        try:
            new_scale_f = float(self.scale_entry.get())
        except ValueError:
            self.nb.write(self.plotter_status, "Invalid scale factor")
            return
        
        self.uploads[i][2] = new_scale_f
        
        dirname, header = os.path.split(self.fnames[i])
        self.nb.write(self.plotter_status, f"{header} Rescaled to {new_scale_f}")
        
        self.nb.plot_integrate(plot_ID)
        self.toplevel.grab_set()
        return
    
    def close(self, plot_ID, logger=None, continue_=False):
        try:
            #self.nb.plot_integrate(plot_ID)

            super().close()
            self.uploads_popup_isopen = False
            
        except ValueError as oops:
            self.nb.write(self.plotter_status, str(oops))
            logger.error("Error: {}".format(oops))
        except Exception:
            logger.error("Error #522: Failed to close uploads popup.")
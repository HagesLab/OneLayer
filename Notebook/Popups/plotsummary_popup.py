# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:56:30 2022

@author: cfai2
"""
import tkinter as tk

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.family':'STIXGeneral'})
matplotlib.rcParams.update({'mathtext.fontset':'stix'})
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
# This lets us pass params to functions called by tkinter buttons
from functools import partial 

from utils import to_array
from utils import autoscale

class PlotSummaryPopup():
    
    def __init__(self, nb):
        """ Draw basic and foundational GUI elements. """
        
        self.nb = nb # Parent notebook
        
        self.sys_plotsummary_popup = tk.Toplevel(self.nb.root)

        self.sys_plotsummary_popup.geometry('%dx%d+0+0' % (self.nb.APP_WIDTH, self.nb.APP_HEIGHT))
        
        self.sys_plotsummary_canvas = tk.Canvas(self.sys_plotsummary_popup)
        self.sys_plotsummary_canvas.grid(row=0,column=0, sticky='nswe')
        
        # Draw everything on this frame
        self.sys_plotsummary_frame = tk.Frame(self.sys_plotsummary_canvas)
        
        if self.nb.sys_flag_dict['symmetric_system'].value():
            self.plotsummary_symmetriclabel = tk.Label(self.sys_plotsummary_frame, 
                                                       text="Note: All distributions "
                                                            "are symmetric about x=0")
            self.plotsummary_symmetriclabel.grid(row=1,column=0)
        
        # Allocate room for and add scrollbars to overall notebook
        self.plotsummary_scroll_y = tk.ttk.Scrollbar(self.sys_plotsummary_popup, orient="vertical", 
                                                     command=self.sys_plotsummary_canvas.yview)
        self.plotsummary_scroll_y.grid(row=0,column=1, sticky='ns')
        self.plotsummary_scroll_x = tk.ttk.Scrollbar(self.sys_plotsummary_popup, orient="horizontal", 
                                                     command=self.sys_plotsummary_canvas.xview)
        self.plotsummary_scroll_x.grid(row=1,column=0,sticky='ew')
        self.sys_plotsummary_canvas.configure(yscrollcommand=self.plotsummary_scroll_y.set, 
                                              xscrollcommand=self.plotsummary_scroll_x.set)
        # Make area for scrollbars as narrow as possible without cutting off
        self.sys_plotsummary_popup.rowconfigure(0,weight=100)
        self.sys_plotsummary_popup.rowconfigure(1,weight=1, minsize=20) 
        self.sys_plotsummary_popup.columnconfigure(0,weight=100)
        self.sys_plotsummary_popup.columnconfigure(1,weight=1, minsize=20)
        
        self.sys_plotsummary_canvas.create_window((0,0), window=self.sys_plotsummary_frame, anchor="nw")
        scroll_cmd = lambda e:self.sys_plotsummary_canvas.configure(scrollregion=self.sys_plotsummary_canvas.bbox('all'))
        self.sys_plotsummary_frame.bind('<Configure>', scroll_cmd)                               
        
        
        active_plotsummary_layers = [name for name in self.nb.module.layers if self.nb.module.layers[name].spacegrid_is_set]
        
        self.draw_sys_plotsummary_buttons(active_plotsummary_layers)
        all_layers = (len(active_plotsummary_layers) > 1 and 
                      len(active_plotsummary_layers) == len(self.nb.module.layers))
        self.update_sys_plotsummary_plots(active_plotsummary_layers,
                                          all_layers)
        
        self.sys_plotsummary_popup.protocol("WM_DELETE_WINDOW", 
                                            self.close)
        ## Temporarily disable the main window while this popup is active
        self.sys_plotsummary_popup.grab_set()
            
        return
            
    def close(self):
        """ Delete this popup """
        self.sys_plotsummary_popup.destroy()
        return
    
    def draw_sys_plotsummary_buttons(self, active_plotsummary_layers):
        """ One button per layer, plus a "Select All" button if all layers are
            defined """
        self.sys_plotsummary_buttongrid = tk.Frame(self.sys_plotsummary_frame)
        self.sys_plotsummary_buttongrid.grid(row=0,column=0)
        
        for i, s in enumerate(active_plotsummary_layers):
            tk.Button(self.sys_plotsummary_buttongrid,text=s, 
                      command=partial(self.update_sys_plotsummary_plots, [s])
                      ).grid(row=0,column=i)
            
        if (len(active_plotsummary_layers) > 1 and 
            len(active_plotsummary_layers) == len(self.nb.module.layers)): # if all layers set
            tk.Button(self.sys_plotsummary_buttongrid,text="All", 
                      command=partial(self.update_sys_plotsummary_plots, active_plotsummary_layers, all_layers=True)
                      ).grid(row=0,column=i+1)
        return
    
    def cleanup(self):
        """ Clear any previously drawn plots. """
        if hasattr(self, "plotsummary_plotwidgets"):
            for pw in self.plotsummary_plotwidgets:
                pw.destroy()
            for tb in self.plotsummary_toolbars:
                tb.destroy()
            for gf in self.plotsummary_graphicframes:
                gf.destroy()
                
    def determine_layer_lengths(self):
        """ Collect total_length of each layer. """
        DEFAULT_LENGTH = 0
        layer_lengths = [self.nb.module.layers[name].total_length 
                         if self.nb.module.layers[name].spacegrid_is_set else DEFAULT_LENGTH
                         for name in self.nb.module.layers]
        
        full_length = sum(layer_lengths)
        
        # Assign a default length to all layers without defined spacegrids,
        # equal to the average of the lengths of layers that do have spacegrids
        layer_lengths = [l if l != DEFAULT_LENGTH else full_length / len(layer_lengths)
                         for l in layer_lengths]
        full_length = sum(layer_lengths)
        return layer_lengths, full_length
    
    def determine_shared_params(self, all_layers):
        """ Shared parameters are treated specially by the all_layers option. """
        if all_layers:
            shared_params = set.intersection(*[set(self.nb.module.layers[layer].params.keys())
                                              for layer in self.nb.module.layers])
        else:
            shared_params = []
            
        return shared_params
    
    def setup_containers(self, num_figs):
        # Following Matplotlib (fig, axes) convention
        self.plotsummary_figs = [None] * num_figs
        self.plotsummary_axes = [{}] * num_figs
        self.plotsummary_canvases = [None] * num_figs
        self.plotsummary_plotwidgets = [None] * num_figs
        self.plotsummary_toolbars = [None] * num_figs
        self.plotsummary_graphicframes = [None] * num_figs
        
    def guess_plot_dims(self, plot_count):
        """ Attempt to arrange plot_count subplots into a nice looking rectangle """
        rdim = np.floor(np.sqrt(plot_count))
        cdim = np.ceil(plot_count / rdim)
        return rdim, cdim
    
    def setup_figures(self, i,rdim, cdim, share_special=False, shared_params=[], layer_name=None):
        self.plotsummary_figs[i] = Figure(figsize=(self.nb.APP_WIDTH / self.nb.APP_DPI,
                                                   self.nb.APP_HEIGHT / self.nb.APP_DPI))
        
        count = 1
        if share_special:
            for param_name in shared_params:
                if all((self.nb.module.layers[layer_name].params[param_name].is_space_dependent for layer_name in self.nb.module.layers)):
                    self.plotsummary_axes[i][param_name] = self.plotsummary_figs[i].add_subplot(int(rdim), int(cdim), int(count))
                    self.plotsummary_axes[i][param_name].set_title("{} {}".format(param_name, self.nb.module.layers[next(iter(self.nb.module.layers))].params[param_name].units))
                    count += 1
        
        else:
            layer = self.nb.module.layers[layer_name]

            for param_name in layer.params:
                if param_name in shared_params:
                    continue
                if layer.params[param_name].is_space_dependent:
                    self.plotsummary_axes[i][param_name] = self.plotsummary_figs[i].add_subplot(int(rdim), int(cdim), int(count))
                    self.plotsummary_axes[i][param_name].set_title("{} {}".format(param_name,layer.params[param_name].units))
                    count += 1
        
        self.plotsummary_canvases[i] = tkagg.FigureCanvasTkAgg(self.plotsummary_figs[i], 
                                                               master=self.sys_plotsummary_frame)
        self.plotsummary_plotwidgets[i] = self.plotsummary_canvases[i].get_tk_widget()
        self.plotsummary_plotwidgets[i].grid(row=2+2*i,column=0)
        
        self.plotsummary_toolbars[i] = tk.Frame(self.sys_plotsummary_frame)
        self.plotsummary_toolbars[i].grid(row=2+(2*i)+1, column=0)
        tkagg.NavigationToolbar2Tk(self.plotsummary_canvases[i], 
                                   self.plotsummary_toolbars[i])
        return
    
    def draw_graphic(self, i, layer_lengths, full_length, share_special=False, layer_name=""):
        width = 300
        h_offset = 10
        height = 200
        w_offset = 50
        
        self.plotsummary_graphicframes[i] = tk.Frame(self.sys_plotsummary_frame)
        self.plotsummary_graphicframes[i].grid(row=2+2*i, column=1)
        
        graphic_text = "All layers" if share_special else layer_name
        tk.Label(self.plotsummary_graphicframes[i], text=graphic_text).grid(row=0,column=0)
        
        graphic_canvas = tk.Canvas(self.plotsummary_graphicframes[i], width=width+4*h_offset,height=height,bg='white')
        graphic_canvas.grid(row=1,column=0, padx=(20,20))
        x1 = h_offset
        x2 = h_offset
            
        if share_special:            
            for j, ll_name in enumerate(self.nb.module.layers):
                fillcolor = 'cornflowerblue'
                
                x2 += layer_lengths[j] / full_length * width
                graphic_canvas.create_rectangle(x1, w_offset, x2, height - w_offset, fill=fillcolor)
                
                
                graphic_canvas.create_line(x1, height - w_offset, x1, height, fill='black')
                    
                graphic_canvas.create_text(x1+0.75*h_offset, height, anchor='w', text="z≈{}".format(int((x1-h_offset) * full_length / width)), 
                                           angle=90)
                x1 = x2
                
            graphic_canvas.create_line(x2, height - w_offset, x2, height, fill='black')
            graphic_canvas.create_text(x2+0.75*h_offset, height, anchor='w', 
                                       text="z≈{}".format(int((x2-h_offset) * full_length / width)),
                                       angle=90)
        
        else:
            for j, ll_name in enumerate(self.nb.module.layers):
                do_nogrid_warning = False
                if ll_name == layer_name:
                    fillcolor = 'cornflowerblue'
                elif self.nb.module.layers[ll_name].spacegrid_is_set:
                    fillcolor= 'gray'
                else:
                    do_nogrid_warning = True
                    fillcolor= 'white'
                
                x2 += layer_lengths[j] / full_length * width
                graphic_canvas.create_rectangle(x1, w_offset, x2, height - w_offset, fill=fillcolor)
                if do_nogrid_warning:
                    graphic_canvas.create_line(x1, w_offset, x2, height - w_offset, fill='black', dash=(3,5))
                    graphic_canvas.create_line(x1, height - w_offset, x2, w_offset, fill='black', dash=(3,5))
                    graphic_canvas.create_text((x1+x2)/2, w_offset*2, text="Grid not set!")
                    
                if ll_name == layer_name:
                    graphic_canvas.create_line(x1, height - w_offset, x1, height, fill='black')
                    graphic_canvas.create_line(x2, height - w_offset, x2, height, fill='black')
                    graphic_canvas.create_text(x1+0.75*h_offset, height, anchor='w', text="z=0", 
                                               angle=90)
                    graphic_canvas.create_text(x2+0.75*h_offset, height, anchor='w', 
                                               text="z≈{}".format(int(layer_lengths[j])),
                                               angle=90)
                x1 = x2
        return
    
    def plot_on_figures(self, i, share_special=False, shared_params=[], layer_name=""):
        # Join this shared parameter's values across all layers
        
        if share_special:
            for param_name in shared_params:
                shared_x = []
                shared_y = []
                cml_total_length = 0
                for layer_name in self.nb.module.layers:
                    layer = self.nb.module.layers[layer_name]
                    param = layer.params[param_name]
                    if param.is_space_dependent:
                        val = to_array(param.value, len(layer.grid_x_nodes), 
                                       param.is_edge)
                        shared_y.append(val)
                        
                        grid_x = layer.grid_x_nodes if not param.is_edge else layer.grid_x_edges
                        shared_x.append(grid_x + cml_total_length)
                        
                        self.plotsummary_axes[0][param_name].axvline(cml_total_length, color='black', linestyle='dashed', linewidth=0.2)
                        cml_total_length += layer.total_length
                        
                self.plotsummary_axes[0][param_name].axvline(cml_total_length, color='black', linestyle='dashed', linewidth=0.2)
                shared_x = np.hstack(shared_x)
                shared_y = np.hstack(shared_y)
                self.plotsummary_axes[0][param_name].plot(shared_x, shared_y)
                self.plotsummary_axes[0][param_name].set_yscale(autoscale(val_array=val))
            
        # Actually plot the parameters
        else:
            layer = self.nb.module.layers[layer_name]
            for param_name in layer.params:
                if param_name in shared_params:
                    continue
                param = layer.params[param_name]
                if param.is_space_dependent:
                    val = to_array(param.value, len(layer.grid_x_nodes), 
                                   param.is_edge)
                    grid_x = layer.grid_x_nodes if not param.is_edge else layer.grid_x_edges
                    self.plotsummary_axes[i][param_name].plot(grid_x, val)
                    self.plotsummary_axes[i][param_name].set_yscale(autoscale(val_array=val))
                    
        return
            
    def update_sys_plotsummary_plots(self, active_plotsummary_layers, all_layers=False):
        """ Draw dynamic elements, including plots of parameters and diagrams of layers. """
    
        self.cleanup()
        
        layer_lengths, full_length = self.determine_layer_lengths()        
        
        shared_params = self.determine_shared_params(all_layers)
        
        num_figs = len(active_plotsummary_layers)
        num_figs = num_figs + 1 if len(shared_params) > 0 else num_figs
        self.setup_containers(num_figs)
            
        if len(shared_params) > 0:
            rdim, cdim = self.guess_plot_dims(len(shared_params))
            
            self.setup_figures(0, rdim, cdim, share_special=True, shared_params=shared_params)
            
            self.draw_graphic(0, layer_lengths, full_length, share_special=True)
            
            self.plot_on_figures(0, share_special=True, shared_params=shared_params)
            skip1 = True
        else:
            skip1 = False
                
        start = 1 if skip1 else 0
        for i, layer_name in enumerate(active_plotsummary_layers, start=start):
            
            rdim, cdim = self.guess_plot_dims(self.nb.module.layers[layer_name].param_count - len(shared_params))
            
            self.setup_figures(i, rdim, cdim, shared_params=shared_params, layer_name=layer_name)
            
            self.draw_graphic(i, layer_lengths, full_length, layer_name=layer_name)
            
            self.plot_on_figures(i, shared_params=shared_params, layer_name=layer_name)
            
        for fig in self.plotsummary_figs:
            fig.tight_layout()
            fig.canvas.draw()
        return
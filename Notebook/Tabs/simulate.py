
import numpy as np
import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import tkinter as tk


def add_tab_simulate(nb):
    """Method to add the menu tab 'Simulate'
    to the inout Notebook object and return it."""

    nb.tab_simulate = tk.ttk.Frame(nb.notebook)

    tk.ttk.Label(
        nb.tab_simulate, 
        text="Select Init. Cond.", 
        style="Header.TLabel"
        ).grid(row=0,column=0,columnspan=2, padx=(9,12))

    tk.ttk.Label(
        nb.tab_simulate,
        text="Simulation Time [ns]"
        ).grid(row=2,column=0)

    nb.simtime_entry = tk.ttk.Entry(nb.tab_simulate, width=9)
    nb.simtime_entry.grid(row=2,column=1)

    tk.ttk.Label(
        nb.tab_simulate,
        text="dt [ns]"
        ).grid(row=3,column=0)

    nb.dt_entry = tk.ttk.Entry(nb.tab_simulate, width=9)
    nb.dt_entry.grid(row=3,column=1)
    
    tk.ttk.Label(
        nb.tab_simulate,
        text="Max solver stepsize [ns]"
        ).grid(row=4,column=0)
    
    nb.hmax_entry = tk.ttk.Entry(nb.tab_simulate, width=9)
    nb.hmax_entry.grid(row=4,column=1)

    nb.enter(nb.dt_entry, "0.5")
    nb.enter(nb.hmax_entry, "0.25")
    
    tk.ttk.Label(
        nb.tab_simulate,
        text="rtol"
        ).grid(row=5,column=0)

    nb.rtol_entry = tk.ttk.Entry(nb.tab_simulate, width=9)
    nb.rtol_entry.grid(row=5,column=1)
    
    tk.ttk.Label(
        nb.tab_simulate,
        text="atol"
        ).grid(row=6,column=0)
    
    nb.atol_entry = tk.ttk.Entry(nb.tab_simulate, width=9)
    nb.atol_entry.grid(row=6,column=1)

    nb.enter(nb.rtol_entry, "1e-5")
    nb.enter(nb.atol_entry, "1e-8")
    
    tk.ttk.Button(
        nb.tab_simulate,
        text="Start Simulation(s)", 
        command=nb.do_Batch
        ).grid(row=7,column=0,columnspan=2,padx=(9,12))

    tk.ttk.Label(
        nb.tab_simulate,
        text="Status"
        ).grid(row=8, column=0, columnspan=2)

    nb.status = tk.Text(nb.tab_simulate, width=28,height=4)
    nb.status.grid(row=9, rowspan=2, column=0, columnspan=2)
    nb.status.configure(state='disabled')

    tk.ttk.Separator(
        nb.tab_simulate,
        orient="vertical", 
        style="Grey Bar.TSeparator"
        ).grid(row=0,rowspan=30,column=2,sticky="ns")

    tk.ttk.Label(
        nb.tab_simulate, 
        text="Simulation - {}".format(nb.module.system_ID)
        ).grid(row=0,column=3,columnspan=3)
    
    nb.sim_fig = Figure(figsize=(14, 8))
    count = 1
    cdim = np.ceil(np.sqrt(nb.module.count_s_outputs()))
    
    rdim = np.ceil(nb.module.count_s_outputs() / cdim)
    
    nb.sim_subplots = {}
    for layer_name, layer in nb.module.layers.items():
        for s_output in layer.s_outputs:
            if s_output not in nb.sim_subplots:
                nb.sim_subplots[s_output] = \
                    nb.sim_fig.add_subplot(
                        int(rdim), 
                        int(cdim), 
                        int(count))
                nb.sim_subplots[s_output].set_title(s_output)
                count += 1

    nb.sim_canvas = \
        tkagg.FigureCanvasTkAgg(nb.sim_fig, master=nb.tab_simulate)
    nb.sim_canvas.get_tk_widget().grid(row=1,column=3,rowspan=12,columnspan=2)
    
    nb.simfig_toolbar_frame = tk.ttk.Frame(master=nb.tab_simulate)
    nb.simfig_toolbar_frame.grid(row=13,column=3,columnspan=2)
    tkagg.NavigationToolbar2Tk(nb.sim_canvas, nb.simfig_toolbar_frame)

    nb.notebook.add(nb.tab_simulate, text="Simulate")

    return nb

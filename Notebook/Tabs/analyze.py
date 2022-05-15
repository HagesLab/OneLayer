
import numpy as np
import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import tkinter as tk
from functools import partial 


def add_tab_analyze(nb):
    """Method to add the menu tab 'Analyze'
    to the inout Notebook object and return it."""
    nb.tab_analyze = tk.ttk.Notebook(nb.notebook)
    nb.tab_overview_analysis = tk.ttk.Frame(nb.tab_analyze)
    nb.tab_detailed_analysis = tk.ttk.Frame(nb.tab_analyze)
    
    nb.analyze_overview_fig = Figure(figsize=(21,8))
    nb.overview_subplots = {}
    count = 1
    total_outputs_count = sum([nb.module.layers[layer].outputs_count for layer in nb.module.layers])
    rdim = np.floor(np.sqrt(total_outputs_count))
    cdim = np.ceil(total_outputs_count / rdim)
    
    all_outputs = []
    for layer_name in nb.module.layers:
        nb.overview_subplots[layer_name] = {}
        for output in nb.module.layers[layer_name].outputs:
            nb.overview_subplots[layer_name][output] = nb.analyze_overview_fig.add_subplot(int(rdim), int(cdim), int(count))
            all_outputs.append("{}: {}".format(layer_name, output))
            count += 1
                
    nb.overview_setup_frame = tk.Frame(nb.tab_overview_analysis)
    nb.overview_setup_frame.grid(row=0,column=0, padx=(20,20))
    
    tk.Label(nb.overview_setup_frame, text="No. samples").grid(row=0,column=0)
    
    nb.overview_samplect_entry = tk.ttk.Entry(nb.overview_setup_frame, width=8)
    nb.overview_samplect_entry.grid(row=1,column=0)
    nb.enter(nb.overview_samplect_entry, "6")
    
    nb.overview_sample_mode = tk.StringVar()
    nb.overview_sample_mode.set("Log")
    tk.ttk.Radiobutton(nb.overview_setup_frame, variable=nb.overview_sample_mode,
                        value="Linear").grid(row=0,column=1)
    tk.Label(nb.overview_setup_frame, text="Linear").grid(row=0,column=2)
    
    tk.ttk.Radiobutton(nb.overview_setup_frame, variable=nb.overview_sample_mode,
                        value="Log").grid(row=1,column=1)
    tk.Label(nb.overview_setup_frame, text="Log").grid(row=1,column=2)
    
    tk.ttk.Radiobutton(nb.overview_setup_frame, variable=nb.overview_sample_mode,
                        value="Custom").grid(row=2,column=1)
    tk.Label(nb.overview_setup_frame, text="Custom").grid(row=2,column=2)
    
    tk.Button(master=nb.overview_setup_frame, text="Select Dataset", 
                    command=nb.plot_overview_analysis).grid(row=0,rowspan=2,column=3)
    
    nb.overview_var_selection = tk.StringVar()
    
    tk.ttk.OptionMenu(nb.tab_overview_analysis, nb.overview_var_selection, 
                        all_outputs[0], *all_outputs).grid(row=0,column=1)
    
    tk.ttk.Button(master=nb.tab_overview_analysis, text="Export", 
                    command=nb.export_overview).grid(row=0,column=2)
    
    
    nb.analyze_overview_canvas = tkagg.FigureCanvasTkAgg(nb.analyze_overview_fig, 
                                                            master=nb.tab_overview_analysis)
    nb.analyze_overview_canvas.get_tk_widget().grid(row=1,column=0,columnspan=99)

    nb.overview_toolbar_frame = tk.ttk.Frame(nb.tab_overview_analysis)
    nb.overview_toolbar_frame.grid(row=2,column=0,columnspan=99)
    
    tkagg.NavigationToolbar2Tk(nb.analyze_overview_canvas, 
                                nb.overview_toolbar_frame).grid(row=0,column=0)
    
    tk.ttk.Label(nb.tab_detailed_analysis, 
                    text="Plot and Integrate Saved Datasets", 
                    style="Header.TLabel").grid(row=0,column=0,columnspan=8)
    
    nb.analyze_fig = Figure(figsize=(9.8,6))
    # add_subplot() starts counting indices with 1 instead of 0
    nb.analyze_subplot0 = nb.analyze_fig.add_subplot(221)
    nb.analyze_subplot1 = nb.analyze_fig.add_subplot(222)
    nb.analyze_subplot2 = nb.analyze_fig.add_subplot(223)
    nb.analyze_subplot3 = nb.analyze_fig.add_subplot(224)
    nb.analysis_plots[0].plot_obj = nb.analyze_subplot0
    nb.analysis_plots[1].plot_obj = nb.analyze_subplot1
    nb.analysis_plots[2].plot_obj = nb.analyze_subplot2
    nb.analysis_plots[3].plot_obj = nb.analyze_subplot3
    
    nb.analyze_canvas = tkagg.FigureCanvasTkAgg(nb.analyze_fig, 
                                                    master=nb.tab_detailed_analysis)
    nb.analyze_canvas.get_tk_widget().grid(row=1,column=0,rowspan=1,columnspan=4, padx=(12,0))

    nb.analyze_plotselector_frame = tk.ttk.Frame(master=nb.tab_detailed_analysis)
    nb.analyze_plotselector_frame.grid(row=2,rowspan=2,column=0,columnspan=4)
    
    tk.ttk.Radiobutton(nb.analyze_plotselector_frame, 
                        variable=nb.active_analysisplot_ID, 
                        value=0).grid(row=0,column=0)

    tk.ttk.Label(nb.analyze_plotselector_frame, 
                    text="Use: Top Left").grid(row=0,column=1)
    
    tk.ttk.Radiobutton(nb.analyze_plotselector_frame, 
                        variable=nb.active_analysisplot_ID, 
                        value=1).grid(row=0,column=2)

    tk.ttk.Label(nb.analyze_plotselector_frame, 
                    text="Use: Top Right").grid(row=0,column=3)
    
    tk.ttk.Radiobutton(nb.analyze_plotselector_frame, 
                        variable=nb.active_analysisplot_ID, 
                        value=2).grid(row=1,column=0)

    tk.ttk.Label(nb.analyze_plotselector_frame, 
                    text="Use: Bottom Left").grid(row=1,column=1)
    
    tk.ttk.Radiobutton(nb.analyze_plotselector_frame, 
                        variable=nb.active_analysisplot_ID, 
                        value=3).grid(row=1,column=2)

    tk.ttk.Label(nb.analyze_plotselector_frame, 
                    text="Use: Bottom Right").grid(row=1,column=3)
    
    nb.analyze_toolbar_frame = tk.ttk.Frame(master=nb.tab_detailed_analysis)
    nb.analyze_toolbar_frame.grid(row=4,column=0,rowspan=4,columnspan=4)
    tkagg.NavigationToolbar2Tk(nb.analyze_canvas, nb.analyze_toolbar_frame).grid(row=0,column=0,columnspan=7)

    tk.ttk.Button(nb.analyze_toolbar_frame, 
                    text="Plot", 
                    command=partial(nb.load_datasets)).grid(row=1,column=0)
    
    nb.analyze_tstep_entry = tk.ttk.Entry(nb.analyze_toolbar_frame, width=9)
    nb.analyze_tstep_entry.grid(row=1,column=1)

    tk.ttk.Button(nb.analyze_toolbar_frame, 
                    text="Time >>", 
                    command=partial(nb.plot_tstep)).grid(row=1,column=2)

    tk.ttk.Button(nb.analyze_toolbar_frame, 
                    text=">> Integrate <<", 
                    command=partial(nb.do_Integrate)).grid(row=1,column=3)

    tk.ttk.Button(nb.analyze_toolbar_frame, 
                    text="Axis Settings", 
                    command=partial(nb.do_change_axis_popup, 
                                    from_integration=0)).grid(row=1,column=4)

    tk.ttk.Button(nb.analyze_toolbar_frame, 
                    text="Export", 
                    command=partial(nb.export_plot, 
                                    from_integration=0)).grid(row=1,column=5)

    tk.ttk.Button(nb.analyze_toolbar_frame, 
                    text="Generate IC", 
                    command=partial(nb.do_IC_carry_popup)).grid(row=1,column=6)

    nb.integration_fig = Figure(figsize=(9,5))
    nb.integration_subplot = nb.integration_fig.add_subplot(111)
    nb.integration_plots[0].plot_obj = nb.integration_subplot

    nb.integration_canvas = tkagg.FigureCanvasTkAgg(nb.integration_fig, 
                                                        master=nb.tab_detailed_analysis)
    nb.integration_canvas.get_tk_widget().grid(row=1,column=5,rowspan=1,columnspan=1, padx=(20,0))

    nb.integration_toolbar_frame = tk.ttk.Frame(master=nb.tab_detailed_analysis)
    nb.integration_toolbar_frame.grid(row=3,column=5, rowspan=2,columnspan=1)
    tkagg.NavigationToolbar2Tk(nb.integration_canvas, 
                                nb.integration_toolbar_frame).grid(row=0,column=0,columnspan=5)

    tk.ttk.Button(nb.integration_toolbar_frame, 
                    text="Axis Settings", 
                    command=partial(nb.do_change_axis_popup, 
                                    from_integration=1)).grid(row=1,column=0)

    tk.ttk.Button(nb.integration_toolbar_frame, 
                    text="Export", 
                    command=partial(nb.export_plot, 
                                    from_integration=1)).grid(row=1,column=1)

    # self.integration_bayesim_button = tk.ttk.Button(self.integration_toolbar_frame, text="Bayesim", command=partial(self.do_bayesim_popup))
    # self.integration_bayesim_button.grid(row=1,column=2)

    nb.analysis_status = tk.Text(nb.tab_detailed_analysis, width=28,height=3)
    nb.analysis_status.grid(row=5,rowspan=3,column=5,columnspan=1)
    nb.analysis_status.configure(state="disabled")

    nb.tab_analyze.add(nb.tab_overview_analysis, text="Overview")
    nb.tab_analyze.add(nb.tab_detailed_analysis, text="Detailed Analysis")
    nb.notebook.add(nb.tab_analyze, text="Analyze")

    return nb

import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import tkinter as tk

from Notebook.Tabs.LGC_frame import create_LGC_frame

from GUI_structs import Flag

def add_tab_inputs(nb):
    """ Add the menu tab 'Inputs' to the main notebook
    which is passed as argument 'nb' and returned at the end."""

    nb.tab_inputs = tk.ttk.Notebook(nb.notebook)
    nb.tab_generation_init = tk.ttk.Frame(nb.tab_inputs)
    nb.tab_rules_init = tk.ttk.Frame(nb.tab_inputs)
    nb.tab_explicit_init = tk.ttk.Frame(nb.tab_inputs)
    
    first_layer = next(iter(nb.module.layers))

    var_dropdown_list = ["{} {}".format(param_name, param.units) 
                            for param_name, param in nb.module.layers[first_layer].params.items()
                            if param.is_space_dependent]
    paramtoolkit_method_dropdown_list = ["POINT", "FILL", "LINE", "EXP"]
    unitless_dropdown_list = [param_name 
                                for param_name, param in nb.module.layers[first_layer].params.items()
                                if param.is_space_dependent]
    
    nb.line_sep_style = tk.ttk.Style()
    nb.line_sep_style.configure("Grey Bar.TSeparator", background='#000000', 
                                    padding=160)

    nb.header_style = tk.ttk.Style()
    nb.header_style.configure("Header.TLabel", background='#D0FFFF',
                                highlightbackground='#000000')

    # We use the grid location specifier for general placement and padx/pady for fine-tuning
    # The other two options are the pack specifier, which doesn't really provide enough versatility,
    # and absolute coordinates, which maximize versatility but are a pain to adjust manually.
    nb.IO_frame = tk.ttk.Frame(nb.tab_inputs)
    nb.IO_frame.grid(row=0,column=0,columnspan=2, pady=(25,0))
    
    tk.ttk.Button(nb.IO_frame, text="Load", 
                    command=nb.select_init_file).grid(row=0,column=0)

    tk.ttk.Button(nb.IO_frame, text="debug", 
                    command=nb.DEBUG).grid(row=0,column=1)

    tk.ttk.Button(nb.IO_frame, text="Save", 
                    command=nb.save_ICfile).grid(row=0,column=2)

    tk.ttk.Button(nb.IO_frame, text="Reset", 
                    command=nb.reset_IC).grid(row=0, column=3)

    nb.spacegrid_frame = tk.ttk.Frame(nb.tab_inputs)
    nb.spacegrid_frame.grid(row=1, column=0, columnspan=2, pady=(10, 10))

    tk.ttk.Label(
        nb.spacegrid_frame,
        text="Space Grid - Start Here",
        style="Header.TLabel"
        ).grid(row=0, column=0, columnspan=2)

    tk.ttk.Label(
        nb.spacegrid_frame, 
        text="Thickness " + nb.module.layers[first_layer].length_unit
        ).grid(row=1 ,column=0)

    nb.thickness_entry = tk.ttk.Entry(nb.spacegrid_frame, width=9)
    nb.thickness_entry.grid(row=1, column=1)

    tk.ttk.Label(
        nb.spacegrid_frame,
        text="Node width " + nb.module.layers[first_layer].length_unit
        ).grid(row=2, column=0)

    nb.dx_entry = tk.ttk.Entry(nb.spacegrid_frame, width=9)
    nb.dx_entry.grid(row=2,column=1)

    nb.params_frame = tk.ttk.Frame(nb.tab_inputs)
    nb.params_frame.grid(row=2, column=0, columnspan=2, rowspan=4, pady=(10, 10))

    tk.ttk.Label(
        nb.params_frame, 
        text="Constant-value Parameters",
        style="Header.TLabel"
        ).grid(row=0, column=0, columnspan=2)
    
    tk.ttk.Button(
        nb.params_frame, 
        text="Fast Param Entry Tool",
        command=nb.do_sys_param_shortcut_popup
        ).grid(row=1, column=0, columnspan=2)
    
    nb.flags_frame = tk.ttk.Frame(nb.tab_inputs)
    nb.flags_frame.grid(row=6, column=0, columnspan=2, pady=(10, 10))

    tk.ttk.Label(
        nb.flags_frame,
        text="Flags",
        style="Header.TLabel"
        ).grid(row=0, column=0, columnspan=2)
    
    # Procedurally generated elements for flags
    i = 1
    nb.sys_flag_dict = {}
    for flag in nb.module.flags_dict:
        nb.sys_flag_dict[flag] = Flag(nb.flags_frame, 
                                        nb.module.flags_dict[flag][0])
        nb.sys_flag_dict[flag].set(nb.module.flags_dict[flag][2])
        
        if not nb.module.flags_dict[flag][1]:
            continue
        else:
            nb.sys_flag_dict[flag].tk_element.grid(row=i, column=0)
            i += 1
            
    nb.ICtab_status = tk.Text(nb.tab_inputs, width=24, height=8)
    nb.ICtab_status.grid(row=7, column=0, columnspan=2)
    nb.ICtab_status.configure(state='disabled')
    
    tk.ttk.Button(
        nb.tab_inputs, 
        text="Print Init. State Summary", 
        command=nb.do_sys_printsummary_popup
        ).grid(row=8, column=0, columnspan=2)
    
    tk.ttk.Button(
        nb.tab_inputs,
        text="Show Init. State Plots",
        command=nb.do_sys_plotsummary_popup
        ).grid(row=9, column=0, columnspan=2)
    
    tk.ttk.Separator(
        nb.tab_inputs,
        orient="horizontal",
        style="Grey Bar.TSeparator"
        ).grid(row=10, column=0, columnspan=2, pady=(10, 10), sticky="ew")
    
    nb.layer_statusbox = tk.Text(nb.tab_inputs, width=24, height=1)
    nb.layer_statusbox.grid(row=11, column=0, columnspan=2)
    
    # Init this dropdown with some default layer
    tk.ttk.OptionMenu(
        nb.tab_inputs,
        nb.current_layer_selection,
        first_layer,
        *nb.module.layers
        ).grid(row=12, column=0)
    
    tk.ttk.Button(
        nb.tab_inputs,
        text="Change to Layer",
        command=nb.change_layer
        ).grid(row=12, column=1)
    
    tk.ttk.Separator(
        nb.tab_inputs,
        orient="vertical", 
        style="Grey Bar.TSeparator"
        ).grid(row=0, rowspan=30, column=2, pady=(24, 0), sticky="ns")
            
    ## Parameter Toolkit:

    nb.param_rules_frame = tk.ttk.Frame(nb.tab_rules_init)
    nb.param_rules_frame.grid(row=0,column=0,padx=(370,0))

    tk.ttk.Label(
        nb.param_rules_frame, 
        text="Add/Edit/Remove Space-Dependent Parameters", 
        style="Header.TLabel"
        ).grid(row=0, column=0, columnspan=3)

    nb.active_paramrule_listbox = \
        tk.Listbox(
            nb.param_rules_frame,
            width=86,
            height=8)
    nb.active_paramrule_listbox.grid(
        row=1, rowspan=3, column=0, columnspan=3, padx=(32, 32))

    tk.ttk.Label(
        nb.param_rules_frame, 
        text="Select parameter to edit:"
        ).grid(row=4, column=0)
    
    tk.ttk.OptionMenu(
        nb.param_rules_frame, 
        nb.init_var_selection, 
        var_dropdown_list[0], 
        *var_dropdown_list
        ).grid(row=4,column=1)

    tk.ttk.Label(nb.param_rules_frame, 
                    text="Select calculation method:").grid(row=5,column=0)

    tk.ttk.OptionMenu(nb.param_rules_frame, 
                        nb.init_shape_selection,
                        paramtoolkit_method_dropdown_list[0], 
                        *paramtoolkit_method_dropdown_list).grid(row=5, column=1)

    tk.ttk.Label(nb.param_rules_frame, 
                    text="Left bound coordinate:").grid(row=6, column=0)

    nb.paramrule_lbound_entry = tk.ttk.Entry(nb.param_rules_frame, width=8)
    nb.paramrule_lbound_entry.grid(row=6,column=1)

    tk.ttk.Label(nb.param_rules_frame, 
                    text="Right bound coordinate:").grid(row=7, column=0)

    nb.paramrule_rbound_entry = tk.ttk.Entry(nb.param_rules_frame, width=8)
    nb.paramrule_rbound_entry.grid(row=7,column=1)

    tk.ttk.Label(nb.param_rules_frame, 
                    text="Left bound value:").grid(row=8, column=0)

    nb.paramrule_lvalue_entry = tk.ttk.Entry(nb.param_rules_frame, width=8)
    nb.paramrule_lvalue_entry.grid(row=8,column=1)

    tk.ttk.Label(nb.param_rules_frame, 
                    text="Right bound value:").grid(row=9, column=0)

    nb.paramrule_rvalue_entry = tk.ttk.Entry(nb.param_rules_frame, width=8)
    nb.paramrule_rvalue_entry.grid(row=9,column=1)

    tk.ttk.Button(nb.param_rules_frame, 
                    text="Add new parameter rule", 
                    command=nb.add_paramrule).grid(row=10,column=0,columnspan=2)

    tk.ttk.Button(nb.param_rules_frame, 
                    text="Delete highlighted rule", 
                    command=nb.delete_paramrule).grid(row=4,column=2)

    tk.ttk.Button(nb.param_rules_frame, 
                    text="Delete all rules for this parameter", 
                    command=nb.deleteall_paramrule).grid(row=5,column=2)

    tk.Message(nb.param_rules_frame, 
                text="The Parameter Toolkit uses a series "
                "of rules and patterns to build a spatially "
                "dependent distribution for any parameter.", 
                width=250).grid(row=6,rowspan=3,column=2,columnspan=2)

    tk.Message(nb.param_rules_frame, 
                text="Warning: Rules are applied "
                "from top to bottom. Order matters!", 
                width=250).grid(row=9,rowspan=3,column=2,columnspan=2)
    
    # These plots were previously attached to self.tab_inputs so that it was visible on all three IC tabs,
    # but it was hard to position them correctly.
    # Attaching to the Parameter Toolkit makes them easier to position
    nb.custom_param_fig = Figure(figsize=(0.25 * nb.APP_WIDTH / nb.APP_DPI, 0.25 * nb.APP_HEIGHT / nb.APP_DPI))
    nb.custom_param_subplot = nb.custom_param_fig.add_subplot(111)
    # Prevent coordinate values from appearing in the toolbar; this would sometimes jostle GUI elements around
    nb.custom_param_subplot.format_coord = lambda x, y: ""
    nb.custom_param_canvas = tkagg.FigureCanvasTkAgg(nb.custom_param_fig, 
                                                        master=nb.param_rules_frame)
    nb.custom_param_canvas.get_tk_widget().grid(row=12, column=0, columnspan=2)

    nb.custom_param_toolbar_frame = tk.ttk.Frame(master=nb.param_rules_frame)
    nb.custom_param_toolbar_frame.grid(row=13,column=0,columnspan=2)
    tkagg.NavigationToolbar2Tk(nb.custom_param_canvas, 
                                nb.custom_param_toolbar_frame)
    
    nb.recent_param_fig = Figure(figsize=(0.25 * nb.APP_WIDTH / nb.APP_DPI, 0.25 * nb.APP_HEIGHT / nb.APP_DPI))
    nb.recent_param_subplot = nb.recent_param_fig.add_subplot(111)
    nb.recent_param_subplot.format_coord = lambda x, y: ""
    nb.recent_param_canvas = tkagg.FigureCanvasTkAgg(nb.recent_param_fig, 
                                                        master=nb.param_rules_frame)
    nb.recent_param_canvas.get_tk_widget().grid(row=12,column=2,columnspan=2)

    nb.recent_param_toolbar_frame = tk.ttk.Frame(master=nb.param_rules_frame)
    nb.recent_param_toolbar_frame.grid(row=13,column=2,columnspan=2)
    tkagg.NavigationToolbar2Tk(nb.recent_param_canvas, 
                                nb.recent_param_toolbar_frame)

    tk.ttk.Button(nb.param_rules_frame, text="⇧", 
                    command=nb.moveup_paramrule).grid(row=1,column=4)

    tk.ttk.OptionMenu(nb.param_rules_frame, 
                        nb.paramtoolkit_viewer_selection, 
                        unitless_dropdown_list[0], 
                        *unitless_dropdown_list).grid(row=2,column=4)

    tk.ttk.Button(nb.param_rules_frame, 
                    text="Change view", 
                    command=nb.refresh_paramrule_listbox).grid(row=2,column=5)

    tk.ttk.Button(nb.param_rules_frame, text="⇩", 
                    command=nb.movedown_paramrule).grid(row=3,column=4)

    ## Param List Upload:

    nb.listupload_frame = tk.ttk.Frame(nb.tab_explicit_init)
    nb.listupload_frame.grid(row=0,column=0,padx=(440,0))

    tk.Message(nb.listupload_frame, 
                text="This tab provides an option "
                "to directly import a list of data points, "
                "on which the TED will do linear interpolation "
                "to fit to the specified space grid.", 
                width=360).grid(row=0,column=0)
    
    tk.ttk.OptionMenu(nb.listupload_frame, 
                        nb.listupload_var_selection, 
                        unitless_dropdown_list[0], 
                        *unitless_dropdown_list).grid(row=1,column=0)

    tk.ttk.Button(nb.listupload_frame, 
                    text="Import", 
                    command=nb.add_listupload).grid(row=2,column=0)
    
    nb.listupload_fig = Figure(figsize=(0.3 * nb.APP_WIDTH / nb.APP_DPI, 0.3 * nb.APP_HEIGHT / nb.APP_DPI))
    nb.listupload_subplot = nb.listupload_fig.add_subplot(111)
    nb.listupload_canvas = tkagg.FigureCanvasTkAgg(nb.listupload_fig, 
                                                        master=nb.listupload_frame)
    nb.listupload_canvas.get_tk_widget().grid(row=0, rowspan=3,column=1)
    
    nb.listupload_toolbar_frame = tk.ttk.Frame(master=nb.listupload_frame)
    nb.listupload_toolbar_frame.grid(row=3,column=1)
    tkagg.NavigationToolbar2Tk(nb.listupload_canvas, 
                                nb.listupload_toolbar_frame)

    ## Laser Generation Condition (LGC): extra input mtds for nanowire-specific applications
    if nb.module.is_LGC_eligible:
        create_LGC_frame(nb)
        nb.tab_inputs.add(nb.tab_generation_init, 
                            text="Laser Generation Conditions")
        
    # Attach sub-frames to input tab and input tab to overall notebook
    nb.tab_inputs.add(nb.tab_rules_init, text="Parameter Toolkit")
    nb.tab_inputs.add(nb.tab_explicit_init, text="Parameter List Upload")
    nb.notebook.add(nb.tab_inputs, text="Inputs")

    return nb

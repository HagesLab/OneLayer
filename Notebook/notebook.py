
import datetime
import os
import itertools
import numpy as np
import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk

# This lets us pass params to functions called by tkinter buttons
from functools import partial 

import carrier_excitations
from GUI_structs import Param_Rule
from GUI_structs import Flag
from GUI_structs import Batchable
from GUI_structs import Raw_Data_Set
from GUI_structs import Integrated_Data_Set
from GUI_structs import Analysis_Plot_State
from GUI_structs import Integration_Plot_State
from utils import to_index
from utils import to_pos
from utils import to_array
from utils import get_all_combinations
from utils import autoscale
from utils import new_integrate       
from io_utils import extract_values
from io_utils import u_read
from io_utils import check_valid_filename
from io_utils import get_split_and_clean_line

# from Notebook.tabs import add_tab_inputs
# from Notebook.tabs import add_tab_simulate
# from Notebook.tabs import add_tab_analyze
from Notebook.base import BaseNotebook



np.seterr(divide='raise', over='warn', under='warn', invalid='raise')
        
class Notebook(BaseNotebook):

    def __init__(self, title, module_list, cli_args):
        """ Create main tkinter object and select module. """
        self.module_list = module_list
        self.module = None
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', False)
        self.root.title(title)
        
        cli_module = cli_args.get("module")
        if cli_module is not None:
            self.module = self.module_list[cli_module]()
            self.module.verify()
            self.verified=True
        else:
            self.do_module_popup()
            self.root.wait_window(self.select_module_popup)

        if self.module is None: 
            return
        if not self.verified: 
            return
        self.prep_notebook()
        
        tab_id = cli_args.get("tab_id")
        if tab_id is not None:
            self.notebook.select(tab_id)
        return
        
    def prep_notebook(self):
        """ Attach elements to notebook based on selected module. """
        
        self.default_dirs = {"Initial":"Initial", "Data":"Data", "PL":"Analysis"}

        self.prepare_main_canvas()
        self.prepare_radiobuttons_and_checkboxes()
        self.prepare_initial_things()
        self.prepare_eligible_modules()
        self.add_menus()
        self.reset_popup_flags()

        self.root.config(menu=self.menu_bar)

        # Set a tkinter graphical theme
        s = ttk.Style()
        s.theme_use('classic')

        self.add_tab_inputs()
        self.add_tab_simulate()
        self.add_tab_analyze()

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_selected)
        #self.tab_inputs.bind("<<NotebookTabChanged>>", self.on_input_subtab_selected)
        
        # Initialize to default layer
        self.change_layer()

        self.test_initialization()


    def run(self):
        if self.module is None: 
            self.root.destroy()
            return

        width, height = self.root.winfo_screenwidth() * 0.8, self.root.winfo_screenheight() * 0.8

        self.root.geometry('%dx%d+0+0' % (width,height))
        self.root.attributes("-topmost", True)
        self.root.after_idle(self.root.attributes,'-topmost',False)
        self.root.mainloop()
        print("Closed TEDs")
        matplotlib.use(starting_backend)
        return

    def quit(self):
        self.do_confirmation_popup("All unsaved data will be lost. "
                                   "Are you sure you want to close TEDs?")
        self.root.wait_window(self.confirmation_popup)
        if not self.confirmed: 
            return
        self.root.destroy()
        print("Closed TEDs")
        matplotlib.use(starting_backend)
        return
    
    def toggle_fullscreen(self):
        self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen'))
        if self.root.attributes('-fullscreen'):
            # Hide scrollbars
            self.main_scroll_y.grid_remove()
            self.main_scroll_x.grid_remove()
        else:
            # Reveal scrollbars
            self.main_scroll_y.grid()
            self.main_scroll_x.grid()
        return

    def change_module(self):
        self.do_confirmation_popup("Warning: This will close the current instance "
                                   "of TEDs (and all unsaved data). Are you sure "
                                   "you want to select a new module?")
        self.root.wait_window(self.confirmation_popup)
        if not self.confirmed: 
            return
        
        self.notebook.destroy()
        self.do_module_popup()
        self.root.wait_window(self.select_module_popup)
        if self.module is None: 
            return
        self.prep_notebook()
        
        return


    def add_menus(self):
        """Adding menu_bar, file_menu, view_menu and tool_menu."""

         # Add menu bars
        self.menu_bar = tk.Menu(self.notebook)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        # TODO: Open file explorer instead of a file dialog
        self.file_menu.add_command(label="Manage Initial Condition Files", 
                                   command=partial(tk.filedialog.askopenfilenames, 
                                                   title="This window does not open anything - Use this window to move or delete IC files", 
                                                   initialdir=self.default_dirs["Initial"]))
        self.file_menu.add_command(label="Manage Data Files", 
                                   command=partial(tk.filedialog.askdirectory, 
                                                   title="This window does not open anything - Use this window to move or delete data files",
                                                   initialdir=self.default_dirs["Data"]))
        self.file_menu.add_command(label="Manage Export Files", 
                                   command=partial(tk.filedialog.askopenfilenames, 
                                                   title="This window does not open anything - Use this window to move or delete export files",
                                                   initialdir=self.default_dirs["PL"]))
        self.file_menu.add_command(label="Change Module", 
                                   command=self.change_module)
        self.file_menu.add_command(label="Exit", 
                                   command=self.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.view_menu.add_command(label="Toggle Fullscreen", 
                                   command=self.toggle_fullscreen)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)

        self.tool_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.tool_menu.add_command(label="Batch Op. Tool", 
                                   command=self.do_batch_popup)
        self.menu_bar.add_cascade(label="Tools", menu=self.tool_menu)

        #self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        #self.help_menu.add_command(label="About", command=self.do_about_popup)
        #self.menu_bar.add_cascade(label="Help", menu=self.help_menu)


    def test_initialization(self):
        """Check the existence of the necessary directories."""

        print("Initialization complete")
        print("Detecting Initial Condition and Data Directories...")
        try:
            os.mkdir(self.default_dirs["Initial"])
            print("No Initial Condition Directory detected; automatically creating...")
        except FileExistsError:
            print("Initial Condition Directory detected")
        
        try:
            os.mkdir(self.default_dirs["Data"])
            print("No Data Directory detected; automatically creating...")
        except FileExistsError:
            print("Data Directory detected")

        try:
            os.mkdir(self.default_dirs["PL"])
            print("No PL Directory detected; automatically creating...")
        except FileExistsError:
            print("PL Directory detected")
            
        print("Checking whether the current system class ({}) "
              "has a dedicated data subdirectory...".format(self.module.system_ID))
        try:
            os.mkdir(os.path.join(self.default_dirs["Data"], self.module.system_ID))
            print("No such subdirectory detected; automatically creating...")
        except FileExistsError:
            print("Subdirectory detected")


    def DEBUG(self):
        """ Print a custom message regarding the system state; 
            this changes often depending on what is being worked on
        """
        print(self.using_LGC)
        print(self.LGC_options)
        print(self.LGC_values)
        return
    
    def change_layer(self, clear=True, update_LGC_display=True):
        self.current_layer_name = self.current_layer_selection.get()
        
        current_layer = self.module.layers[self.current_layer_name]
        self.set_thickness_and_dx_entryboxes(state='unlock')
        if current_layer.spacegrid_is_set:
            self.enter(self.thickness_entry, str(current_layer.total_length))
            self.enter(self.dx_entry, str(current_layer.dx))
            self.set_thickness_and_dx_entryboxes(state='lock')
        else:
            if clear:
                self.enter(self.thickness_entry, "")
                self.enter(self.dx_entry, "")
            
        # Put new layer's params into param selection tabs
        var_dropdown_list = [str(param_name + param.units) 
                             for param_name, param in self.module.layers[self.current_layer_name].params.items()
                             if param.is_space_dependent]
        unitless_dropdown_list = [param_name 
                                  for param_name, param in self.module.layers[self.current_layer_name].params.items()
                                  if param.is_space_dependent]
        self.paramrule_var_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, 
                                                        self.init_var_selection, 
                                                        var_dropdown_list[0], 
                                                        *var_dropdown_list)
        self.paramrule_var_dropdown.grid(row=4,column=1)
        
        self.paramrule_viewer_dropdown = tk.ttk.OptionMenu(self.param_rules_frame, 
                                                           self.paramtoolkit_viewer_selection, 
                                                           unitless_dropdown_list[0], 
                                                           *unitless_dropdown_list)
        self.paramrule_viewer_dropdown.grid(row=2,column=4)
        self.update_paramrule_listbox(unitless_dropdown_list[0])
        
        
        self.listupload_dropdown = tk.ttk.OptionMenu(self.listupload_frame, 
                                                     self.listupload_var_selection, 
                                                     unitless_dropdown_list[0], 
                                                     *unitless_dropdown_list)
        self.listupload_dropdown.grid(row=1,column=0)
        
        
        self.write(self.ICtab_status, "Switched to layer:\n{}".format(self.current_layer_name))
        
        self.write(self.layer_statusbox, "On layer: {}".format(self.current_layer_name))
        
        if update_LGC_display:
            if (self.module.system_ID in self.LGC_eligible_modules and 
                self.using_LGC[self.current_layer_name]):
                for option, val in self.LGC_options[self.current_layer_name].items():
                    self.LGC_optionboxes[option].set(val)
                    
                for param_name, box in self.LGC_entryboxes_dict.items():
                    if param_name in self.LGC_values[self.current_layer_name]:
                        self.enter(box, str(self.LGC_values[self.current_layer_name][param_name]))
                    else:
                        self.enter(box, "")
                        
        
        return

    def update_system_summary(self):
        """ Transfer parameter values from the Initial Condition tab 
            to the summary popup windows.
        """
        if self.sys_printsummary_popup_isopen:
            self.write(self.printsummary_textbox, self.module.DEBUG_print())
            
        if self.sys_plotsummary_popup_isopen:
            set_layers = {name for name in self.module.layers 
                          if self.module.layers[name].spacegrid_is_set}
            for layer_name in set_layers:
                layer = self.module.layers[layer_name]
                for param_name in layer.params:
                    param = layer.params[param_name]
                    if param.is_space_dependent:
                        val = to_array(param.value, len(layer.grid_x_nodes), 
                                       param.is_edge)
                        grid_x = layer.grid_x_nodes if not param.is_edge else layer.grid_x_edges
                        self.sys_param_summaryplots[(layer_name,param_name)].plot(grid_x, val)
                        self.sys_param_summaryplots[(layer_name,param_name)].set_yscale(autoscale(val_array=val))
                
            self.plotsummary_fig.tight_layout()
            self.plotsummary_fig.canvas.draw()

        return
    
    ## Functions to create popups and manage
    def do_module_popup(self):
        """ Popup for selecting the active module (e.g. Nanowire) """
        self.select_module_popup = tk.Toplevel(self.root)
        tk.Label(self.select_module_popup, 
                 text="The following TEDs modules were found; "
                 "select one to continue: ").grid(row=0,column=0)
        
        self.modules_list = self.module_list
        self.module_names = list(self.modules_list.keys())
        self.module_listbox = tk.Listbox(self.select_module_popup, width=40, height=10)
        self.module_listbox.grid(row=1,column=0)
        self.module_listbox.delete(0,tk.END)
        self.module_listbox.insert(0,*(self.module_names))
        
        tk.Button(self.select_module_popup, text="Continue", 
                  command=partial(self.on_select_module_popup_close, True)).grid(row=2,column=0)
        
        self.select_module_popup.protocol("WM_DELETE_WINDOW", 
                                          partial(self.on_select_module_popup_close, 
                                                  continue_=False))
        self.select_module_popup.attributes("-topmost", True)
        self.select_module_popup.after_idle(self.select_module_popup.attributes,
                                            '-topmost',False)
        
        self.select_module_popup.grab_set()
        
        return
    
    def on_select_module_popup_close(self, continue_=False):
        """ Do basic verification checks defined by OneD_Model.verify() 
            and inform tkinter of selected module 
        """
        try:
            if continue_:
                self.verified=False
                self.module = self.modules_list[self.module_names[self.module_listbox.curselection()[0]]]()
                self.module.verify()
                self.verified=True
                
            self.select_module_popup.destroy()

        except IndexError:
            print("No module selected: Select a module from the list")
        except AssertionError as oops:
            print("Error: could not verify selected module")
            print(str(oops))
            
        return
        
    def do_confirmation_popup(self, text, hide_cancel=False):
        """ General purpose popup for important operations (e.g. deleting something)
            which should require user confirmation
        """
        self.confirmation_popup = tk.Toplevel(self.root)
        
        tk.Message(self.confirmation_popup, text=text, 
                   width=(float(self.root.winfo_screenwidth()) / 4)).grid(row=0,column=0, columnspan=2)
        
        if not hide_cancel:
            tk.Button(self.confirmation_popup, text="Cancel", 
                      command=partial(self.on_confirmation_popup_close, 
                                      continue_=False)).grid(row=1,column=0)
        
        tk.Button(self.confirmation_popup, text='Continue', 
                  command=partial(self.on_confirmation_popup_close, 
                                  continue_=True)).grid(row=1,column=1)
        
        self.confirmation_popup.protocol("WM_DELETE_WINDOW", 
                                         self.on_confirmation_popup_close)
        self.confirmation_popup.grab_set()        
        return
    
    def on_confirmation_popup_close(self, continue_=False):
        """ Inform caller of do_confirmation_popup of whether user confirmation 
            was received 
        """
        self.confirmed = continue_
        self.confirmation_popup.destroy()
        return
    
    
    def do_sys_printsummary_popup(self):
        """ Display as text the current space grid and parameters. """
        # Don't open more than one of this window at a time
        if not self.sys_printsummary_popup_isopen: 
            self.sys_printsummary_popup = tk.Toplevel(self.root)
            
            self.printsummary_textbox = tk.scrolledtext.ScrolledText(self.sys_printsummary_popup, 
                                                                    width=100,height=30)
            self.printsummary_textbox.grid(row=0,column=0,padx=(20,0), pady=(20,20))
            
            self.sys_printsummary_popup_isopen = True
            
            self.update_system_summary()
            
            self.sys_printsummary_popup.protocol("WM_DELETE_WINDOW", 
                                                 self.on_sys_printsummary_popup_close)
            return
        
    def on_sys_printsummary_popup_close(self):
        try:
            self.sys_printsummary_popup.destroy()
            self.sys_printsummary_popup_isopen = False
        except Exception:
            print("Error #2022: Failed to close shortcut popup.")
        return
    
    def do_sys_plotsummary_popup(self):
        """ Display as series of plots the current parameter distributions. """
        set_layers = {name for name in self.module.layers if self.module.layers[name].spacegrid_is_set}
        if not set_layers: 
            return
        
        if not self.sys_plotsummary_popup_isopen:
            self.sys_plotsummary_popup = tk.Toplevel(self.root)
            plot_count = sum([self.module.layers[layer_name].param_count
                              for layer_name in set_layers])
            count = 1
            rdim = np.floor(np.sqrt(plot_count))
            #rdim = 4
            cdim = np.ceil(plot_count / rdim)
            
            if self.sys_flag_dict['symmetric_system'].value():
                self.plotsummary_symmetriclabel = tk.Label(self.sys_plotsummary_popup, 
                                                           text="Note: All distributions "
                                                                "are symmetric about x=0")
                self.plotsummary_symmetriclabel.grid(row=0,column=0)

            self.plotsummary_fig = Figure(figsize=(20,10))
            self.sys_param_summaryplots = {}
            for layer_name in set_layers:
                layer = self.module.layers[layer_name]
                for param_name in layer.params:
                    if layer.params[param_name].is_space_dependent:
                        self.sys_param_summaryplots[(layer_name,param_name)] = self.plotsummary_fig.add_subplot(int(rdim), int(cdim), int(count))
                        self.sys_param_summaryplots[(layer_name,param_name)].set_title("{}-{} {}".format(layer_name, param_name,layer.params[param_name].units))
                        count += 1
            
            self.plotsummary_canvas = tkagg.FigureCanvasTkAgg(self.plotsummary_fig, 
                                                              master=self.sys_plotsummary_popup)
            self.plotsummary_plotwidget = self.plotsummary_canvas.get_tk_widget()
            self.plotsummary_plotwidget.grid(row=1,column=0)
            
            self.sys_plotsummary_popup_isopen = True
            self.update_system_summary()
            
            self.sys_plotsummary_popup.protocol("WM_DELETE_WINDOW", 
                                                self.on_sys_plotsummary_popup_close)
            ## Temporarily disable the main window while this popup is active
            self.sys_plotsummary_popup.grab_set()
            
            return
        
    def on_sys_plotsummary_popup_close(self):
        try:
            self.sys_plotsummary_popup.destroy()
            self.sys_plotsummary_popup_isopen = False
        except Exception:
            print("Error #2023: Failed to close plotsummary popup.")
        return
        
    def do_sys_param_shortcut_popup(self):
        """ Open a box for inputting (spatially constant) parameters. """ 
        if not self.sys_param_shortcut_popup_isopen:
            current_layer = self.module.layers[self.current_layer_name]
            try:
                self.set_init_x()
                assert current_layer.spacegrid_is_set, "Error: could not set space grid"
    
            except ValueError:
                self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
                return
    
            except (AssertionError, Exception) as oops:
                self.write(self.ICtab_status, oops)
                return
        
            self.sys_param_shortcut_popup = tk.Toplevel(self.root)
            
            tk.ttk.Label(self.sys_param_shortcut_popup, 
                         text="Parameter Short-cut Tool", 
                         style="Header.TLabel").grid(row=0,column=0)
            
            tk.Message(self.sys_param_shortcut_popup, 
                       text="Are the values of certain parameters "
                       "constant across the system? "
                       "Enter those values here and "
                       "press \"Continue\" to apply "
                       "them on all space grid points.", 
                       width=300).grid(row=0,column=1)
            
            self.sys_param_list_frame = tk.ttk.Frame(self.sys_param_shortcut_popup)
            self.sys_param_list_frame.grid(row=1,column=0,columnspan=2)
            
            self.sys_param_entryboxes_dict = {}
            self.sys_param_labels_dict = {}
            row_count = 0
            col_count = 0
            max_per_col = 6
            for param in current_layer.params:
                self.sys_param_labels_dict[param] = tk.ttk.Label(self.sys_param_list_frame, 
                                                                 text="{} {}".format(param, 
                                                                                     current_layer.params[param].units))
                self.sys_param_labels_dict[param].grid(row=row_count, 
                                                       column=col_count)
                self.sys_param_entryboxes_dict[param] = tk.ttk.Entry(self.sys_param_list_frame, 
                                                                     width=9)
                self.sys_param_entryboxes_dict[param].grid(row=row_count, 
                                                           column=col_count + 1)
                
                if isinstance(current_layer.params[param].value, (float, int)):
                    formatted_val = current_layer.params[param].value
                    if formatted_val > 1e4: 
                        formatted_val = "{:.3e}".format(formatted_val)
                    else: 
                        formatted_val = str(formatted_val)
                    
                    self.enter(self.sys_param_entryboxes_dict[param], formatted_val)
                
                else:
                    self.enter(self.sys_param_entryboxes_dict[param], "[list]")
                row_count += 1
                if row_count == max_per_col:
                    row_count = 0
                    col_count += 2
                    
            tk.Button(self.sys_param_shortcut_popup, 
                      text="Continue", 
                      command=partial(self.on_sys_param_shortcut_popup_close, 
                                      True)).grid(row=2,column=1)
            
            # FIXME: Temporary patch while we decide between surface recombination and seq-transfer models
            
            if self.module.system_ID == "MAPI_Rubrene" and self.current_layer_name == "Rubrene":
                if self.sys_flag_dict["do_sct"].value():
                    disable_these = ["St"]
                    
                else:
                    disable_these = ["Ssct", "Sp", "uc_permitivity", "W_VB", "mu_P_up"]
                    
                for param in disable_these:
                    self.enter(self.sys_param_entryboxes_dict[param], 0)
                    self.sys_param_entryboxes_dict[param].config(state='disabled')
                    
            self.sys_param_shortcut_popup.protocol("WM_DELETE_WINDOW", 
                                                   self.on_sys_param_shortcut_popup_close)
            self.sys_param_shortcut_popup_isopen = True
            ## Temporarily disable the main window while this popup is active
            self.sys_param_shortcut_popup.grab_set()
            
            return
                    
        else:
            print("Error #2020: Opened more than one sys param shortcut popup at a time")
            
    def on_sys_param_shortcut_popup_close(self, continue_=False):
        """ Transfer and store collected parameters. """ 
        try:
            if continue_:
                current_layer = self.module.layers[self.current_layer_name]
                changed_params = []
                err_msg = ["The following params were not updated:"]
                for param in current_layer.params:
                    val = self.sys_param_entryboxes_dict[param].get()
                    if not val: 
                        continue
                    else:
                        try:
                            val = float(val)
                            minimum = current_layer.params[param].valid_range[0]
                            maximum = current_layer.params[param].valid_range[1]
                            assert (val >= minimum), "Error: min value for {} is {} but {} was entered".format(param, minimum, val)
                            assert (val <= maximum), "Error: max value for {} is {} but {} was entered".format(param, maximum, val)
                            
                        except Exception as e:
                            err_msg.append("{}: {}".format(param, str(e)))
                            continue
                    
                    self.paramtoolkit_currentparam = param
                    self.deleteall_paramrule()
                    current_layer.params[param].value = val
                    changed_params.append(param)
                    
                if changed_params:
                    self.update_IC_plot(plot_ID="recent")
                    self.do_confirmation_popup("Updated: {}".format(changed_params), 
                                               hide_cancel=True)
                    self.root.wait_window(self.confirmation_popup)
                    
                if "delta_N" in changed_params or "delta_P" in changed_params:
                    self.using_LGC[self.current_layer_name] = False
                    
                if len(err_msg) > 1:
                    self.do_confirmation_popup("\n".join(err_msg), hide_cancel=True)
                    self.root.wait_window(self.confirmation_popup)
                    
                    
            self.write(self.ICtab_status, "")
            self.sys_param_shortcut_popup.destroy()
            self.sys_param_shortcut_popup_isopen = False
        except Exception as e:
            print("Error #2021: Failed to close shortcut popup.")
            print(e)
        
        return

    def do_batch_popup(self):
        """ Open tool for making batches of similar initial condition files. """
        if not self.batch_popup_isopen:
            
            for layer_name, layer in self.module.layers.items():
                try:
                    assert layer.spacegrid_is_set
                
                except Exception:
                    self.write(self.ICtab_status, 
                               "Error: layer {} does not have params initialized yet".format(layer_name))
                    return
         
            max_batchable_params = 4
            self.batch_param = tk.StringVar()

            self.batch_popup = tk.Toplevel(self.root)
            
            tk.ttk.Label(self.batch_popup, 
                         text="Batch IC Tool", 
                         style="Header.TLabel").grid(row=0,column=0)
            
            tk.Message(self.batch_popup, 
                       text="This Batch Tool allows you "
                       "to generate many copies of "
                       "the currently-loaded IC, "
                       "varying up to {} parameters "
                       "between all of them.".format(max_batchable_params), 
                       width=300).grid(row=1,column=0)

            tk.Message(self.batch_popup, 
                       text="All copies will be stored "
                       "in a new folder with the name "
                       "you enter into the appropriate box.", 
                       width=300).grid(row=2,column=0)

            tk.Message(self.batch_popup, 
                       text="For best results, load a "
                       "complete IC file or fill "
                       "in values for all params "
                       "before using this tool.", 
                       width=300).grid(row=3,column=0)

            tk.ttk.Label(self.batch_popup, 
                         text="Select Layer {} Batch Parameter:".format(self.current_layer_name)).grid(row=0,column=1)
            
            self.batch_entry_frame = tk.ttk.Frame(self.batch_popup)
            self.batch_entry_frame.grid(row=1,column=1,columnspan=3, rowspan=3)
           
            # Contextually-dependent options for batchable params
            self.batchables_array = []
            LGC_active = []
            batchable_params = []
            for layer in self.module.layers:
                LGC_active.append(self.module.system_ID in self.LGC_eligible_modules 
                                      and self.using_LGC[layer])
                for param in self.module.layers[layer].params:
                    
                    if not (LGC_active[-1] and (param == "delta_N" or param == "delta_P")):
                        batchable_params.append("{}-{}".format(layer, param))
                        
                if LGC_active[-1]:
                    for param in self.LGC_values[layer]:
                        batchable_params.append("{}-{}".format(layer, param))
            
            if any(LGC_active):
                
                self.LGC_instruction1 = tk.Message(self.batch_popup, 
                                                   text="Additional options for generating "
                                                        "delta_N and delta_P batches "
                                                        "are available when using the "
                                                        "Laser Generation Condition tool.", 
                                                   width=300)
                self.LGC_instruction1.grid(row=4,column=0)
                
                self.LGC_instruction2 = tk.Message(self.batch_popup, 
                                                   text="Please note that TEDs will "
                                                   "use the current values and settings on "
                                                   "the L.G.C. tool's tab "
                                                   "to complete the batches when one "
                                                   "or more of these options are selected.", 
                                                   width=300)
                self.LGC_instruction2.grid(row=5,column=0)
                
                # Boolean logic is fun
                # The main idea is to hide certain parameters based on which options were used to construct the LGC
                # LGC_params = [key for key in self.LGC_entryboxes_dict.keys() if not (
                #             (self.check_calculate_init_material_expfactor.get() and (key == "LGC_absorption_cof")) or
                #             (not self.check_calculate_init_material_expfactor.get() and (key == "A0" or key == "Eg")) or
                #             (self.LGC_gen_power_mode.get() == "power-spot" and (key == "Power_Density" or key == "Max_Gen" or key == "Total_Gen")) or
                #             (self.LGC_gen_power_mode.get() == "density" and (key == "Power" or key == "Spotsize" or key == "Max_Gen" or key == "Total_Gen")) or
                #             (self.LGC_gen_power_mode.get() == "max-gen" and (key == "Power" or key == "Spotsize" or key == "Power_Density" or key == "Total_Gen")) or
                #             (self.LGC_gen_power_mode.get() == "total-gen" and (key == "Power_Density" or key == "Power" or key == "Spotsize" or key == "Max_Gen"))
                #             )]

            for i in range(max_batchable_params):
                batch_param_name = tk.StringVar()
                optionmenu = tk.ttk.OptionMenu(self.batch_entry_frame, 
                                               batch_param_name, "", "", 
                                               *batchable_params)
                
                optionmenu.grid(row=i,column=0,padx=(20,20))
                batch_param_entry = tk.ttk.Entry(self.batch_entry_frame, width=80)
                batch_param_entry.grid(row=i,column=1,columnspan=2)
                
                if i == 0: 
                    self.enter(batch_param_entry, 
                               "Enter a list of space-separated "
                               "values for the selected Batch Parameter")
                
                self.batchables_array.append(Batchable(optionmenu, 
                                                       batch_param_entry, 
                                                       batch_param_name))
                    
            self.batch_status = tk.Text(self.batch_popup, width=30,height=3)
            self.batch_status.grid(row=6,column=0)
            self.batch_status.configure(state='disabled')

            self.batch_name_entry = tk.ttk.Entry(self.batch_popup, width=24)
            self.enter(self.batch_name_entry, "Enter name for batch folder")
            self.batch_name_entry.grid(row=6,column=1)

            tk.ttk.Button(self.batch_popup, 
                          text="Create Batch", 
                          command=self.create_batch_init).grid(row=6,column=2)

            self.batch_popup.protocol("WM_DELETE_WINDOW", self.on_batch_popup_close)
            self.batch_popup.grab_set()
            self.batch_popup_isopen = True

        else:
            print("Error #102: Opened more than one batch popup at a time")
        return

    def on_batch_popup_close(self):
        try:
            self.batch_popup.destroy()
            print("Batch popup closed")
            self.batch_popup_isopen = False
        except Exception:
            print("Error #103: Failed to close batch popup.")

        return

    def do_resetIC_popup(self):
        """ Clear stored parameter values. """
        if not self.resetIC_popup_isopen:

            self.resetIC_popup = tk.Toplevel(self.root)

            tk.ttk.Label(self.resetIC_popup, 
                         text="Which Parameters "
                         "should be cleared?", 
                         style="Header.TLabel").grid(row=0,column=0,columnspan=2)

            self.resetIC_checkbutton_frame = tk.ttk.Frame(self.resetIC_popup)
            self.resetIC_checkbutton_frame.grid(row=1,column=0,columnspan=2)

            # Let's try some procedurally generated checkbuttons: 
            # one created automatically per layer
            self.resetIC_checklayers = {}
            self.resetIC_checkbuttons = {}

            for layer_name in self.module.layers:
                self.resetIC_checklayers[layer_name] = tk.IntVar()

                self.resetIC_checkbuttons[layer_name] = tk.ttk.Checkbutton(self.resetIC_checkbutton_frame, 
                                                                           text=layer_name, 
                                                                           variable=self.resetIC_checklayers[layer_name], 
                                                                           onvalue=1, 
                                                                           offvalue=0)

            for i, cb in enumerate(self.resetIC_checkbuttons):
                self.resetIC_checkbuttons[cb].grid(row=i,column=0, pady=(6,6))

            
            tk.ttk.Separator(self.resetIC_popup, 
                             orient="horizontal", 
                             style="Grey Bar.TSeparator").grid(row=2,column=0,columnspan=2, pady=(10,10), sticky="ew")

            self.resetIC_check_clearall = tk.IntVar()
            tk.Checkbutton(self.resetIC_popup, 
                           text="Clear All", 
                           variable=self.resetIC_check_clearall, 
                           onvalue=1, offvalue=0).grid(row=3,column=0)

            tk.Button(self.resetIC_popup, 
                      text="Continue", 
                      command=partial(self.on_resetIC_popup_close, 
                                      True)).grid(row=3,column=1)

            self.resetIC_popup.protocol("WM_DELETE_WINDOW", 
                                        self.on_resetIC_popup_close)
            self.resetIC_popup.grab_set()
            self.resetIC_popup_isopen = True
            return

        else:
            print("Error #700: Opened more than one resetIC popup at a time")

        return

    def on_resetIC_popup_close(self, continue_=False):
        try:
            self.resetIC_selected_layers = []
            self.resetIC_do_clearall = False
            if continue_:
                self.resetIC_do_clearall = self.resetIC_check_clearall.get()
                if self.resetIC_do_clearall:
                    self.resetIC_selected_layers = list(self.resetIC_checklayers.keys())
                else:
                    self.resetIC_selected_layers = [layer_name for layer_name in self.resetIC_checklayers 
                                                    if self.resetIC_checklayers[layer_name].get()]

            self.resetIC_popup.destroy()
            print("resetIC popup closed")
            self.resetIC_popup_isopen = False

        except Exception:
            print("Error #601: Failed to close Bayesim popup")
        return

    def do_plotter_popup(self, plot_ID):
        """ Select datasets for plotting on Analyze tab. """
        if not self.plotter_popup_isopen:

            self.plotter_popup = tk.Toplevel(self.root)

            tk.ttk.Label(self.plotter_popup, 
                         text="Select simulated data", 
                         style="Header.TLabel").grid(row=0,column=0,columnspan=2)

            def sims_sort_alpha():
                self.data_listbox.delete(0,tk.END)
                self.data_list = [file for file in os.listdir(os.path.join(self.default_dirs["Data"], self.module.system_ID)) 
                                  if not file.endswith(".txt")]
                
                self.data_listbox.insert(0,*(self.data_list))
                return
                
            def sims_sort_recent():
                self.data_listbox.delete(0,tk.END)
                self.data_list = [file for file in os.listdir(os.path.join(self.default_dirs["Data"], self.module.system_ID)) 
                                  if not file.endswith(".txt")]
                
                self.data_list = sorted(self.data_list, key=lambda file: os.path.getmtime(os.path.join(self.default_dirs["Data"], self.module.system_ID, file)), reverse=True)
                self.data_listbox.insert(0,*(self.data_list))
                return
            
            sorting_frame = tk.Frame(self.plotter_popup)
            sorting_frame.grid(row=1,column=0)
            
            tk.Button(sorting_frame, text='A-Z',
                      command=sims_sort_alpha).grid(row=0,column=0)
            
            tk.Button(sorting_frame, text='Recent First',
                      command=sims_sort_recent).grid(row=0,column=1)
            
            
            data_listbox_frame = tk.Frame(self.plotter_popup)
            data_listbox_frame.grid(row=2,column=0)
            
            self.data_listbox = tk.Listbox(data_listbox_frame, width=60, 
                                           height=40, 
                                           selectmode="extended")
            self.data_listbox.grid(row=0,column=0)
            

            data_listbox_scrollbar = tk.ttk.Scrollbar(data_listbox_frame, orient="vertical",
                                                           command=self.data_listbox.yview)
            data_listbox_scrollbar.grid(row=0,column=1, sticky='ns')
            
            self.data_listbox.config(yscrollcommand=data_listbox_scrollbar.set)
            sims_sort_alpha()
            
            plotter_options_frame = tk.Frame(self.plotter_popup)
            plotter_options_frame.grid(row=2,column=1)
            all_outputs = []
            for layer_name, layer in self.module.layers.items():
                all_outputs += [output for output in layer.outputs if layer.outputs[output].analysis_plotable]
            tk.OptionMenu(plotter_options_frame, self.data_var, 
                          *all_outputs).grid(row=0,column=0)

            tk.Checkbutton(plotter_options_frame, text="Auto-integrate", 
                           variable=self.check_autointegrate, 
                           onvalue=1, offvalue=0).grid(row=0,column=1)
            
            tk.Button(plotter_options_frame, text="Continue", 
                      command=partial(self.on_plotter_popup_close, 
                                      plot_ID, continue_=True)).grid(row=1,column=0,columnspan=2)

            self.plotter_status = tk.Text(plotter_options_frame, width=24,height=2)
            self.plotter_status.grid(row=2,column=0,columnspan=2, padx=(20,20))
            self.plotter_status.configure(state="disabled")

            self.plotter_popup.protocol("WM_DELETE_WINDOW", partial(self.on_plotter_popup_close, 
                                                                    plot_ID, continue_=False))
            self.plotter_popup.grab_set()
            self.plotter_popup_isopen = True

        else:
            print("Error #501: Opened more than one plotter popup at a time")
        return

    def on_plotter_popup_close(self, plot_ID, continue_=False):
        try:
			# There are two ways for a popup to close: by the user pressing "Continue" or the user cancelling or pressing "X"
			# We only interpret the input on the popup if the user wants to continue
            self.confirmed = continue_
            if continue_:
                assert (self.data_var.get()), "Select a data type from the drop-down menu"
                self.analysis_plots[plot_ID].data_filenames = []
                # This year for Christmas, I want Santa to implement tk.filedialog.askdirectories() so we can select multiple directories like we can do with files
                dir_names = [self.data_list[i] for i in self.data_listbox.curselection()]
                for next_dir in dir_names:
                    self.analysis_plots[plot_ID].data_filenames.append(next_dir)

                #self.analysis_plots[plot_ID].remove_duplicate_filenames()
                
                assert self.analysis_plots[plot_ID].data_filenames, "Select data files"

            self.plotter_popup.destroy()
            print("Plotter popup closed")
            self.plotter_popup_isopen = False

        except AssertionError as oops:
            self.write(self.plotter_status, str(oops))
        except Exception:
            print("Error #502: Failed to close plotter popup.")

        return

    def do_integration_timemode_popup(self):
        """ Select which timesteps to integrate through. """
        if not self.integration_popup_isopen:
            self.integration_popup = tk.Toplevel(self.root)

            tk.ttk.Label(self.integration_popup, 
                         text="Select which time steps "
                         "to integrate over", 
                         style="Header.TLabel").grid(row=1,column=0,columnspan=3)

            tk.ttk.Radiobutton(self.integration_popup, 
                               variable=self.fetch_PLmode, 
                               value='All time steps').grid(row=2,column=0)

            tk.Label(self.integration_popup, 
                     text="All time steps").grid(row=2,column=1)

            tk.ttk.Radiobutton(self.integration_popup, 
                               variable=self.fetch_PLmode, 
                               value='Current time step').grid(row=3,column=0)

            tk.Label(self.integration_popup, 
                     text="Current time step").grid(row=3,column=1)

            tk.Button(self.integration_popup, text="Continue", 
                      command=partial(self.on_integration_popup_close, 
                                      continue_=True)).grid(row=4,column=0,columnspan=3)

            self.integration_popup.protocol("WM_DELETE_WINDOW", partial(self.on_integration_popup_close, 
                                                                        continue_=False))
            self.integration_popup.grab_set()
            self.integration_popup_isopen = True
            
        else:
            print("Error #420: Opened more than one integration popup at a time")
        return

    def on_integration_popup_close(self, continue_=False):
        try:
            if continue_:
                self.PL_mode = self.fetch_PLmode.get()
            else:
                self.PL_mode = ""

            self.integration_popup.destroy()
            print("Integration popup closed")
            self.integration_popup_isopen = False
        except Exception:
            print("Error #421: Failed to close PLmode popup.")

        return

    def do_integration_getbounds_popup(self):
        """ Select spatial region to integrate through. """
        if not self.integration_getbounds_popup_isopen:
            plot_ID = self.active_analysisplot_ID.get()
            active_plot = self.analysis_plots[plot_ID]
            datatype = active_plot.datagroup.type
            where_layer = self.module.find_layer(datatype)
            
            self.integration_getbounds_popup = tk.Toplevel(self.root)

            tk.ttk.Radiobutton(self.integration_getbounds_popup, 
                               variable=self.fetch_intg_mode, 
                               value='single').grid(row=0,column=0, rowspan=3)

            tk.ttk.Label(self.integration_getbounds_popup, 
                         text="Single integral", 
                         style="Header.TLabel").grid(row=0,column=1, rowspan=3, padx=(0,20))

            tk.Label(self.integration_getbounds_popup, 
                     text="Enter bounds of integration " 
                     + self.module.layers[where_layer].length_unit).grid(row=0,column=2,columnspan=4)

            tk.Label(self.integration_getbounds_popup, 
                     text="Lower bound: x=").grid(row=1,column=2)

            self.integration_lbound_entry = tk.Entry(self.integration_getbounds_popup, 
                                                     width=9)
            self.integration_lbound_entry.grid(row=2,column=2)

            tk.Label(self.integration_getbounds_popup, 
                     text="Upper bound: x=").grid(row=1,column=5)

            self.integration_ubound_entry = tk.Entry(self.integration_getbounds_popup, 
                                                     width=9)
            self.integration_ubound_entry.grid(row=2,column=5)

            tk.ttk.Separator(self.integration_getbounds_popup, 
                             orient="horizontal", 
                             style="Grey Bar.TSeparator").grid(row=3,column=0,columnspan=30, pady=(10,10), sticky="ew")

            tk.ttk.Radiobutton(self.integration_getbounds_popup, 
                               variable=self.fetch_intg_mode, 
                               value='multiple').grid(row=4,column=0, rowspan=3)

            tk.ttk.Label(self.integration_getbounds_popup, 
                         text="Multiple integrals", 
                         style="Header.TLabel").grid(row=4,column=1, rowspan=3, padx=(0,20))

            tk.Label(self.integration_getbounds_popup, 
                     text="Enter space-separated "
                     "e.g. (100 200 300...) Centers "
                     "{}: ".format(self.module.layers[where_layer].length_unit)).grid(row=5,column=2)

            self.integration_center_entry = tk.Entry(self.integration_getbounds_popup, 
                                                     width=30)
            self.integration_center_entry.grid(row=5,column=3,columnspan=3)

            tk.Label(self.integration_getbounds_popup, 
                     text="Width {}: +/- ".format(self.module.layers[where_layer].length_unit)).grid(row=6,column=2)

            self.integration_width_entry = tk.Entry(self.integration_getbounds_popup, 
                                                    width=9)
            self.integration_width_entry.grid(row=6,column=3)

            tk.ttk.Separator(self.integration_getbounds_popup, 
                             orient="horizontal", 
                             style="Grey Bar.TSeparator").grid(row=7,column=0,columnspan=30, pady=(10,10), sticky="ew")

            tk.Button(self.integration_getbounds_popup, text="Continue", 
                      command=partial(self.on_integration_getbounds_popup_close, 
                                      continue_=True)).grid(row=8,column=5)

            self.integration_getbounds_status = tk.Text(self.integration_getbounds_popup, 
                                                        width=24,height=2)
            self.integration_getbounds_status.grid(row=8,rowspan=2,column=0,columnspan=5)
            self.integration_getbounds_status.configure(state="disabled")

            self.integration_getbounds_popup.protocol("WM_DELETE_WINDOW", 
                                                      partial(self.on_integration_getbounds_popup_close, 
                                                              continue_=False))
            self.integration_getbounds_popup.grab_set()
            self.integration_getbounds_popup_isopen = True
        else:
            print("Error #422: Opened more than one integration getbounds popup at a time")
        return

    def on_integration_getbounds_popup_close(self, continue_=False):
        """ Read in the pairs of integration bounds as-is. """
        # Checking if they make sense is do_Integrate()'s job
        try:
            self.confirmed = continue_
            if continue_:
                self.integration_bounds = []
                if self.fetch_intg_mode.get() == "single":
                    print("Single integral")
                    lbound = float(self.integration_lbound_entry.get())
                    ubound = float(self.integration_ubound_entry.get())
                    if (lbound > ubound):
                        raise KeyError("Error: upper bound too small")
                        
                    if (lbound < 0 and ubound < 0):
                        raise KeyError("Error: bounds out of range")

                    self.integration_bounds.append([lbound, ubound])
                    

                elif self.fetch_intg_mode.get() == "multiple":
                    print("Multiple integrals")
                    if self.integration_center_entry.get() == "Aboma":
                        centers = [0,1600,3600,5000,6400,8000,14200]
                        #centers = [0,2200,3400,5200,6400,7200,8600,10000]
                    else:
                        centers = list(set(extract_values(self.integration_center_entry.get(), ' ')))

                    width = float(self.integration_width_entry.get())

                    if width < 0: 
                        raise KeyError("Error: width must be non-negative")

                    for center in centers:
                        if center < 0:
                            raise KeyError("Error: center {} is out of range".format(center))
                        else:
                            self.integration_bounds.append([center - width, center + width])

                else:
                    raise KeyError("Select \"Single\" or \"Multiple\"")

                print("Over: {}".format(self.integration_bounds))

            else:
                self.write(self.analysis_status, "Integration cancelled")

            self.integration_getbounds_popup.destroy()
            print("PL getbounds popup closed")
            self.integration_getbounds_popup_isopen = False

        except (OSError, KeyError) as uh_oh:
            self.write(self.integration_getbounds_status, uh_oh)

        except Exception:
            self.write(self.integration_getbounds_status, "Error: missing or invalid paramters")

        return

    def do_PL_xaxis_popup(self):
        """ If integrating over single timestep, 
            select which parameter to plot as horizontal axis. 
        """
        if not self.PL_xaxis_popup_isopen:
            self.xaxis_param = ""
            self.xaxis_selection = tk.StringVar()
            self.PL_xaxis_popup = tk.Toplevel(self.root)

            tk.ttk.Label(self.PL_xaxis_popup, text="Select parameter for x axis", 
                         style="Header.TLabel").grid(row=0,column=0,columnspan=3)


            plot_ID = self.active_analysisplot_ID.get()
            active_plot = self.analysis_plots[plot_ID]
            datatype = active_plot.datagroup.type
            where_layer = self.module.find_layer(datatype)
            tk.OptionMenu(self.PL_xaxis_popup, 
                          self.xaxis_selection, 
                          *[param for param in self.module.layers[where_layer].params]).grid(row=1,column=1)

            tk.Button(self.PL_xaxis_popup, text="Continue", 
                      command=partial(self.on_PL_xaxis_popup_close, 
                                      continue_=True)).grid(row=1,column=2)

            self.PL_xaxis_status = tk.Text(self.PL_xaxis_popup, width=24,height=2)
            self.PL_xaxis_status.grid(row=2,rowspan=2,column=0,columnspan=3)
            self.PL_xaxis_status.configure(state="disabled")

            self.PL_xaxis_popup.protocol("WM_DELETE_WINDOW", 
                                         partial(self.on_PL_xaxis_popup_close, 
                                                 continue_=False))
            self.PL_xaxis_popup.grab_set()
            self.PL_xaxis_popup_isopen = True
        else:
            print("Error #424: Opened more than one PL xaxis popup at a time")
        return

    def on_PL_xaxis_popup_close(self, continue_=False):
        try:
            if continue_:
                self.xaxis_param = self.xaxis_selection.get()
                if not self.xaxis_param:
                    self.write(self.PL_xaxis_status, "Select a parameter")
                    return
            self.PL_xaxis_popup.destroy()
            print("PL xaxis popup closed")
            self.PL_xaxis_popup_isopen = False
        except Exception:
            print("Error #425: Failed to close PL xaxis popup.")

        return
    
    def do_change_axis_popup(self, from_integration):
        """ Select new axis parameters for analysis plots. """
        # Don't open if no data plotted
        if from_integration:
            plot_ID = self.active_integrationplot_ID.get()
            if self.integration_plots[plot_ID].datagroup.size() == 0: 
                return

        else:
            plot_ID = self.active_analysisplot_ID.get()
            if self.analysis_plots[plot_ID].datagroup.size() == 0: 
                return

        if not self.change_axis_popup_isopen:
            self.change_axis_popup = tk.Toplevel(self.root)

            tk.ttk.Label(self.change_axis_popup, 
                         text="Select axis settings", 
                         style="Header.TLabel").grid(row=0,column=0,columnspan=2)

            self.xframe = tk.Frame(master=self.change_axis_popup)
            self.xframe.grid(row=1,column=0,padx=(0,20),pady=(20,0))

            tk.Label(self.xframe, text="X Axis").grid(row=0,column=0,columnspan=2)

            tk.ttk.Radiobutton(self.xframe, variable=self.xaxis_type, 
                               value='linear').grid(row=1,column=0)

            tk.Label(self.xframe, text="Linear").grid(row=1,column=1)

            tk.ttk.Radiobutton(self.xframe, variable=self.xaxis_type, 
                               value='symlog').grid(row=2,column=0)

            tk.Label(self.xframe, text="Log").grid(row=2,column=1)

            tk.Label(self.xframe, text="Lower").grid(row=3,column=0)

            self.xlbound = tk.Entry(self.xframe, width=9)
            self.xlbound.grid(row=3,column=1)

            tk.Label(self.xframe, text="Upper").grid(row=4,column=0)

            self.xubound = tk.Entry(self.xframe, width=9)
            self.xubound.grid(row=4,column=1)

            self.yframe = tk.Frame(master=self.change_axis_popup)
            self.yframe.grid(row=1,column=1,padx=(0,20),pady=(20,0))

            tk.Label(self.yframe, text="Y Axis").grid(row=0,column=0,columnspan=2)

            tk.ttk.Radiobutton(self.yframe, variable=self.yaxis_type, 
                               value='linear').grid(row=1,column=0)

            tk.Label(self.yframe, text="Linear").grid(row=1,column=1)

            tk.ttk.Radiobutton(self.yframe, variable=self.yaxis_type, 
                               value='symlog').grid(row=2,column=0)

            tk.Label(self.yframe, text="Log").grid(row=2,column=1)

            tk.Label(self.yframe, text="Lower").grid(row=3,column=0)

            self.ylbound = tk.Entry(self.yframe, width=9)
            self.ylbound.grid(row=3,column=1)

            tk.Label(self.yframe, text="Upper").grid(row=4,column=0)

            self.yubound = tk.Entry(self.yframe, width=9)
            self.yubound.grid(row=4,column=1)

            tk.Checkbutton(self.change_axis_popup, 
                           text="Display legend?", 
                           variable=self.check_display_legend, 
                           onvalue=1, offvalue=0).grid(row=2,column=0,columnspan=2)
            
            tk.Checkbutton(self.change_axis_popup, 
                           text="Freeze axes?", 
                           variable=self.check_freeze_axes, 
                           onvalue=1, offvalue=0).grid(row=3,column=0,columnspan=2)

            tk.Button(self.change_axis_popup, text="Continue", 
                      command=partial(self.on_change_axis_popup_close, 
                                      from_integration, continue_=True)).grid(row=4, column=0,columnspan=2)

            self.change_axis_status = tk.Text(self.change_axis_popup, width=24,height=2)
            self.change_axis_status.grid(row=5,rowspan=2,column=0,columnspan=2)
            self.change_axis_status.configure(state="disabled")

            # Set the default values in the entry boxes to be the current options of the plot (in case the user only wants to make a few changes)
            if not (from_integration):
                active_plot = self.analysis_plots[plot_ID]

            else:
                active_plot = self.integration_plots[plot_ID]

            self.enter(self.xlbound, active_plot.xlim[0])
            self.enter(self.xubound, active_plot.xlim[1])
            self.enter(self.ylbound, "{:.2e}".format(active_plot.ylim[0]))
            self.enter(self.yubound, "{:.2e}".format(active_plot.ylim[1]))
            self.xaxis_type.set(active_plot.xaxis_type)
            self.yaxis_type.set(active_plot.yaxis_type)
            self.check_display_legend.set(active_plot.display_legend)
            self.check_freeze_axes.set(active_plot.do_freeze_axes)

            self.change_axis_popup.protocol("WM_DELETE_WINDOW", partial(self.on_change_axis_popup_close, 
                                                                        from_integration, 
                                                                        continue_=False))
            self.change_axis_popup.grab_set()
            self.change_axis_popup_isopen = True
        else:
            print("Error #440: Opened more than one change axis popup at a time")
        return

    def on_change_axis_popup_close(self, from_integration, continue_=False):
        try:
            if continue_:
                assert self.xaxis_type and self.yaxis_type, "Error: invalid axis type"
                assert (self.xlbound.get()
                        and self.xubound.get() 
                        and self.ylbound.get()
                        and self.yubound.get()), "Error: missing bounds"
                bounds = [float(self.xlbound.get()), float(self.xubound.get()), 
                          float(self.ylbound.get()), float(self.yubound.get())]
            
                if not (from_integration):
                    plot_ID = self.active_analysisplot_ID.get()
                    plot = self.analysis_plots[plot_ID].plot_obj
                    
                else:
                    plot_ID = self.active_integrationplot_ID.get()
                    plot = self.integration_plots[plot_ID].plot_obj

                # Set plot axis params and save in corresponding plot state object, 
                # if the selected plot has such an object
                plot.set_yscale(self.yaxis_type.get())
                plot.set_xscale(self.xaxis_type.get())

                plot.set_ylim(bounds[2], bounds[3])
                plot.set_xlim(bounds[0], bounds[1])

                if self.check_display_legend.get():
                    plot.legend().set_draggable(True)
                    
                else:
                    plot.legend('', frameon=False)

                if not (from_integration):
                    self.analyze_fig.tight_layout()
                    self.analyze_fig.canvas.draw()
                    
                else:
                    self.integration_fig.tight_layout()
                    self.integration_fig.canvas.draw()

                # Save these params to pre-populate the popup the next time it's opened
                if not (from_integration):
                    self.analysis_plots[plot_ID].yaxis_type = self.yaxis_type.get()
                    self.analysis_plots[plot_ID].xaxis_type = self.xaxis_type.get()
                    self.analysis_plots[plot_ID].ylim = (bounds[2], bounds[3])
                    self.analysis_plots[plot_ID].xlim = (bounds[0], bounds[1])
                    self.analysis_plots[plot_ID].display_legend = self.check_display_legend.get()
                    self.analysis_plots[plot_ID].do_freeze_axes = self.check_freeze_axes.get()
                else:
                    self.integration_plots[plot_ID].yaxis_type = self.yaxis_type.get()
                    self.integration_plots[plot_ID].xaxis_type = self.xaxis_type.get()
                    self.integration_plots[plot_ID].ylim = (bounds[2], bounds[3])
                    self.integration_plots[plot_ID].xlim = (bounds[0], bounds[1])
                    self.integration_plots[plot_ID].display_legend = self.check_display_legend.get()

            self.change_axis_popup.destroy()

            self.change_axis_popup_isopen = False

        except (ValueError, AssertionError) as oops:
            self.write(self.change_axis_status, oops)
            return
        except Exception:
            print("Error #441: Failed to close change axis popup.")

        return

    def do_IC_carry_popup(self):
        """ Open a tool to regenerate IC files based on current state of analysis plots. """
        plot_ID = self.active_analysisplot_ID.get()

        if not self.analysis_plots[plot_ID].datagroup.size(): 
            return

        if not self.IC_carry_popup_isopen:
            self.IC_carry_popup = tk.Toplevel(self.root)

            tk.ttk.Label(self.IC_carry_popup, 
                         text="Select data to include in new IC",
                         style="Header.TLabel").grid(row=0,column=0,columnspan=2)
            
            self.carry_checkbuttons = {}
            rcount = 1
            for layer_name, layer in self.module.layers.items():
                self.carry_checkbuttons[layer_name] = {}
                for var in layer.s_outputs:
                    self.carry_checkbuttons[layer_name][var] = tk.Checkbutton(self.IC_carry_popup, 
                                                                              text=var, 
                                                                              variable=self.carryover_include_flags[layer_name][var])
                    self.carry_checkbuttons[layer_name][var].grid(row=rcount, column=0)
                    rcount += 1

            self.carry_IC_listbox = tk.Listbox(self.IC_carry_popup, width=30,height=10, 
                                               selectmode='extended')
            self.carry_IC_listbox.grid(row=4,column=0,columnspan=2)
            for key, dataset in self.analysis_plots[plot_ID].datagroup.datasets.items():
                over_time = (self.analysis_plots[plot_ID].time > dataset.total_time)
                if over_time: continue
                self.carry_IC_listbox.insert(tk.END, key)

            tk.Button(self.IC_carry_popup, text="Continue", 
                      command=partial(self.on_IC_carry_popup_close,
                                      continue_=True)).grid(row=5,column=0,columnspan=2)

            self.IC_carry_popup.protocol("WM_DELETE_WINDOW", 
                                         partial(self.on_IC_carry_popup_close, 
                                                 continue_=False))
            self.IC_carry_popup.grab_set()
            self.IC_carry_popup_isopen = True
            
        else:
            print("Error #510: Opened more than one IC carryover popup at a time")
        return

    def on_IC_carry_popup_close(self, continue_=False):
        try:
            if continue_:
                plot_ID = self.active_analysisplot_ID.get()
                active_plot = self.analysis_plots[plot_ID]
                active_sets = active_plot.datagroup.datasets
                datasets = [self.carry_IC_listbox.get(i) for i in self.carry_IC_listbox.curselection()]
                if not datasets: 
                    return
                
                include_flags = {}
                for layer_name in self.module.layers:
                    include_flags[layer_name] = {}
                    for iflag in self.carryover_include_flags[layer_name]:
                        include_flags[layer_name][iflag] = self.carryover_include_flags[layer_name][iflag].get()
                    
                status_msg = ["Files generated:"]
                for key in datasets:
                    new_filename = tk.filedialog.asksaveasfilename(initialdir = self.default_dirs["Initial"], 
                                                                   title="Save IC text file for {}".format(key), 
                                                                   filetypes=[("Text files","*.txt")])
                    if not new_filename: 
                        continue

                    if new_filename.endswith(".txt"): 
                        new_filename = new_filename[:-4]
                    
                    param_dict_copy = dict(active_sets[key].params_dict)

                    grid_x = active_sets[key].grid_x
                    
                    filename = active_sets[key].filename
                    sim_data = {}
                    for layer_name, layer in self.module.layers.items():
                        sim_data[layer_name] = {}
                        for var in layer.s_outputs:
                            path_name = os.path.join(self.default_dirs["Data"], 
                                                        self.module.system_ID,
                                                        filename,
                                                        "{}-{}.h5".format(filename, var))
                            floor_tstep = int(active_plot.time / active_sets[key].dt)
                            interpolated_step = u_read(path_name, t0=floor_tstep, t1=floor_tstep+2)
                            
                            if active_plot.time == active_sets[key].total_time:
                                pass
                            else:
                                slope = (interpolated_step[1] - interpolated_step[0]) / (active_sets[key].dt)
                                interpolated_step = interpolated_step[0] + slope * (active_plot.time - floor_tstep * active_sets[key].dt)
                            
                            sim_data[layer_name][var] = interpolated_step

                    self.module.get_IC_carry(sim_data, param_dict_copy, 
                                               include_flags, grid_x)
                    
                    with open(new_filename + ".txt", "w+") as ofstream:
                        ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " 
                                       + str(datetime.datetime.now().date()) 
                                       + " AT " + str(datetime.datetime.now().time()) 
                                       + "\n")
                        ofstream.write("System_class: {}\n".format(self.module.system_ID))
                        ofstream.write("f$ System Flags:\n")
                        for flag in self.module.flags_dict:
                            ofstream.write("{}: {}\n".format(flag, active_sets[key].flags[flag]))
                        for layer_name, layer_params in param_dict_copy.items():
                            ofstream.write("L$: {}\n".format(layer_name))
                            ofstream.write("p$ Space Grid:\n")
                            ofstream.write("Total_length: {}\n".format(layer_params["Total_length"]))
                            ofstream.write("Node_width: {}\n".format(layer_params["Node_width"]))
                            ofstream.write("p$ System Parameters:\n")
                            
                            for param in layer.params:
                                if not (param == "Total_length" or param == "Node_width"):
                                    param_values = layer_params[param]
                                    param_values *= self.module.layers[layer_name].convert_out[param]
                                    if isinstance(param_values, np.ndarray):
                                        # Write the array in a more convenient format
                                        ofstream.write("{}: {:.8e}".format(param, param_values[0]))
                                        for value in param_values[1:]:
                                            ofstream.write("\t{:.8e}".format(value))
                                            
                                        ofstream.write('\n')
                                    else:
                                        # The param value is just a single constant
                                        ofstream.write("{}: {}\n".format(param, param_values))

                    status_msg.append("{}-->{}".format(filename, new_filename))
                    
                # If NO new files saved
                if len(status_msg) == 1: 
                    status_msg.append("(none)")
                self.do_confirmation_popup("\n".join(status_msg), hide_cancel=True)
                self.root.wait_window(self.confirmation_popup)

            self.IC_carry_popup.destroy()

            self.IC_carry_popup_isopen = False

        except OSError:
            self.write(self.analysis_status, "Error: failed to regenerate IC file")
            
        except Exception:
            print("Error #511: Failed to close IC carry popup.")

        return

    def do_timeseries_popup(self, ts_ID, td_gridt, td):
        ts_popup = tk.Toplevel(self.root)
        
        # Determine a nonconflicting identifier by counting how many timeseries popups are open
        tspopup_ID = 0
        while tspopup_ID in self.active_timeseries:
            tspopup_ID += 1
        ts_popup.title("Timeseries({})".format(tspopup_ID))
        td_fig = Figure(figsize=(6,4))
        td_subplot = td_fig.add_subplot(111)
        
        td_canvas = tkagg.FigureCanvasTkAgg(td_fig, master=ts_popup)
        td_plotwidget = td_canvas.get_tk_widget()
        td_plotwidget.grid(row=0,column=0, columnspan=2)
        
        
        name = td[next(iter(td))][ts_ID][0]
        where_layer = self.module.find_layer(name)
        scale_f = self.module.layers[where_layer].convert_out[name]
        td_subplot.set_yscale(autoscale(val_array=td[next(iter(td))][ts_ID][1]))
        td_subplot.set_ylabel(name + " " + self.module.layers[where_layer].outputs[name].units)
        td_subplot.set_xlabel("Time " + self.module.time_unit)
        td_subplot.set_title("Timeseries({})".format(tspopup_ID))
        
        assert tspopup_ID not in self.active_timeseries, "Error: a timeseries was overwritten"
        self.active_timeseries[tspopup_ID] = []
        for tag in td:
            print(list(td[tag][ts_ID][1] * scale_f))
            td_subplot.plot(td_gridt[tag], td[tag][ts_ID][1] * scale_f, label=tag.strip('_'))
            self.active_timeseries[tspopup_ID].append((tag, td_gridt[tag], td[tag][ts_ID][1] * scale_f))
    
        td_subplot.legend().set_draggable(True)
        td_fig.tight_layout()
        td_fig.canvas.draw()
        
        tframe = tk.Frame(master=ts_popup)
        tframe.grid(row=1,column=0,columnspan=2)
        tkagg.NavigationToolbar2Tk(td_canvas, tframe).grid(row=0,column=0,columnspan=2)
        
        tk.ttk.Button(tframe, text="Export All", command=partial(self.export_timeseries, tspopup_ID, tail=False)).grid(row=2,column=0, padx=(10,10))
        tk.ttk.Button(tframe, text="Export Tail", command=partial(self.export_timeseries, tspopup_ID, tail=True)).grid(row=2,column=1, padx=(10,10))
        ts_popup.protocol("WM_DELETE_WINDOW", partial(self.on_timeseries_popup_close, ts_popup,
                                                 tspopup_ID))
        
        return
    
    def on_timeseries_popup_close(self, ts_popup, tspopup_ID):
        try:
            del self.active_timeseries[tspopup_ID]
        except Exception as e:
            print("Failed to clear time series")
            print(e)
        ts_popup.destroy()
        return
    
    ## Plotter for simulation tab    
    def update_sim_plots(self, index, failed_vars, do_clear_plots=True):
        """ Plot snapshots of simulated data on simulate tab at regular time intervals. """
        
        for layer_name, layer in self.module.layers.items():
            convert_out = layer.convert_out
            for variable, output_obj in layer.s_outputs.items():
                
                plot = self.sim_subplots[layer_name][variable]
                
                if do_clear_plots: 
                    plot.cla()
    
                    ymin = np.amin(self.sim_data[variable]) * output_obj.yfactors[0]
                    ymax = np.amax(self.sim_data[variable]) * output_obj.yfactors[1]
                    plot.set_ylim(ymin * convert_out[variable], 
                                  ymax * convert_out[variable])
    
                plot.set_yscale(output_obj.yscale)
                
                grid_x = layer.grid_x_nodes if not output_obj.is_edge else layer.grid_x_edges
                if not failed_vars[variable]:
                    plot.plot(grid_x, self.sim_data[variable] * convert_out[variable])
    
                plot.set_xlabel("x {}".format(layer.length_unit))
                plot.set_ylabel("{} {}".format(variable, output_obj.units))
    
                plot.set_title("Time: {} ns".format(self.simtime * index / self.n))
            
        self.sim_fig.tight_layout()
        self.sim_fig.canvas.draw()
        return
    
    ## Func for overview analyze tab
    def fetch_metadata(self, data_filename):
        """ Read and store parameters from a metadata.txt file """
        path = os.path.join(self.default_dirs["Data"], self.module.system_ID, data_filename, "metadata.txt")
        assert os.path.exists(path), "Error: Missing metadata for {}".format(self.module.system_ID)
        
        with open(path, "r") as ifstream:
            param_values_dict = {}
            flag_values_dict = {}

            read_flag = 0
        
            if not ("$$ METADATA") in next(ifstream):
                raise OSError("Error: metadata is not a valid TEDs file")
        
            system_class = next(ifstream).strip('\n')
            system_class = system_class[system_class.find(' ') + 1:]
            if not system_class == self.module.system_ID:
                raise ValueError("Error: selected file is not a {}".format(self.module.system_ID))
                        
            # Extract parameters
            for line in ifstream:
            
                if ("#" in line) or not line.strip('\n'):
                    continue

                elif "L$" in line:
                    layer_name = line[line.find(":")+1:].strip(' \n')
                    param_values_dict[layer_name] = {}
                    continue
                
                elif "p$ Space Grid" in line:
                    read_flag = 'g'
                
                elif "p$ System Parameters" in line:
                    read_flag = 'p'
                
                elif "f$" in line:
                    read_flag = 'f'
                    
                elif "t$" in line:
                    read_flag = 't'
                                
                elif (read_flag == 'g'):
                    line = line.strip('\n')
                    # Unlike an MSF here the space grid is static - 
                    # and comparable to all other parameters which are static here too
                    if line.startswith("Total_length"):
                        param_values_dict[layer_name]["Total_length"] = float(line[line.rfind(' ') + 1:])
                        
                    elif line.startswith("Node_width"):
                        param_values_dict[layer_name]["Node_width"] = float(line[line.rfind(' ') + 1:])
                    
                elif (read_flag == 'p'):
                    param = line[0:line.find(':')]
                    new_value = line[line.find(' ') + 1:].strip('\n')
                    if '\t' in new_value:
                        param_values_dict[layer_name][param] = np.array(extract_values(new_value, '\t'))
                    else: 
                        param_values_dict[layer_name][param] = float(new_value)
    
                elif (read_flag == 'f'):
                    line = line.strip('\n')
                    flag_values_dict[line[0:line.find(':')]] = int(line[line.rfind(' ') + 1:])

                elif (read_flag == 't'):
                    line = line.strip('\n')
                    if line.startswith("Total-Time"):
                        total_time = float(line[line.rfind(' ') + 1:])
                    elif line.startswith("dt"):
                        dt = float(line[line.rfind(' ') + 1:])

        for layer_name, layer in param_values_dict.items():
            assert set(self.module.layers[layer_name].params.keys()).issubset(set(param_values_dict[layer_name].keys())), "Error: metadata is missing params"
        return param_values_dict, flag_values_dict, total_time, dt
    
    def plot_overview_analysis(self):
        """ Plot dataset and calculations from OneD_Model.get_overview_analysis() 
            on Overview tab. 
        """
        data_dirname = tk.filedialog.askdirectory(title="Select a dataset", 
                                                  initialdir=self.default_dirs["Data"])
        if not data_dirname:
            print("No data set selected :(")
            return

        data_filename = data_dirname[data_dirname.rfind('/')+1:]
        
        
        try:
            param_values_dict, flag_values_dict, total_time, dt = self.fetch_metadata(data_filename)
                 
            data_n = int(0.5 + total_time / dt)
            data_m = {}
            data_edge_x = {}
            data_node_x = {}
            for layer_name in param_values_dict:
                total_length = param_values_dict[layer_name]["Total_length"]
                dx = param_values_dict[layer_name]["Node_width"]
                data_m[layer_name] = int(0.5 + total_length / dx)
                data_edge_x[layer_name] = np.linspace(0,  total_length,
                                                      data_m[layer_name]+1)
                data_node_x[layer_name] = np.linspace(dx / 2, total_length - dx / 2, 
                                                      data_m[layer_name])
            data_node_t = np.linspace(0, total_time, data_n + 1)
            
            if self.overview_sample_mode.get() == 'Log':
                try:
                    sample_ct = int(self.overview_samplect_entry.get())
                    assert sample_ct > 0
                except Exception:
                    sample_ct = 5
                    
                tstep_list = np.append([0], np.geomspace(1, data_n, num=sample_ct-1, dtype=int))
            elif self.overview_sample_mode.get() == 'Linear':
                try:
                    sample_ct = int(self.overview_samplect_entry.get())
                    assert sample_ct > 0
                except Exception:
                    sample_ct = 5
                    
                tstep_list = np.linspace(0, data_n, num=sample_ct)
                
            else:
                sample_ct = self.overview_samplect_entry.get()
                tstep_list = np.array(np.array(extract_values(sample_ct, ' ')) / dt, dtype=int)
                
        except AssertionError as oops:
            self.do_confirmation_popup(str(oops), hide_cancel=True)
            self.root.wait_window(self.confirmation_popup)
            return
        
        except ValueError:
            self.do_confirmation_popup("Error: {} is missing or has unusual metadata.txt".format(data_filename), hide_cancel=True)
            self.root.wait_window(self.confirmation_popup)
            return
        
        for layer_name, layer in self.overview_subplots.items():
            for subplot in layer:
                plot_obj = layer[subplot]
                output_info_obj = self.module.layers[layer_name].outputs[subplot]
                plot_obj.cla()
                plot_obj.set_yscale(output_info_obj.yscale)
                plot_obj.set_xlabel(output_info_obj.xlabel)
                if output_info_obj.xvar == "time":
                    plot_obj.set_title("{} {}".format(output_info_obj.display_name, output_info_obj.integrated_units))
                else:
                    plot_obj.set_title("{} {}".format(output_info_obj.display_name, output_info_obj.units))
            
            
        self.overview_values = self.module.get_overview_analysis(param_values_dict, flag_values_dict,
                                                      total_time, dt,
                                                      tstep_list, data_dirname, 
                                                      data_filename)
        
        warning_msg = ["Error: the following occured while generating the overview"]
        for layer_name, layer in self.module.layers.items():
            for output_name, output_info in layer.outputs.items():
                try:
                    values = self.overview_values[layer_name][output_name]
                    if not isinstance(values, np.ndarray): 
                        raise KeyError
                except KeyError:
                    warning_msg.append("Warning: {}'s get_overview_analysis() did not return data for {}\n".format(self.module.system_ID, output_name))
                    continue
                except Exception:
                    warning_msg.append("Error: could not calculate {}".format(output_name))
                    continue
    
                if output_info.xvar == "time":
                    grid_x = data_node_t
                elif output_info.xvar == "position":
                    grid_x = data_node_x[layer_name] if not output_info.is_edge else data_edge_x[layer_name]
                else:
                    warning_msg.append("Warning: invalid xvar {} in system class definition for output {}\n".format(output_info.xvar, output_name))
                    continue
                
                if values.ndim == 2: # time/space variant outputs
                    for i in range(len(values)):
                        self.overview_subplots[layer_name][output_name].plot(grid_x, values[i], 
                                                                             label="{:.3f} ns".format(tstep_list[i] * dt))
                else: # Time variant only
                    self.overview_subplots[layer_name][output_name].plot(grid_x, values)

        for layer_name, layer in self.module.layers.items():
            for output_name in layer.s_outputs:
                self.overview_subplots[layer_name][output_name].legend().set_draggable(True)
                break
        if len(warning_msg) > 1:
            self.do_confirmation_popup("\n".join(warning_msg), hide_cancel=True)
            self.root.wait_window(self.confirmation_popup)
            
        self.analyze_overview_fig.tight_layout()
        self.analyze_overview_fig.canvas.draw()
        return

    ## Funcs for detailed analyze tab

    def plot_analyze(self, plot_ID, force_axis_update=False):
        """ Plot dataset object from make_rawdataset() on analysis tab. """
        try:
            active_plot_data = self.analysis_plots[plot_ID]
            active_datagroup = active_plot_data.datagroup
            subplot = active_plot_data.plot_obj
            
            subplot.cla()
            
            if not active_datagroup.size(): 
                self.analyze_fig.tight_layout()
                self.analyze_fig.canvas.draw()
                return

            where_layer = self.module.find_layer(active_datagroup.type)
            is_edge = self.module.layers[where_layer].outputs[active_datagroup.type].is_edge
            convert_out = self.module.layers[where_layer].convert_out
            
            if force_axis_update or not active_plot_data.do_freeze_axes:
                active_plot_data.xlim = (0, active_plot_data.datagroup.get_max_x(is_edge))
                active_plot_data.xaxis_type = 'linear'
                max_val = active_datagroup.get_maxval() * convert_out[active_datagroup.type]
                min_val = active_datagroup.get_minval() * convert_out[active_datagroup.type]
                active_plot_data.ylim = (min_val * 0.9, max_val * 1.1)
                active_plot_data.yaxis_type = autoscale(min_val=min_val, max_val=max_val)

            
            subplot.set_yscale(active_plot_data.yaxis_type)
            subplot.set_xscale(active_plot_data.xaxis_type)

            subplot.set_ylim(*active_plot_data.ylim)
            subplot.set_xlim(*active_plot_data.xlim)

            # This data is in TEDs units since we just used it in a calculation - convert back to common units first
            for dataset in active_datagroup.datasets.values():
                if active_plot_data.time <= dataset.total_time:
                    label = dataset.tag(for_matplotlib=True) + "*" if dataset.flags["symmetric_system"] else dataset.tag(for_matplotlib=True)
                    subplot.plot(dataset.grid_x, dataset.data * convert_out[active_datagroup.type], label=label)
                else:
                    print("Warning: time out of range for dataset {}".format(dataset.tag()))
                    
            subplot.set_xlabel("x {}".format(self.module.layers[where_layer].length_unit))
            subplot.set_ylabel("{} {}".format(active_datagroup.type, self.module.layers[where_layer].outputs[active_datagroup.type].units))
            if active_plot_data.display_legend:
                subplot.legend().set_draggable(True)
            subplot.set_title("Time: {} / {} ns".format(active_plot_data.time, active_datagroup.get_maxtime()))
            self.analyze_fig.tight_layout()
            self.analyze_fig.canvas.draw()
            
            active_plot_data.ylim = subplot.get_ylim()
            active_plot_data.xlim = subplot.get_xlim()

        except Exception:
            self.write(self.analysis_status, "Error #106: Plot failed")
            return

        return

    def make_rawdataset(self, data_filename, plot_ID, datatype):
        """Create a dataset object and prepare to plot on analysis tab."""
        # Select data type of incoming dataset from existing datasets
        active_plot = self.analysis_plots[plot_ID]

        try:
            param_values_dict, flag_values_dict, total_time, dt = self.fetch_metadata(data_filename)
            
            data_m = {}
            data_edge_x = {}
            data_node_x = {}
            for layer_name in param_values_dict:
                total_length = param_values_dict[layer_name]["Total_length"]
                dx = param_values_dict[layer_name]["Node_width"]
                data_m[layer_name] = int(0.5 + total_length / dx)
                data_edge_x[layer_name] = np.linspace(0,  total_length,
                                                      data_m[layer_name]+1)
                data_node_x[layer_name] = np.linspace(dx / 2, total_length - dx / 2, 
                                                      data_m[layer_name])
        except AssertionError as oops:
            return str(oops)
        except Exception:
            return "Error: missing or has unusual metadata.txt"

		# Now that we have the parameters from metadata, fetch the data itself
        sim_data = {}
        for layer_name, layer in self.module.layers.items():
            sim_data[layer_name] = {}
            for sim_datatype in layer.s_outputs:
                path_name = os.path.join(self.default_dirs["Data"], 
                                            self.module.system_ID,
                                            data_filename,
                                            "{}-{}.h5".format(data_filename, sim_datatype))
                sim_data[layer_name][sim_datatype] = u_read(path_name, t0=0, 
                                                            single_tstep=True)
        where_layer = self.module.find_layer(datatype)
        try:
            values = self.module.prep_dataset(datatype, sim_data, param_values_dict, flag_values_dict)
        except Exception as e:
            return "Error: Unable to calculate {} using prep_dataset\n{}".format(datatype, e)
        
        try:
            assert isinstance(values, np.ndarray)
            assert values.ndim == 1
            
        except Exception:
            return ("Error: Unable to calculate {} using prep_dataset\n"
                    "prep_dataset did not return a 1D array".format(datatype))

        if self.module.layers[where_layer].outputs[datatype].is_edge: 
            return Raw_Data_Set(values, data_edge_x[where_layer], data_node_x[where_layer], 
                                total_time, dt, 
                                param_values_dict, flag_values_dict, datatype, data_filename, 
                                active_plot.time)
        else:
            return Raw_Data_Set(values, data_node_x[where_layer], data_node_x[where_layer], 
                                total_time, dt,
                                param_values_dict, flag_values_dict, datatype, data_filename, 
                                active_plot.time)

    def load_datasets(self):
        """ Interpret selection from do_plotter_popup()."""
        
        plot_ID = self.active_analysisplot_ID.get()
        self.do_plotter_popup(plot_ID)
        self.root.wait_window(self.plotter_popup)
        if not self.confirmed: 
            return
        active_plot = self.analysis_plots[plot_ID]
        datatype = self.data_var.get()

        active_plot.time = 0
        active_plot.datagroup.clear()
        err_msg = ["Error: the following data could not be plotted"]
        for i in range(0, len(active_plot.data_filenames)):
            data_filename = active_plot.data_filenames[i]
            short_filename = data_filename[data_filename.rfind('/') + 1:]
            new_data = self.make_rawdataset(short_filename, plot_ID, datatype)

            if isinstance(new_data, str):
                err_msg.append("{}: {}".format(short_filename, new_data))
            else:
                active_plot.datagroup.add(new_data, new_data.tag())
    
        if len(err_msg) > 1:
            self.do_confirmation_popup("\n".join(err_msg), 
                                       hide_cancel=True)
            self.root.wait_window(self.confirmation_popup)
        
        self.plot_analyze(plot_ID, force_axis_update=True)
        
        if self.check_autointegrate.get():
            self.write(self.analysis_status, "Data read finished; integrating...")
            self.do_Integrate(bypass_inputs=True)
            
        else:
            self.write(self.analysis_status, "Data read finished")
        
        return

    def plot_tstep(self):
        """ Step already plotted data forward (or backward) in time"""
        plot_ID = self.active_analysisplot_ID.get()
        active_plot = self.analysis_plots[plot_ID]
        try:
            active_plot.add_time(float(self.analyze_tstep_entry.get()))
        except ValueError:
            self.write(self.analysis_status, "Invalid number of time steps")
            return

        active_datagroup = active_plot.datagroup
        # Search data files for data at new time
        # Interpolate if necessary
        for tag, dataset in active_datagroup.datasets.items():
            sim_data = {}
            for layer_name, layer in self.module.layers.items():
                sim_data[layer_name] = {}
                for sim_datatype in layer.s_outputs:
                    path_name = os.path.join(self.default_dirs["Data"], 
                                                self.module.system_ID,
                                                dataset.filename,
                                                "{}-{}.h5".format(dataset.filename, sim_datatype))
                    
                    floor_tstep = int(active_plot.time / dataset.dt)
                    interpolated_step = u_read(path_name, t0=floor_tstep, t1=floor_tstep+2)
                    over_time = False
                    
                    if active_plot.time > dataset.total_time:
                        interpolated_step = np.zeros_like(dataset.node_x)
                        over_time = True
                        
                    elif active_plot.time == dataset.total_time:
                        pass
                    else:
                        slope = (interpolated_step[1] - interpolated_step[0]) / (dataset.dt)
                        interpolated_step = interpolated_step[0] + slope * (active_plot.time - floor_tstep * dataset.dt)
                    
                    sim_data[layer_name][sim_datatype] = interpolated_step
        
            if over_time:
                dataset.data = interpolated_step
            else:
                dataset.data = self.module.prep_dataset(active_datagroup.type, sim_data, 
                                                          dataset.params_dict, dataset.flags)
            dataset.current_time = active_plot.time
            
            
        self.plot_analyze(plot_ID, force_axis_update=False)
        self.write(self.analysis_status, "")
        return

    ## Status box update helpers
    def write(self, textBox, text):
        """ Edit the text in status display boxes."""
        textBox.configure(state='normal')
        textBox.delete(1.0, tk.END)
        textBox.insert(tk.END, text)
        textBox.configure(state='disabled') # Prevents user from altering the status box
        return

    def enter(self, entryBox, text):
        """ Fill user entry boxes with text. """
        entryBox.delete(0,tk.END)
        entryBox.insert(0,text)
        return

    ## Tab change event handlers
	# This doesn't do anything meaningful at the moment but could be of use for things that need to be updated every time the user goes to a new tab
    def on_tab_selected(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")

        if (tab_text == "Inputs"):
            print("Inputs tab selected")
            #self.update_IC_filebox()

        elif (tab_text == "Simulate"):
            
            self.do_Batch()
            print("Simulate tab selected")
            

        elif (tab_text == "Analyze"):
            print("Analyze tab selected")

        return


    def do_Batch(self):
        """ Interpret values entered into simulate tab and prepare simulations 
            for each selected IC file
        """
        # Test for valid entry values
        try:
            # self.simtime = float(self.simtime_entry.get())      # [ns]
            # self.dt = float(self.dt_entry.get())           # [ns]
            # self.hmax = self.hmax_entry.get()
            self.simtime = float(10)      # [ns]
            self.dt = float(0.5)           # [ns]
            self.hmax = float(0.1)
            
            self.n = int(0.5 + self.simtime / self.dt)           # Number of time steps

            # Upper limit on number of time steps
            
            assert (self.simtime > 0),"Error: Invalid simulation time"
            assert (self.dt > 0 and self.dt <= self.simtime),"Error: Invalid dt"
            
            if not self.hmax: # Default hmax
                self.hmax = np.inf
                
            self.hmax = float(self.hmax)
            
            assert (self.hmax >= 0),"Error: Invalid solver stepsize"
            
            if self.dt > self.simtime / 10:
                self.do_confirmation_popup("Warning: a very large time stepsize was entered. "
                                           "Results may be less accurate with large stepsizes. "
                                           "Are you sure you want to continue?")
                self.root.wait_window(self.confirmation_popup)
                if not self.confirmed: 
                    return
                
            if self.hmax and self.hmax < 1e-3:
                self.do_confirmation_popup("Warning: a very small solver stepsize was entered. "
                                           "Results may be slow with small solver stepsizes. "
                                           "Are you sure you want to continue?")
                self.root.wait_window(self.confirmation_popup)
                if not self.confirmed: 
                    return
                
            if (self.n > 1e5):
                self.do_confirmation_popup("Warning: a very small time stepsize was entered. "
                                           "Results may be slow with small time stepsizes. "
                                           "Are you sure you want to continue?")
                self.root.wait_window(self.confirmation_popup)
                if not self.confirmed: 
                    return
            
        except ValueError:
            self.write(self.status, "Error: Invalid parameters")
            return

        except (AssertionError, Exception) as oops:
            self.write(self.status, oops)
            return

        IC_files = tk.filedialog.askopenfilenames(initialdir=self.default_dirs["Initial"], 
                                                  title="Select IC text file", 
                                                  filetypes=[("Text files","*.txt")])
        if not IC_files: 
            return

        batch_num = 0
        self.sim_warning_msg = ["The following occured while simulating:"]
        for IC in IC_files:
            batch_num += 1
            # self.IC_file_name = IC
            self.IC_file_name = "/Users/frederik.eistrup/personal/upconversion/OneLayer/Initial/fredstry/fredstry.txt"          
            self.load_ICfile()
            self.write(self.status, 
                       "Now calculating {} : ({} of {})".format(self.IC_file_name[self.IC_file_name.rfind("/") + 1:self.IC_file_name.rfind(".txt")], 
                                                                str(batch_num), str(len(IC_files))))
            self.do_Calculate()
            
        self.write(self.status, "Simulations complete")
        self.load_ICfile()

        if len(self.sim_warning_msg) > 1:
            self.do_confirmation_popup("\n".join(self.sim_warning_msg), hide_cancel=True)
        return

    def do_Calculate(self):
        """ Setup initial condition using IC file and OneD_Model.calc_inits(), 
            simulate using OneD_Model.simulate(),
            and prepare output directory for results.
        """
        ## Setup parameters
        try:
            # Construct the data folder's name from the corresponding IC file's name
            shortened_IC_name = self.IC_file_name[self.IC_file_name.rfind("/") + 1:self.IC_file_name.rfind(".txt")]
            data_file_name = shortened_IC_name
            temp_params = {}
            num_nodes = {}
            for layer_name, layer in self.module.layers.items():
                num_nodes[layer_name] = int(0.5 + layer.total_length / layer.dx)
                temp_params[layer_name] = dict(layer.params)

            init_conditions = self.module.calc_inits()
            assert isinstance(init_conditions, dict), "Error: module calc_inits() did not return a dict of initial conditions\n"
            
            # Can't verify this ahead of time unfortunately so verify on demand
            for layer_name, layer in self.module.layers.items():
                for variable in layer.s_outputs:
                    assert variable in init_conditions, "Error: Module calc_inits() did not return value for simulation output variable {}\n".format(variable)
                    assert isinstance(init_conditions[variable], np.ndarray), "Error: module calc_inits() returned an invalid value (values must be numpy arrays) for output {}\n".format(variable)
                    assert init_conditions[variable].ndim == 1, "Error: module calc_inits() did not return a 1D numpy array for output {}\n".format(variable)
        
        except ValueError:
            self.sim_warning_msg.append("Error: Invalid parameters for {}".format(data_file_name))
            return
        
        except AssertionError as oops:
            self.sim_warning_msg.append(str(oops))
            return

        except Exception as oops:
            self.sim_warning_msg.append("Error: \"{}\" reported while setting up {}".format(oops, data_file_name))
            return
    
        try:
            print("Attempting to create {} data folder".format(data_file_name))
            dirname = os.path.join(self.default_dirs["Data"], 
                                    self.module.system_ID,
                                    data_file_name)
            # Append a number to the end of the new directory's name if an overwrite would occur
            # This is what happens if you download my_file.txt twice and the second copy is saved as my_file(1).txt, for example
            assert "Data" in dirname
            assert self.module.system_ID in dirname
            if os.path.isdir(dirname):
                print("{} folder already exists; trying alternate name".format(data_file_name))
                append = 1
                while (os.path.isdir("{}({})".format(dirname, append))):
                    append += 1

                dirname = "{}({})".format(dirname, append)
                
                
                self.sim_warning_msg.append("Overwrite warning - {} already exists "
                                            "in Data directory\nSaving as {} instead".format(data_file_name, dirname))
                
                data_file_name = "{}({})".format(data_file_name, append)
                
            os.mkdir("{}".format(dirname))

        except Exception:
            self.sim_warning_msg.append("Error: unable to create directory for results "
                                        "of simulation {}\n".format(shortened_IC_name))
            return


        ## Calculate!
        #atom = tables.Float64Atom()

        ## Create data files
        # for layer_name, layer in self.module.layers.items():
        #     for variable in layer.s_outputs:
        #         path = os.path.join(dirname, "{}-{}.h5".format(data_file_name, variable))
        #         with tables.open_file(path, mode='w') as ofstream:
        #             length = num_nodes[layer_name] 
        #             if layer.s_outputs[variable].is_edge:
        #                 length += 1
    
        #             # Important - "data" must be used as the array name here, as pytables will use the string "data" 
        #             # to name the attribute earray.data, which is then used to access the array
        #             earray = ofstream.create_earray(ofstream.root, "data", atom, (0, length))
        #             earray.append(np.reshape(init_conditions[variable], (1, length)))
        
        ## Setup simulation plots and plot initial
        
        self.sim_data = dict(init_conditions)
        #self.update_sim_plots(0)
        flag_values = {f:flag.value() for f, flag in self.sys_flag_dict.items()}

        try:
            self.module.simulate(os.path.join(dirname, data_file_name), 
                                   num_nodes, self.n, self.dt,
                                   flag_values, self.hmax, init_conditions)
            
        except FloatingPointError as e:
            print(e)
            self.sim_warning_msg.append("Error: an unusual value occurred while simulating {}. "
                                        "This file may have invalid parameters.".format(data_file_name))
            for file in os.listdir(dirname):
                tpath = os.path.join(dirname, file)
                os.remove(tpath)
                
            os.rmdir(dirname)
            return
        
        except KeyboardInterrupt:
            print("### Aborting {} ###".format(data_file_name))
            self.sim_warning_msg.append("Abort signal received while simulating {}\n".format(data_file_name))
            for file in os.listdir(dirname):
                tpath = os.path.join(dirname, file)
                os.remove(tpath)
                
            os.rmdir(dirname)
            return
        
        except Exception as oops:
            self.sim_warning_msg.append("Error \"{}\" occurred while simulating {}\n".format(oops, data_file_name))
            for file in os.listdir(dirname):
                tpath = os.path.join(dirname, file)
                os.remove(tpath)
                
            os.rmdir(dirname)
            
            return

        self.write(self.status, "Finalizing...")

        failed_vars = {}
        for i in range(1,6):
            for var in self.sim_data:
                path_name = os.path.join(dirname, "{}-{}.h5".format(data_file_name, var))
                failed_vars[var] = 0
                try:
                    self.sim_data[var] = u_read(path_name, t0=int(self.n * i / 5), 
                                                single_tstep=True)
                    
                except Exception:
                    self.sim_data[var] = 0
                    failed_vars[var] = 1
            is_first = (i == 1)
            self.update_sim_plots(self.n, failed_vars, do_clear_plots=is_first)
            
        failed_vars = [var for var, fail_state in failed_vars.items() if fail_state]
        if failed_vars:
            self.sim_warning_msg.append("Warning: unable to plot {} for {}. Output data "
                                        "may not have been saved correctly.\n".format(failed_vars, data_file_name))
        
        # Save metadata: list of param values used for the simulation
        # Inverting the unit conversion between the inputted params and the calculation engine is also necessary to regain the originally inputted param values
        with open(os.path.join(dirname, "metadata.txt"), "w+") as ofstream:
            ofstream.write("$$ METADATA FOR CALCULATIONS PERFORMED ON {} AT {}\n".format(datetime.datetime.now().date(),datetime.datetime.now().time()))
            ofstream.write("System_class: {}\n".format(self.module.system_ID))
            
            ofstream.write("f$ System Flags:\n")
            for flag in self.sys_flag_dict:
                ofstream.write("{}: {}\n".format(flag, self.sys_flag_dict[flag].value()))
            
            for layer_name, layer in self.module.layers.items():
                ofstream.write("L$: {}\n".format(layer_name))
                ofstream.write("p$ Space Grid:\n")
                ofstream.write("Total_length: {}\n".format(layer.total_length))
                ofstream.write("Node_width: {}\n".format(layer.dx))
            
                ofstream.write("p$ System Parameters:\n")
            
                for param in layer.params:
                    param_values = layer.params[param].value
                    if isinstance(param_values, np.ndarray):
                        ofstream.write("{}: {:.8e}".format(param, param_values[0]))
                        for value in param_values[1:]:
                            ofstream.write("\t{:.8e}".format(value))
                            
                        ofstream.write('\n')
                    else:
                        ofstream.write("{}: {}\n".format(param, param_values))
            # The following params are exclusive to metadata files
            ofstream.write("t$ Time grid:\n")
            ofstream.write("Total-Time: {}\n".format(self.simtime))
            ofstream.write("dt: {}\n".format(self.dt))
        return

    def do_Integrate(self, bypass_inputs=False):
        """ Interpret values from series of integration popups to integrate datasets. """
        plot_ID = self.active_analysisplot_ID.get()
        
        # Replace this with an appropriate getter function if more integration 
        # plots are added
        ip_ID = 0
        
        self.write(self.analysis_status, "")

        active_plot = self.analysis_plots[plot_ID]
        active_datagroup = active_plot.datagroup
        if not active_datagroup.datasets: 
            return

        # Collect instructions from user using a series of popup windows
        if not bypass_inputs:
            self.do_integration_timemode_popup()
            self.root.wait_window(self.integration_popup) # Pause here until popup is closed
            if not self.PL_mode:
                self.write(self.analysis_status, "Integration cancelled")
                return
    
            self.do_integration_getbounds_popup()
            self.root.wait_window(self.integration_getbounds_popup)
            if not self.confirmed:
                return
            
            if self.PL_mode == "Current time step":
                self.do_PL_xaxis_popup()
                self.root.wait_window(self.PL_xaxis_popup)
                if not self.xaxis_param:
                    self.write(self.analysis_status, "Integration cancelled")
                    return
                print("Selected param {}".format(self.xaxis_param))
                self.integration_plots[ip_ID].x_param = self.xaxis_param
    
            else:
                self.integration_plots[ip_ID].x_param = "Time"
                
        else:
            # A "default integration behavior": integrate the present data over all time and space steps
            self.PL_mode = "All time steps"
            self.integration_plots[ip_ID].x_param = "Time"
            where_layer = self.module.find_layer(active_datagroup.type)
            is_edge = self.module.layers[where_layer].outputs[active_datagroup.type].is_edge
            self.integration_bounds = [[0,active_datagroup.get_max_x(is_edge)]]
            
        # Clean up the I_plot and prepare to integrate given selections
        # A lot of the following is a data transfer between the 
        # sending active_datagroup and the receiving I_plot
        self.integration_plots[ip_ID].datagroup.clear()
        self.integration_plots[ip_ID].mode = self.PL_mode
        self.integration_plots[ip_ID].global_gridx = None

        if self.PL_mode == "All time steps":
            td_gridt = {}
            td = {}
        
        counter = 0
        
        for tag in active_datagroup.datasets:
            data_filename = active_datagroup.datasets[tag].filename
            datatype = active_datagroup.datasets[tag].type
            where_layer = self.module.find_layer(datatype)
            print("Now integrating {}".format(data_filename))

            # Unpack needed params from the dictionaries of params
            dx = active_datagroup.datasets[tag].params_dict[where_layer]["Node_width"]
            total_length = active_datagroup.datasets[tag].params_dict[where_layer]["Total_length"]
            total_time = active_datagroup.datasets[tag].total_time
            current_time = active_datagroup.datasets[tag].current_time
            dt = active_datagroup.datasets[tag].dt
            n = active_datagroup.datasets[tag].num_tsteps
            symmetric_flag = active_datagroup.datasets[tag].flags["symmetric_system"]

            if current_time > total_time: continue

            if self.PL_mode == "Current time step":
                show_index = int(current_time / dt)
                end_index = show_index+2
            else:
                show_index = None
                end_index = None

            # Clean up any bounds that extend past the confines of the system
            # The system usually exists from x=0 to x=total_length, 
            # but can accept x=-total_length to x=total_length if symmetric

            for bounds in self.integration_bounds:
                l_bound = bounds[0]
                u_bound = bounds[1]
               
                if (u_bound > total_length):
                    u_bound = total_length
                    
                if (l_bound > total_length):
                    l_bound = total_length

                if symmetric_flag:

                    if (l_bound < -total_length):
                        l_bound = -total_length
                else:
                    if (l_bound < 0):
                        l_bound = 0

                include_negative = symmetric_flag and (l_bound < 0)

                print("Bounds after cleanup: {} to {}".format(l_bound, u_bound))

                j = to_index(u_bound, dx, total_length)
                i = to_index(abs(l_bound), dx, total_length)
                if include_negative:  
                    nen = [-l_bound > to_pos(i, dx) + dx / 2,
                                       u_bound > to_pos(j, dx) + dx / 2]
                    
                    space_bounds = [(0,i,nen[0], 0, -l_bound), (0,j,nen[1], 0, u_bound)]
                else:
                    nen = u_bound > to_pos(j, dx) + dx / 2 or l_bound == u_bound
                    space_bounds = [(i,j,nen, l_bound, u_bound)]

                do_curr_t = self.PL_mode == "Current time step"
                
                pathname = os.path.join(self.default_dirs["Data"], self.module.system_ID, data_filename, data_filename)
                
                extra_data = {}
                for c, s in enumerate(space_bounds):
                    sim_data = {}
                    for layer_name, layer in self.module.layers.items():
                        sim_data[layer_name] = {}
                        extra_data[layer_name] = {}
                        for sim_datatype in layer.s_outputs:

                            if do_curr_t:
                                interpolated_step = u_read("{}-{}.h5".format(pathname, sim_datatype), 
                                                           t0=show_index, t1=end_index, l=s[0], r=s[1]+1, 
                                                           single_tstep=False, need_extra_node=s[2], 
                                                           force_1D=False)
                                if current_time == total_time:
                                    pass
                                else:
                                    floor_tstep = int(current_time / dt)
                                    slope = (interpolated_step[1] - interpolated_step[0]) / (dt)
                                    interpolated_step = interpolated_step[0] + slope * (current_time - floor_tstep * dt)
                                
                                sim_data[layer_name][sim_datatype] = np.array(interpolated_step)
                                
                                interpolated_step = u_read("{}-{}.h5".format(pathname, sim_datatype), 
                                                                             t0=show_index, t1=end_index, single_tstep=False, force_1D=False)
                                if current_time == total_time:
                                    pass
                                else:
                                    floor_tstep = int(current_time / dt)
                                    slope = (interpolated_step[1] - interpolated_step[0]) / (dt)
                                    interpolated_step = interpolated_step[0] + slope * (current_time - floor_tstep * dt)
                                    
                                extra_data[layer_name][sim_datatype] = np.array(interpolated_step)
                            
                            else:
                                sim_data[layer_name][sim_datatype] = u_read("{}-{}.h5".format(pathname, sim_datatype), 
                                                                            t0=show_index, t1=end_index, l=s[0], r=s[1]+1, 
                                                                            single_tstep=False, need_extra_node=s[2], 
                                                                            force_1D=False) 
                                
                                if c == 0:
                                    extra_data[layer_name][sim_datatype] = u_read("{}-{}.h5".format(pathname, sim_datatype), 
                                                                                  t0=show_index, t1=end_index, single_tstep=False, force_1D=False)
            
                    data = self.module.prep_dataset(datatype, sim_data, 
                                                      active_datagroup.datasets[tag].params_dict, 
                                                      active_datagroup.datasets[tag].flags,
                                                      True, s[0], s[1], s[2], extra_data)
                    
                    if c == 0: I_data = new_integrate(data, s[3], s[4], dx, total_length, s[2])
                    else: I_data += new_integrate(data, s[3], s[4], dx, total_length, s[2])
                            
                if self.PL_mode == "Current time step":
                    # Don't forget to change out of TEDs units, or the x axis won't match the parameters the user typed in
                    grid_xaxis = float(active_datagroup.datasets[tag].params_dict[where_layer][self.xaxis_param]
                                       * self.module.layers[where_layer].convert_out[self.xaxis_param])
                    xaxis_label = self.xaxis_param + " [WIP]"

                elif self.PL_mode == "All time steps":
                    grid_xaxis = np.linspace(0, total_time, n + 1)
                    xaxis_label = "Time [ns]"

                ext_tag = data_filename + "__" + str(l_bound) + "_to_" + str(u_bound)
                self.integration_plots[ip_ID].datagroup.add(Integrated_Data_Set(I_data, grid_xaxis, total_time, dt,
                                                                                active_datagroup.datasets[tag].params_dict, 
                                                                                active_datagroup.datasets[tag].flags,
                                                                                active_datagroup.datasets[tag].type, 
                                                                                ext_tag))
            
                if self.PL_mode == "All time steps":
                    try:
                        td[ext_tag] = self.module.get_timeseries(pathname, active_datagroup.datasets[tag].type, I_data, total_time, dt,
                                                                 active_datagroup.datasets[tag].params_dict, active_datagroup.datasets[tag].flags)
                    except Exception:
                        print("Error: failed to calculate time series")
                        td[ext_tag] = None
                        
                    if td[ext_tag] is not None:
                        td_gridt[ext_tag] = np.linspace(0, total_time, n + 1)
                        
                counter += 1
                print("Integration: {} of {} complete".format(counter, active_datagroup.size() * len(self.integration_bounds)))

        subplot = self.integration_plots[ip_ID].plot_obj
        datagroup = self.integration_plots[ip_ID].datagroup
        subplot.cla()
        
        self.integration_plots[ip_ID].xaxis_type = 'linear'

        self.integration_plots[ip_ID].yaxis_type = autoscale(min_val=datagroup.get_minval(), 
                                                             max_val=datagroup.get_maxval())
        #self.integration_plots[ip_ID].ylim = max * 1e-12, max * 10
        where_layer = self.module.find_layer(datagroup.type)
        
        subplot.set_yscale(self.integration_plots[ip_ID].yaxis_type)
        #subplot.set_ylim(self.integration_plots[ip_ID].ylim)
        subplot.set_xlabel(xaxis_label)
        subplot.set_ylabel(datagroup.type +  " " + self.module.layers[where_layer].outputs[datagroup.type].integrated_units)
        subplot.set_title("Integrated {}".format(datagroup.type))

        
        for key in datagroup.datasets:
            if self.PL_mode == "Current time step":
                f = subplot.scatter
            elif self.PL_mode == "All time steps":
                f = subplot.plot
            f(datagroup.datasets[key].grid_x, 
                datagroup.datasets[key].data * 
                self.module.layers[where_layer].convert_out[datagroup.type] *
                self.module.layers[where_layer].iconvert_out[datagroup.type], 
                label=datagroup.datasets[key].tag(for_matplotlib=True))

            
        self.integration_plots[ip_ID].xlim = subplot.get_xlim()
        self.integration_plots[ip_ID].ylim = subplot.get_ylim()
                
        subplot.legend().set_draggable(True)

        self.integration_fig.tight_layout()
        self.integration_fig.canvas.draw()
        
        self.write(self.analysis_status, "Integration complete")

        if self.PL_mode == "All time steps" and len(td_gridt):
            
            num_td_per_curve = len(td[next(iter(td))])
            for i in range(num_td_per_curve):
                try:
                    self.do_timeseries_popup(i, td_gridt, td)
                except Exception:
                    continue
        return

    ## Initial Condition Managers

    def reset_IC(self, force=False):
        """ On IC tab:
            For each selected layer with a set spacegrid 
            (i.e. has values that need resetting):
            # 1. Remove all param_rules from the listbox
            # 2. Remove all param_rules stored in Module
            # 3. Remove all values stored in Module
            # + any cool visual effects
        """

        self.do_resetIC_popup()
        self.root.wait_window(self.resetIC_popup)

        if (not self.resetIC_selected_layers):
            print("No layers selected :(")
            return

        for layer_name in self.resetIC_selected_layers:
            layer = self.module.layers[layer_name]
            if layer.spacegrid_is_set:
                self.current_layer_selection.set(layer_name)
                self.change_layer()
                for param in layer.params:
                    # Step 1 and 2
                    self.paramtoolkit_currentparam = param
                
                    # These two lines changes the text displayed in the param_rule 
                    # display box's menu and is for cosmetic purposes only
                    self.update_paramrule_listbox(param)
                    self.paramtoolkit_viewer_selection.set(param)
                    
                    self.deleteall_paramrule()
                    
                    # Step 3
                    layer.params[param].value = 0
                   
                if self.using_LGC[layer_name]:
                    self.using_LGC[layer_name] = False
                    self.LGC_values[layer_name] = {}
                    self.LGC_options[layer_name] = {}
            
                self.set_thickness_and_dx_entryboxes(state='unlock')
                layer.total_length = None
                layer.dx = None
                layer.grid_x_edges = []
                layer.grid_x_nodes = []
                layer.spacegrid_is_set = False


        self.update_IC_plot(plot_ID="clearall")                             
        self.update_system_summary()                    
        self.write(self.ICtab_status, "Selected layers cleared")
        return

	## This is a patch of a consistency issue involving initial conditions - 
    ## we require different variables of a single initial condition
	## to fit to the same spatial mesh, which can get messed up if the 
    ## user changes the mesh while editing initial conditions.

    # First, implement a way to temporarily remove the user's ability 
    # to change variables associated with the spatial mesh
    def set_thickness_and_dx_entryboxes(self, state):
        """ Toggle ability to change spatial mesh while parameters are stored. """
        if state =='lock':
            self.thickness_entry.configure(state='disabled')
            self.dx_entry.configure(state='disabled')

        elif state =='unlock':
            self.thickness_entry.configure(state='normal')
            self.dx_entry.configure(state='normal')

        return

    
    def set_init_x(self):
        """Generate and lock in new spatial mesh. 
           A new mesh can only be generated when the previous mesh 
           is discarded using reset_IC().
        """
        current_layer = self.module.layers[self.current_layer_name]
        if current_layer.spacegrid_is_set:
            return

        thickness = float(self.thickness_entry.get())
        dx = float(self.dx_entry.get())

        assert (thickness > 0 and dx > 0), "Error: invalid thickness or dx"

        assert dx <= thickness / 2, "Error: space step size too large"

        # Upper limit on number of space steps
        assert (int(0.5 + thickness / dx) <= 1e6), "Error: too many space steps"

        if dx > thickness / 10:
            self.do_confirmation_popup("Warning: a very large space stepsize was entered. "
                                       "Results may be less accurate with large stepsizes. "
                                       "Are you sure you want to continue?")
            self.root.wait_window(self.confirmation_popup)
            if not self.confirmed: 
                return
            
        if abs(thickness / dx - int(thickness / dx)) > 1e-10:
            self.do_confirmation_popup("Warning: the selected thickness cannot be "
                                       "partitioned evenly into the selected stepsize. "
                                       "Integration may lose accuracy as a result. "
                                       "Are you sure you want to continue?")
            self.root.wait_window(self.confirmation_popup)
            if not self.confirmed: 
                return
            
        current_layer.total_length = thickness
        current_layer.dx = dx
        current_layer.grid_x_nodes = np.linspace(dx / 2,thickness - dx / 2, int(0.5 + thickness / dx))
        current_layer.grid_x_edges = np.linspace(0, thickness, int(0.5 + thickness / dx) + 1)
        current_layer.spacegrid_is_set = True
        self.set_thickness_and_dx_entryboxes(state='lock')
        return

    def add_LGC(self):
        """Calculate and store laser generation profile"""
        # Update current working layer
        if not self.LGC_layer.get():
            self.write(self.ICtab_status, "Select a layer to apply this excitation on")
            return
        
        if not self.current_layer_name == self.LGC_layer.get():
            self.current_layer_selection.set(self.LGC_layer.get())
            
            # User shortcut - if no layers' spacegrids set yet, let the first LGC set a spacegrid
            any_layers_set =  any([layer.spacegrid_is_set for name, layer in self.module.layers.items()])
            self.change_layer(clear=any_layers_set, update_LGC_display=False)
        current_layer = self.module.layers[self.current_layer_name]
        
        try:
            self.set_init_x()
            assert current_layer.spacegrid_is_set, "Error: could not set space grid"

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return
        
        except (AssertionError, Exception) as oops:
            self.write(self.ICtab_status, oops)
            return

        try:
            assert (not (self.LGC_optionboxes["long_expfactor"].get() == "") and self.LGC_optionboxes["power_mode"].get()), "Error: select material param and power generation options "
        except AssertionError as oops:
            self.write(self.ICtab_status, oops)
            return

        # Remove all param_rules for delta_N and delta_P, 
        # as we will be reassigning them shortly.
        self.paramtoolkit_currentparam = "delta_N"
        self.deleteall_paramrule()
        self.paramtoolkit_currentparam = "delta_P"
        self.deleteall_paramrule()

        # Establish constants; calculate alpha
        h = 6.626e-34   # [J*s]
        c = 2.997e8     # [m/s]
        hc_evnm = h * c * 6.241e18 * 1e9    # [J*m] to [eV*nm]
        hc_nm = h * c * 1e9     # [J*m] to [J*nm] 
        
        try:
            if current_layer.params['delta_N'].is_edge and current_layer.params['delta_P'].is_edge:
                grid_x = current_layer.grid_x_edges
            elif not (current_layer.params['delta_N'].is_edge or current_layer.params['delta_P'].is_edge):
                grid_x = current_layer.grid_x_nodes
            else:
                raise ValueError
        except ValueError:
            self.write(self.ICtab_status, "Node mismatch - dN is edge but dP is node or vice versa")
            return

        if (self.LGC_optionboxes["long_expfactor"].get()):
            try: 
                A0 = float(self.A0_entry.get())   # [cm^-1 eV^-1/2] or [cm^-1 eV^-2]
            except Exception:
                self.write(self.ICtab_status, "Error: missing or invalid A0")
                return

            try: 
                Eg = float(self.Eg_entry.get())   # [eV]
            except Exception:
                self.write(self.ICtab_status, "Error: missing or invalid Eg")
                return

            try: 
                wavelength = float(self.pulse_wavelength_entry.get())  # [nm]
                assert wavelength > 0
            except Exception:
                self.write(self.ICtab_status, "Error: missing or invalid pulsed laser wavelength")
                return

            if self.LGC_optionboxes["incidence"].get() == "direct":
                alpha = A0 * (hc_evnm / wavelength - Eg) ** 0.5     # [cm^-1]

            elif self.LGC_optionboxes["incidence"].get() == "indirect":
                alpha = A0 * (hc_evnm / wavelength - Eg) ** 2

            else:
                self.write(self.ICtab_status, "Select \"direct\" or \"indirect\"")
                return

        else:
            try: 
                alpha = float(self.LGC_absorption_cof_entry.get()) # [cm^-1]
                
            except Exception:
                self.write(self.ICtab_status, "Error: missing or invalid α")
                return
            

        alpha_nm = alpha * 1e-7 # [cm^-1] to [nm^-1]

        if (self.LGC_optionboxes["power_mode"].get() == "power-spot"):
            try: 
                power = float(self.power_entry.get()) * 1e-6  # [uJ/s] to [J/s]
                spotsize = float(self.spotsize_entry.get()) * ((1e7) ** 2)     # [cm^2] to [nm^2]
                assert spotsize > 0
            except Exception:
                self.write(self.ICtab_status, "Error: missing power or spot size")
                return

            
            try: 
                wavelength = float(self.pulse_wavelength_entry.get())              # [nm]
                assert wavelength > 0
            except Exception:
                self.write(self.ICtab_status, "Error: missing or invalid pulsed laser wavelength")
                return

            if (self.pulse_freq_entry.get() == "cw" or self.sys_flag_dict["check_do_ss"].value()):
                freq = 1e9 # Convert [J/s] power to [J/ns]
            else:
                try:
                    freq = float(self.pulse_freq_entry.get()) * 1e3    # [kHz] to [1/s]
                    assert freq > 0
                except Exception:
                    self.write(self.ICtab_status, "Error: missing or invalid pulse frequency")
                    return

            # Note: add_LGC() automatically converts into TEDs units. For consistency add_LGC should really deposit values in common units.
            current_layer.params["delta_N"].value = carrier_excitations.pulse_laser_power_spotsize(power, spotsize, 
                                                                                                   freq, wavelength, alpha_nm, 
                                                                                                   grid_x, hc=hc_nm)
        
        elif (self.LGC_optionboxes["power_mode"].get() == "density"):
            try: power_density = float(self.power_density_entry.get()) * 1e-6 * ((1e-7) ** 2)  # [uW / cm^2] to [J/s nm^2]
            except Exception:
                self.write(self.ICtab_status, "Error: missing power density")
                return

            try: 
                wavelength = float(self.pulse_wavelength_entry.get())              # [nm]
                assert wavelength > 0
            except Exception:
                self.write(self.ICtab_status, "Error: missing or invalid pulsed laser wavelength")
                return
            if (self.pulse_freq_entry.get() == "cw" or self.sys_flag_dict["check_do_ss"].value()):
                freq = 1e9 # Convert [J/s] power to [J/ns]
            else:
                try:
                    freq = float(self.pulse_freq_entry.get()) * 1e3    # [kHz] to [1/s]
                    assert freq > 0
                except Exception:
                    self.write(self.ICtab_status, "Error: missing or invalid pulse frequency")
                    return

            current_layer.params["delta_N"].value = carrier_excitations.pulse_laser_powerdensity(power_density, freq, 
                                                                                                 wavelength, alpha_nm, 
                                                                                                 grid_x, hc=hc_nm)
        
        elif (self.LGC_optionboxes["power_mode"].get() == "max-gen"):
            try: max_gen = float(self.max_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except Exception:
                self.write(self.ICtab_status, "Error: missing max gen")
                return

            current_layer.params["delta_N"].value = carrier_excitations.pulse_laser_maxgen(max_gen, alpha_nm, 
                                                                                           grid_x)
        
        elif (self.LGC_optionboxes["power_mode"].get() == "total-gen"):
            try: total_gen = float(self.total_gen_entry.get()) * ((1e-7) ** 3) # [cm^-3] to [nm^-3]
            except Exception:
                self.write(self.ICtab_status, "Error: missing total gen")
                return

            current_layer.params["delta_N"].value = carrier_excitations.pulse_laser_totalgen(total_gen, current_layer.total_length, 
                                                                                             alpha_nm, grid_x)
        
        else:
            self.write(self.ICtab_status, "An unexpected error occurred while calculating the power generation params")
            return
        
        if self.LGC_direction.get() == "reverse":
            current_layer.params["delta_N"].value = current_layer.params["delta_N"].value[::-1]
        
        current_layer.params["delta_N"].value *= current_layer.convert_out["delta_N"]
        ## Assuming that the initial distributions of holes and electrons are identical
        current_layer.params["delta_P"].value = np.array(current_layer.params["delta_N"].value)

        self.update_IC_plot(plot_ID="LGC")
        self.paramtoolkit_currentparam = "delta_N"
        self.update_IC_plot(plot_ID="custom")
        self.paramtoolkit_currentparam = "delta_P"
        self.update_IC_plot(plot_ID="recent")
        self.using_LGC[self.current_layer_name] = True
        
        # If LGC profile successful, record all parameters used to generate
        self.LGC_options[self.current_layer_name] = {LGC_option: self.LGC_optionboxes[LGC_option].get() \
                                                     for LGC_option in self.LGC_optionboxes}
        self.LGC_values[self.current_layer_name] = {}
        
        if (self.LGC_options[self.current_layer_name]["long_expfactor"]):
            self.LGC_values[self.current_layer_name]["A0"] = A0
            self.LGC_values[self.current_layer_name]["Eg"] = Eg
            self.LGC_values[self.current_layer_name]["Pulse_Wavelength"] = wavelength
        else:
            self.LGC_values[self.current_layer_name]["LGC_absorption_cof"] = alpha
            
        if (self.LGC_options[self.current_layer_name]["power_mode"] == "power-spot"):
            self.LGC_values[self.current_layer_name]["Power"] = power * 1e6
            self.LGC_values[self.current_layer_name]["Spotsize"] = spotsize * (1e-7) ** 2
            self.LGC_values[self.current_layer_name]["Pulse_Wavelength"] = wavelength
            if (self.pulse_freq_entry.get() == "cw" or self.sys_flag_dict["check_do_ss"].value()):
                pass
            else:
                self.LGC_values[self.current_layer_name]["Pulse_Freq"] = freq * 1e-3
                    
        elif (self.LGC_options[self.current_layer_name]["power_mode"] == "density"):
            self.LGC_values[self.current_layer_name]["Power_Density"] = power_density * 1e6 * ((1e7) ** 2)
            self.LGC_values[self.current_layer_name]["Pulse_Wavelength"] = wavelength

            if (self.pulse_freq_entry.get() == "cw" or self.sys_flag_dict["check_do_ss"].value()):
                pass
            else:
                self.LGC_values[self.current_layer_name]["Pulse_Freq"] = freq * 1e-3
                
        elif (self.LGC_options[self.current_layer_name]["power_mode"] == "max-gen"):
            self.LGC_values[self.current_layer_name]["Max_Gen"] = max_gen * ((1e7) ** 3)

        elif (self.LGC_options[self.current_layer_name]["power_mode"] == "total-gen"):
            self.LGC_values[self.current_layer_name]["Total_Gen"] = total_gen * ((1e7) ** 3)
        
        return

    ## Special functions for Parameter Toolkit:
    def add_paramrule(self):
        """ Add a new parameter rule to the selected parameter. """
        current_layer = self.module.layers[self.current_layer_name]
        try:
            self.set_init_x()
            assert current_layer.spacegrid_is_set, "Error: could not set space grid"

        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except (AssertionError, Exception) as oops:
            self.write(self.ICtab_status, oops)
            return

        try:
            new_param_name = self.init_var_selection.get()
            if "[" in new_param_name: 
                new_param_name = new_param_name[:new_param_name.find("[")]

            assert (float(self.paramrule_lbound_entry.get()) >= 0),  "Error: left bound coordinate too low"
            
            if (self.init_shape_selection.get() == "POINT"):
                assert (float(self.paramrule_lbound_entry.get()) <= current_layer.total_length), "Error: right bound coordinate too large"
                
                new_param_rule = Param_Rule(new_param_name, "POINT", 
                                            float(self.paramrule_lbound_entry.get()), 
                                            -1, 
                                            float(self.paramrule_lvalue_entry.get()), 
                                            -1)

            elif (self.init_shape_selection.get() == "FILL"):
                assert (float(self.paramrule_rbound_entry.get()) <= current_layer.total_length), "Error: right bound coordinate too large"
                assert (float(self.paramrule_lbound_entry.get()) < float(self.paramrule_rbound_entry.get())), "Error: Left bound coordinate is larger than right bound coordinate"

                new_param_rule = Param_Rule(new_param_name, "FILL", 
                                            float(self.paramrule_lbound_entry.get()), 
                                            float(self.paramrule_rbound_entry.get()), 
                                            float(self.paramrule_lvalue_entry.get()), 
                                            -1)

            elif (self.init_shape_selection.get() == "LINE"):
                assert (float(self.paramrule_rbound_entry.get()) <= current_layer.total_length), "Error: right bound coordinate too large"
                assert (float(self.paramrule_lbound_entry.get()) < float(self.paramrule_rbound_entry.get())), "Error: Left bound coordinate is larger than right bound coordinate"

                new_param_rule = Param_Rule(new_param_name, "LINE", 
                                            float(self.paramrule_lbound_entry.get()), 
                                            float(self.paramrule_rbound_entry.get()), 
                                            float(self.paramrule_lvalue_entry.get()), 
                                            float(self.paramrule_rvalue_entry.get()))

            elif (self.init_shape_selection.get() == "EXP"):
                assert (float(self.paramrule_rbound_entry.get()) <= current_layer.total_length), "Error: right bound coordinate too large"
                assert (float(self.paramrule_lbound_entry.get()) < float(self.paramrule_rbound_entry.get())), "Error: Left bound coordinate is larger than right bound coordinate"
                assert (float(self.paramrule_lvalue_entry.get()) != 0), "Error: left value cannot be 0"
                assert (float(self.paramrule_rvalue_entry.get()) != 0), "Error: right value cannot be 0"
                assert (float(self.paramrule_lvalue_entry.get()) * float(self.paramrule_rvalue_entry.get()) > 0), "Error: values must have same sign"
                new_param_rule = Param_Rule(new_param_name, "EXP", 
                                            float(self.paramrule_lbound_entry.get()), 
                                            float(self.paramrule_rbound_entry.get()), 
                                            float(self.paramrule_lvalue_entry.get()), 
                                            float(self.paramrule_rvalue_entry.get()))

            else:
                raise ValueError

        except ValueError:
            self.write(self.ICtab_status, "Error: Missing Parameters")
            return

        except AssertionError as oops:
            self.write(self.ICtab_status, str(oops))
            return

        self.module.add_param_rule(self.current_layer_name, new_param_name, new_param_rule)

        self.paramtoolkit_viewer_selection.set(new_param_name)
        self.update_paramrule_listbox(new_param_name)

        if self.module.system_ID in self.LGC_eligible_modules:
            if new_param_name == "delta_N" or new_param_name == "delta_P": 
                self.using_LGC[self.current_layer_name] = False
        self.update_IC_plot(plot_ID="recent")
        return

    def refresh_paramrule_listbox(self):
        """ Update the listbox to show rules for the selected param and display a snapshot of it"""
        if self.module.layers[self.current_layer_name].spacegrid_is_set:
            self.update_paramrule_listbox(self.paramtoolkit_viewer_selection.get())
            self.update_IC_plot(plot_ID="custom")
        return
    
    def update_paramrule_listbox(self, param_name):
        """ Grab current param's rules from module and show them in the param_rule listbox"""
        if not param_name:
            self.write(self.ICtab_status, "Select a parameter")
            return

        # 1. Clear the viewer
        self.hideall_paramrules()

        # 2. Write in the new rules
        current_param_rules = self.module.layers[self.current_layer_name].params[param_name].param_rules
        self.paramtoolkit_currentparam = param_name

        for param_rule in current_param_rules:
            self.active_paramrule_list.append(param_rule)
            self.active_paramrule_listbox.insert(len(self.active_paramrule_list) - 1,
                                                 param_rule.get())

        
        self.write(self.ICtab_status, "")

        return

    # These two reposition the order of param_rules
    def moveup_paramrule(self):
        """ Shift selected param rule upward. This changes order in which rules are used to generate a distribution. """
        try:
            currentSelectionIndex = self.active_paramrule_listbox.curselection()[0]
        except IndexError:
            return
        
        if (currentSelectionIndex > 0):
            # Two things must be done here for a complete swap:
            # 1. Change the order param rules appear in the box
            self.active_paramrule_list[currentSelectionIndex], self.active_paramrule_list[currentSelectionIndex - 1] = self.active_paramrule_list[currentSelectionIndex - 1], self.active_paramrule_list[currentSelectionIndex]
            self.active_paramrule_listbox.delete(currentSelectionIndex)
            self.active_paramrule_listbox.insert(currentSelectionIndex - 1, 
                                                 self.active_paramrule_list[currentSelectionIndex - 1].get())
            self.active_paramrule_listbox.selection_set(currentSelectionIndex - 1)

            # 2. Change the order param rules are applied when calculating Parameter's values
            self.module.swap_param_rules(self.current_layer_name, self.paramtoolkit_currentparam, 
                                           currentSelectionIndex)
            self.update_IC_plot(plot_ID="recent")
        return

    def movedown_paramrule(self):
        """ Shift selected param rule downward. This changes order in which rules are used to generate a distribution. """
        try:
            currentSelectionIndex = self.active_paramrule_listbox.curselection()[0] + 1
        except IndexError:
            return
        
        if (currentSelectionIndex < len(self.active_paramrule_list)):
            self.active_paramrule_list[currentSelectionIndex], self.active_paramrule_list[currentSelectionIndex - 1] = self.active_paramrule_list[currentSelectionIndex - 1], self.active_paramrule_list[currentSelectionIndex]
            self.active_paramrule_listbox.delete(currentSelectionIndex)
            self.active_paramrule_listbox.insert(currentSelectionIndex - 1, 
                                                 self.active_paramrule_list[currentSelectionIndex - 1].get())
            self.active_paramrule_listbox.selection_set(currentSelectionIndex)
            
            self.module.swap_param_rules(self.current_layer_name, self.paramtoolkit_currentparam, 
                                           currentSelectionIndex)
            self.update_IC_plot(plot_ID="recent")
        return

    def hideall_paramrules(self, doPlotUpdate=True):
        """ Wrapper - Call hide_paramrule() until listbox is empty"""
        while (self.active_paramrule_list):
            # These first two lines mimic user repeatedly selecting topmost paramrule in listbox
            self.active_paramrule_listbox.select_set(0)
            self.active_paramrule_listbox.event_generate("<<ListboxSelect>>")

            self.hide_paramrule()
        return

    def hide_paramrule(self):
        """Remove user-selected param rule from box (but don't touch Module's saved info)"""
        self.active_paramrule_list.pop(self.active_paramrule_listbox.curselection()[0])
        self.active_paramrule_listbox.delete(self.active_paramrule_listbox.curselection()[0])
        return
    
    def deleteall_paramrule(self, doPlotUpdate=True):
        """ Deletes all rules for current param. 
            Use reset_IC instead to delete all rules for every param
        """
        try:
            if (self.module.layers[self.current_layer_name].params[self.paramtoolkit_currentparam].param_rules):
                self.module.removeall_param_rules(self.current_layer_name, self.paramtoolkit_currentparam)
                self.hideall_paramrules()
                self.update_IC_plot(plot_ID="recent")
            
        except KeyError:
            return
        return

    def delete_paramrule(self):
        """ Deletes selected rule for current param. """
        try:
            if (self.module.layers[self.current_layer_name].params[self.paramtoolkit_currentparam].param_rules):
                try:
                    self.module.remove_param_rule(self.current_layer_name, self.paramtoolkit_currentparam, 
                                                    self.active_paramrule_listbox.curselection()[0])
                    self.hide_paramrule()
                    self.update_IC_plot(plot_ID="recent")
                except IndexError:
                    self.write(self.ICtab_status, "No rule selected")
                    return
        except KeyError:
            return
        return

    
    def add_listupload(self):
        """ Generate a parameter distribution using list from .txt file"""
        current_layer = self.module.layers[self.current_layer_name]
        msg = ["The following occured while importing the list:"]
        try:
            self.set_init_x()
            assert current_layer.spacegrid_is_set, "Error: could not set space grid"
        except ValueError:
            self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
            return

        except (AssertionError, Exception) as oops:
            self.write(self.ICtab_status, oops)
            return
        
        var = self.listupload_var_selection.get()
        is_edge = current_layer.params[var].is_edge
        
        valuelist_filename = tk.filedialog.askopenfilename(initialdir="", 
                                                           title="Select Values from text file", 
                                                           filetypes=[("Text files","*.txt")])
        if not valuelist_filename:
            return

        IC_values_list = []
        with open(valuelist_filename, 'r') as ifstream:
            for line in ifstream:
                if (not line.strip('\n') or "#" in line): 
                    continue

                else: IC_values_list.append(line.strip('\n'))

           
        temp_IC_values = np.zeros_like(current_layer.grid_x_nodes) if not is_edge else np.zeros_like(current_layer.grid_x_edges)

        try:
            IC_values_list.sort(key = lambda x:float(x[0:x.find('\t')]))
            
            if len(IC_values_list) < 2:
                raise ValueError("Not enough points in list")
        except Exception as e:
            msg.append("Error: Unable to read point list")
            msg.append(str(e))
            self.do_confirmation_popup("\n".join(msg), hide_cancel=True)
            self.root.wait_window(self.confirmation_popup)
            return
        
    
        for i in range(len(IC_values_list) - 1):
            try:
                first_valueset = extract_values(IC_values_list[i], '\t') #[x1, y(x1)]
                second_valueset = extract_values(IC_values_list[i+1], '\t') #[x2, y(x2)]
                
            except ValueError:
                msg.append("Warning: Unusual point list content")

            # Linear interpolate from provided param list to specified grid points
            lindex = to_index(first_valueset[0], current_layer.dx, 
                              current_layer.total_length, is_edge)
            rindex = to_index(second_valueset[0], current_layer.dx, 
                              current_layer.total_length, is_edge)
            
            if (first_valueset[0] - to_pos(lindex, current_layer.dx, is_edge) >= current_layer.dx / 2): 
                lindex += 1

            intermediate_x_indices = np.arange(lindex, rindex + 1, 1)

            for j in intermediate_x_indices: # y-y0 = (y1-y0)/(x1-x0) * (x-x0)
                try:
                    if (second_valueset[0] > current_layer.total_length): 
                        raise IndexError
                    temp_IC_values[j] = first_valueset[1] + (to_pos(j, current_layer.dx) - first_valueset[0]) * (second_valueset[1] - first_valueset[1]) / (second_valueset[0] - first_valueset[0])
                except IndexError:
                    msg.append("Warning: point {} out of bounds".format(second_valueset))
                    break
                except Exception as e:
                    msg.append("Warning: unable to assign value for position {}".format(j))
                    msg.append(str(e))
                    temp_IC_values[j] = 0
                
        if self.module.system_ID in self.LGC_eligible_modules:
            if var == "delta_N" or var == "delta_P": 
                self.using_LGC[self.current_layer_name] = False
        
        self.paramtoolkit_currentparam = var
        self.deleteall_paramrule()
        current_layer.params[var].value = temp_IC_values
        self.update_IC_plot(plot_ID="listupload")
        self.update_IC_plot(plot_ID="recent")
        
        if len(msg) > 1:
            self.do_confirmation_popup("\n".join(msg), hide_cancel=True)
            self.root.wait_window(self.confirmation_popup)
        return

    def update_IC_plot(self, plot_ID):
        """ Plot selected parameter distribution on Initial Condition tab."""
        if plot_ID=="recent": 
            plot = self.recent_param_subplot
        elif plot_ID=="custom": 
            plot = self.custom_param_subplot
        elif plot_ID=="LGC": 
            plot = self.LGC_subplot
        elif plot_ID=="listupload": 
            plot = self.listupload_subplot
        elif plot_ID=="clearall":
            self.recent_param_subplot.cla()
            self.recent_param_fig.canvas.draw()
            self.custom_param_subplot.cla()
            self.custom_param_fig.canvas.draw()
            self.LGC_subplot.cla()
            self.LGC_fig.canvas.draw()
            self.listupload_subplot.cla()
            self.listupload_fig.canvas.draw()
            return
            
        plot.cla()

        if plot_ID=="LGC":
            param_name="delta_N"
        else: 
            param_name = self.paramtoolkit_currentparam
        param_obj = self.module.layers[self.current_layer_name].params[param_name]
        grid_x = (self.module.layers[self.current_layer_name].grid_x_edges 
                  if param_obj.is_edge else self.module.layers[self.current_layer_name].grid_x_nodes)
        # Support for constant value shortcut: temporarily create distribution
        # simulating filling across module with that value
        val_array = to_array(param_obj.value, len(self.module.layers[self.current_layer_name].grid_x_nodes), 
                             param_obj.is_edge)

        plot.set_yscale(autoscale(val_array=val_array))

        if self.sys_flag_dict['symmetric_system'].value():
            plot.plot(np.concatenate((-np.flip(grid_x), grid_x), axis=0), 
                      np.concatenate((np.flip(val_array), val_array), axis=0), 
                      label=param_name)

            ymin, ymax = plot.get_ylim()
            plot.fill([-grid_x[-1], 0, 0, -grid_x[-1]], [ymin, ymin, ymax, ymax], 
                      'b', alpha=0.1, edgecolor='r')
        else:
            plot.plot(grid_x, val_array, label=param_name)

        plot.set_xlabel("x {}".format(self.module.layers[self.current_layer_name].length_unit))
        plot.set_ylabel("{} {}".format(param_name, param_obj.units))
        
        if plot_ID=="recent": 
            self.update_system_summary()
            
            plot.set_title("Recently Changed: {}".format(param_name))
            self.recent_param_fig.tight_layout()
            self.recent_param_fig.canvas.draw()
        elif plot_ID=="custom": 
            plot.set_title("Snapshot: {}".format(param_name))
            self.custom_param_fig.tight_layout()
            self.custom_param_fig.canvas.draw()
        elif plot_ID=="LGC": 
            if self.sys_flag_dict['check_do_ss'].value():
                plot.set_title("Carriers added per ns.")
            else:
                plot.set_title("Initial carrier profile")
            self.LGC_fig.tight_layout()
            self.LGC_fig.canvas.draw()
        elif plot_ID=="listupload": 
            plot.set_title("Recent list upload")
            self.listupload_fig.tight_layout()
            self.listupload_fig.canvas.draw()

        self.write(self.ICtab_status, "Initial Condition Updated")
        return

    ## Initial Condition I/O

    def create_batch_init(self):
        """ Workhorse of batch init tool. Creates a batch of similar IC files based on the inputted parameter grid."""
        warning_mssg = ["The following occured while creating the batch:"]
        try:
            batch_values = {}

            for batchable in self.batchables_array:
                if batchable.param_name.get():
                    batch_values[batchable.param_name.get()] = extract_values(batchable.tk_entrybox.get(), ' ')
            
            if not batch_values: # If no batch params were selected
                raise ValueError
        except ValueError:
            self.write(self.batch_status, "Error: Invalid batch values")
            return

        print(batch_values)

        try:
            batch_dir_name = self.batch_name_entry.get()
            assert batch_dir_name, "Error: Batch folder must have a name"
            assert check_valid_filename(batch_dir_name), "File names may not contain certain symbols such as ., <, >, /, \\, *, ?, :, \", |"
        except (AssertionError, Exception) as e:
            self.write(self.batch_status, e)
            return

        try:
            os.mkdir(os.path.join(self.default_dirs["Initial"], batch_dir_name))
        except FileExistsError:
            self.write(self.batch_status, 
                       "Error: {} folder already exists".format(batch_dir_name))
            return
        
        # Record the original values of the module, so we can restore them after the batch algo finishes
        original_param_values = {}
        for layer in self.module.layers:
            original_param_values[layer] = {}
            for param_name, param in self.module.layers[layer].params.items():
                original_param_values[layer][param_name] = param.value
                
        # This algorithm was shamelessly stolen from our bay.py script...                
        batch_combinations = get_all_combinations(batch_values)        
                
        # Apply each combination to module, going through LGC if necessary
        for batch_set in batch_combinations:
            filename = ""
            for b in batch_set:
                layer, param = b.split('-')
                filename += str("__{}_{:.4e}".format(b, batch_set[b]))
                
                if (self.module.system_ID in self.LGC_eligible_modules 
                    and layer in self.LGC_values
                    and param in self.LGC_values[layer]):
                    self.enter(self.LGC_entryboxes_dict[param], str(batch_set[b]))
                    
                else:
                    self.module.layers[layer].params[param].value = batch_set[b]

                
            if (self.module.system_ID in self.LGC_eligible_modules
                and any(self.using_LGC.values())): 
                self.add_LGC()
                
            try:
                self.write_init_file(os.path.join(self.default_dirs["Initial"], 
                                                    batch_dir_name,
                                                    "{}.txt".format(filename)))
            except Exception as e:
                warning_mssg.append("Error: failed to create batch file {}".format(filename))
                warning_mssg.append(str(e))
                
        # Restore the original values of module
        for layer in self.module.layers:
            for param_name, param in self.module.layers[layer].params.items():
                param.value = original_param_values[layer][param_name]
        
        
        if len(warning_mssg) > 1:
            self.do_confirmation_popup("\n".join(warning_mssg), hide_cancel=True)
        else:
            self.do_confirmation_popup("Batch {} created successfully".format(batch_dir_name), hide_cancel=True)
            
        self.root.wait_window(self.confirmation_popup)
        return
    

    def save_ICfile(self):
        """Wrapper for write_init_file() - this one is for IC files user saves from the Initial tab and is called when the Save button is clicked"""
    
        try:
            assert all([layer.spacegrid_is_set for name, layer in self.module.layers.items()]), "Error: set all space grids first"
            new_filename = tk.filedialog.asksaveasfilename(initialdir=self.default_dirs["Initial"], 
                                                           title="Save IC text file", 
                                                           filetypes=[("Text files","*.txt")])
            
            if not new_filename: 
                return

            if new_filename.endswith(".txt"): 
                new_filename = new_filename[:-4]
            self.write_init_file(new_filename + ".txt")

        except AssertionError as oops:
            self.write(self.ICtab_status, str(oops))
            
        except ValueError as uh_Oh:
            print(uh_Oh)
            
        return

    def write_init_file(self, newFileName, dir_name=""):
        """ Write current state of module into an initial condition (IC) file."""
        try:
            with open(newFileName, "w+") as ofstream:
                print(dir_name + newFileName + " opened successfully")

                # We don't really need to note down the time of creation, but it could be useful for interaction with other programs.
                ofstream.write("$$ INITIAL CONDITION FILE CREATED ON " + str(datetime.datetime.now().date()) + " AT " + str(datetime.datetime.now().time()) + "\n")
                ofstream.write("System_class: {}\n".format(self.module.system_ID))
                ofstream.write("f$ System Flags:\n")
                
                for flag in self.module.flags_dict:
                    ofstream.write("{}: {}\n".format(flag, self.sys_flag_dict[flag].value()))
                      
                for layer_name, layer in self.module.layers.items():
                    ofstream.write("L$: {}\n".format(layer_name))
                    ofstream.write("p$ Space Grid:\n")
                    ofstream.write("Total_length: {}\n".format(layer.total_length))
                    ofstream.write("Node_width: {}\n".format(layer.dx))
                
                    ofstream.write("p$ System Parameters:\n")
                
                    # Saves occur as-is: any missing parameters are saved with whatever default value module gives them
                    for param in layer.params:
                        param_values = layer.params[param].value
                        if isinstance(param_values, np.ndarray):
                            # Write the array in a more convenient format
                            ofstream.write("{}: {:.8e}".format(param, param_values[0]))
                            for value in param_values[1:]:
                                ofstream.write("\t{:.8e}".format(value))
                                
                            ofstream.write('\n')
                        else:
                            # The param value is just a single constant
                            ofstream.write("{}: {}\n".format(param, param_values))
                      
                    if self.module.system_ID in self.LGC_eligible_modules and self.using_LGC[layer_name]:
                        ofstream.write("p$ Laser Parameters\n")
                        for laser_param in self.LGC_values[layer_name]:
                            ofstream.write("{}: {}\n".format(laser_param,
                                                             self.LGC_values[layer_name][laser_param]))
    
                        ofstream.write("p$ Laser Options\n")
                        for laser_option in self.LGC_options[layer_name]:
                            ofstream.write("{}: {}\n".format(laser_option, 
                                                             self.LGC_options[layer_name][laser_option]))
                
        except OSError:
            self.write(self.ICtab_status, "Error: failed to create IC file")
            return

        self.write(self.ICtab_status, "IC file generated")
        return

    
    def select_init_file(self):
        """Wrapper for load_ICfile with user selection from IC tab"""
        self.IC_file_name = tk.filedialog.askopenfilename(initialdir=self.default_dirs["Initial"], 
                                                          title="Select IC text files", 
                                                          filetypes=[("Text files","*.txt")])
        if not self.IC_file_name: 
            return

        self.load_ICfile()
        return

    def load_ICfile(self, cycle_through_IC_plots=True):
        """Read parameters from IC file and store into module."""

        warning_flag = False
        warning_mssg = ""

        try:
            print("Poked file: {}".format(self.IC_file_name))
            with open(self.IC_file_name, 'r') as ifstream:
                init_param_values_dict = {}

                flag_values_dict = {}
                LGC_values = {}
                LGC_options = {}
                total_length = {}
                dx = {}

                initFlag = 0
                
                if not ("$$ INITIAL CONDITION FILE CREATED ON") in next(ifstream):
                    raise OSError("Error: this file is not a valid TEDs file")
                
                system_class = next(ifstream).strip('\n')
                system_class = system_class[system_class.find(' ') + 1:]
                if not system_class == self.module.system_ID:
                    raise ValueError("Error: selected file is not a {}".format(self.module.system_ID))
                                
                # Extract parameters, ICs
                for line in ifstream:
                    line = line.strip('\n')
                    line_split = get_split_and_clean_line(line)

                    if ("#" in line) or not line:
                        continue

                    # There are three "$" markers in an IC file: "Space Grid", "System Parameters" and "System Flags"
                    # each corresponding to a different section of the file

                    if "p$ Space Grid" in line:
                        print("Now searching for space grid: length and dx")
                        initFlag = 'g'
                        continue
                        
                    elif "p$ System Parameters" in line:
                        print("Now searching for system parameters...")
                        initFlag = 'p'
                        continue
                        
                    elif "f$" in line:
                        print("Now searching for flag values...")
                        initFlag = 'f'
                        continue
                        
                    elif "$ Laser Parameters" in line:
                        print("Now searching for laser parameters...")
                        if self.module.system_ID not in self.LGC_eligible_modules:
                            print("Warning: laser params found for unsupported module (these will be ignored)")
                        initFlag = 'l'
                        continue
                        
                    elif "$ Laser Options" in line:
                        print("Now searching for laser options...")
                        if self.module.system_ID not in self.LGC_eligible_modules:
                            print("Warning: laser options found for unsupported module (these will be ignored)")
                        initFlag = 'o'
                        continue

                    if len(line_split) > 1:
                    
                        if "L$" in line:
                            layer_name = line_split[1]
                            init_param_values_dict[layer_name] = {}
                            LGC_values[layer_name] = {}
                            LGC_options[layer_name] = {}
                            print("Found layer {}".format(layer_name))
                            continue
                        
                        if (initFlag == 'g'):
                            if line.startswith("Total_length"):
                                total_length[layer_name] = (line_split[1])
                                
                            elif line.startswith("Node_width"):
                                dx[layer_name] = (line_split[1])
                            
                        elif (initFlag == 'p'):
                            init_param_values_dict[layer_name][line_split[0]] = (line_split[1])

                        elif (initFlag == 'f'):
                            flag_values_dict[line_split[0]] = (line_split[1])
                            
                        elif (initFlag == 'l'):
                            LGC_values[layer_name][line_split[0]] = (line_split[1])
                            
                        elif (initFlag == 'o'):
                            LGC_options[layer_name][line_split[0]] = (line_split[1])


        except Exception as oops:
            self.write(self.ICtab_status, oops)
            return

        ## At this point everything from the file has been read. 
        # Whether those values are valid is another story, but having a valid space grid is essential.
        
        try:
            for layer_name in self.module.layers:
                total_length[layer_name] = float(total_length[layer_name])
                dx[layer_name] = float(dx[layer_name])
                if ((total_length[layer_name] <= 0) or (dx[layer_name] <= 0)
                    or (dx[layer_name] > total_length[layer_name] / 2)):
                    raise ValueError
        except Exception:
            self.write(self.ICtab_status, "Error: invalid space grid for layer {}".format(layer_name))
            return
        
        # Clear values in any IC generation areas; this is done to minimize ambiguity between IC's that came from the recently loaded file and any other values that may exist on the GUI
        if self.module.system_ID in self.LGC_eligible_modules:
            for key in self.LGC_entryboxes_dict:
                self.enter(self.LGC_entryboxes_dict[key], "")
            
        for layer_name, layer in self.module.layers.items():
            self.current_layer_name = layer_name
            for param in layer.params:
                self.paramtoolkit_currentparam = param
                
                self.update_paramrule_listbox(param)            
                self.deleteall_paramrule()
                
                layer.params[param].value = 0

            self.set_thickness_and_dx_entryboxes(state='unlock')
            layer.total_length = None
            layer.dx = None
            layer.grid_x_edges = []
            layer.grid_x_nodes = []
            layer.spacegrid_is_set = False


        for layer_name, layer in self.module.layers.items():
            self.current_layer_name = layer_name
            try:
                self.set_thickness_and_dx_entryboxes(state='unlock')
                self.enter(self.thickness_entry, total_length[self.current_layer_name])
                self.enter(self.dx_entry, dx[self.current_layer_name])
                self.set_init_x()
                assert layer.spacegrid_is_set, "Error: could not set space grid"
            except ValueError:
                self.write(self.ICtab_status, "Error: invalid thickness or space stepsize")
                return
    
            except (AssertionError, Exception) as oops:
                self.write(self.ICtab_status, oops)
                return

        for flag in self.module.flags_dict:
            # All we need to do here is mark the appropriate GUI elements as selected
            try:
                self.sys_flag_dict[flag].tk_var.set(flag_values_dict[flag])
            except Exception:
                warning_mssg += "\nWarning: could not apply value for flag: {}".format(flag)
                warning_mssg += "\nFlags must have integer value 1 or 0"
                warning_flag += 1
            
        for layer_name, layer in self.module.layers.items():
            self.current_layer_name = layer_name
            for param_name, param in layer.params.items():
                try:
                    new_value = init_param_values_dict[layer_name][param_name]
                    if '\t' in new_value: # If an array / list of points was stored
                        assert param.is_space_dependent
                        param.value = np.array(extract_values(new_value, '\t'))
                    else: 
                        param.value = float(new_value)
                    
                    self.paramtoolkit_currentparam = param_name
                    if cycle_through_IC_plots: 
                        self.update_IC_plot(plot_ID="recent")
                except Exception:
                    warning_mssg += ("\nWarning: could not apply value for param: {}".format(param_name))
                    warning_flag += 1
                        
            if self.module.system_ID in self.LGC_eligible_modules: 
                self.using_LGC[layer_name] = bool(LGC_values[layer_name])
                if LGC_values[layer_name]:
                    for laser_param in LGC_values[layer_name]:
                        if float(LGC_values[layer_name][laser_param]) > 1e3:
                            self.enter(self.LGC_entryboxes_dict[laser_param], "{:.3e}".format(float(LGC_values[layer_name][laser_param])))
                        else:
                            self.enter(self.LGC_entryboxes_dict[laser_param], LGC_values[layer_name][laser_param])
                
                for laser_option in LGC_options[layer_name]:
                    self.LGC_optionboxes[laser_option].set(LGC_options[layer_name][laser_option])
            
                if self.using_LGC[layer_name]:
                    self.LGC_layer.set(layer_name)
                    self.add_LGC()
            
        # Sync display to latest layer
        self.current_layer_selection.set(self.current_layer_name)
        self.change_layer()
        if not warning_flag: 
            self.write(self.ICtab_status, "IC file loaded successfully")
        else: 
            self.write(self.ICtab_status, "IC file loaded with {} issue(s); see console".format(warning_flag))
            self.do_confirmation_popup(warning_mssg, hide_cancel=True)
            self.root.wait_window(self.confirmation_popup)
        return

    # Data I/O

    def export_plot(self, from_integration):
        """ Write out currently plotted data as .csv file"""
        if from_integration:
            plot_ID = self.active_integrationplot_ID.get()
            datagroup = self.integration_plots[plot_ID].datagroup
            plot_info = self.integration_plots[plot_ID]
            if not datagroup.size(): 
                return
            
            where_layer = self.module.find_layer(datagroup.type)
            if plot_info.mode == "Current time step": 
                paired_data = [[datagroup.datasets[key].grid_x, 
                                datagroup.datasets[key].data * 
                                self.module.layers[where_layer].convert_out[datagroup.type] *
                                self.module.layers[where_layer].iconvert_out[datagroup.type]] 
                               for key in datagroup.datasets]

                header = "{} {}, {}".format(plot_info.x_param, 
                                            self.module.layers[where_layer].params[plot_info.x_param].units, 
                                            datagroup.type)

            else: # if self.I_plot.mode == "All time steps"
                paired_data = datagroup.build(self.module.layers[where_layer].convert_out, self.module.layers[where_layer].iconvert_out)
                paired_data = np.array(list(map(list, itertools.zip_longest(*paired_data, fillvalue=-1))))
                
                header = "".join(["Time [ns]," + 
                                  datagroup.datasets[key].filename + 
                                  "," for key in datagroup.datasets])
                

        else:
            plot_ID = self.active_analysisplot_ID.get()
            if not self.analysis_plots[plot_ID].datagroup.size(): 
                return
            where_layer = self.module.find_layer(self.analysis_plots[plot_ID].datagroup.type)
            paired_data = self.analysis_plots[plot_ID].datagroup.build(self.module.layers[where_layer].convert_out)
            
            # Transpose a list of arrays
            paired_data = np.array(list(map(list, itertools.zip_longest(*paired_data, fillvalue=-1))))
            
            header = "".join(["x {},".format(self.module.layers[where_layer].length_unit) + 
                              self.analysis_plots[plot_ID].datagroup.datasets[key].filename + 
                              "," for key in self.analysis_plots[plot_ID].datagroup.datasets])

        export_filename = tk.filedialog.asksaveasfilename(initialdir=self.default_dirs["PL"], 
                                                          title="Save data", 
                                                          filetypes=[("csv (comma-separated-values)","*.csv")])
        
        # Export to .csv
        if export_filename:
            try:
                if export_filename.endswith(".csv"): 
                    export_filename = export_filename[:-4]
                np.savetxt("{}.csv".format(export_filename), paired_data, 
                           delimiter=',', header=header)
                self.write(self.analysis_status, "Export complete")
            except PermissionError:
                self.write(self.analysis_status, "Error: unable to access PL export destination")
        
        return
    
    def export_timeseries(self, tspopup_ID, tail=False):
        paired_data = list(self.active_timeseries[tspopup_ID])
        
        # paired_data = [(tag, tgrid, values), (...,...,...), ...]
        # Unpack list of array tuples into list of arrays

        header = []
        while isinstance(paired_data[0], tuple):
            header.append("Time [ns]")
            header.append(paired_data[0][0])
            # unpack
            if tail:
                paired_data.append(paired_data[0][1][-10:])
                paired_data.append(paired_data[0][2][-10:])
            else:
                paired_data.append(paired_data[0][1])
                paired_data.append(paired_data[0][2])
            paired_data.pop(0)
            
        paired_data = np.array(list(map(list, itertools.zip_longest(*paired_data, fillvalue=-1))))
        
        
        export_filename = tk.filedialog.asksaveasfilename(initialdir=self.default_dirs["PL"], 
                                                          title="Save data", 
                                                          filetypes=[("csv (comma-separated-values)","*.csv")])
        # Export to .csv
        if export_filename:
            try:
                if export_filename.endswith(".csv"): 
                    export_filename = export_filename[:-4]
                np.savetxt("{}.csv".format(export_filename), paired_data, 
                           delimiter=',', header=",".join(header))
                self.write(self.analysis_status, "Timeseries export complete")
            except PermissionError:
                self.write(self.analysis_status, "Error: unable to access PL export destination")
        
        return
            
    def export_overview(self):
        try:
            # Has an overview been calculated?
            self.overview_values
        except AttributeError:
            return
        
        output_name = self.overview_var_selection.get()
        
        layer_name, output_name = output_name.split(": ")
        paired_data = self.overview_values[layer_name][output_name]

        if not isinstance(paired_data, np.ndarray):
            print("No data found for export")
            return
        
        # Grab x axis from matplotlib by force
        grid_x = self.overview_subplots[layer_name][output_name].lines[0]._xorig
        paired_data = np.vstack((grid_x, paired_data)).T
        
        export_filename = tk.filedialog.asksaveasfilename(initialdir=self.default_dirs["PL"], 
                                                          title="Save data", 
                                                          filetypes=[("csv (comma-separated-values)","*.csv")])
        # Export to .csv
        if export_filename:
            try:
                if export_filename.endswith(".csv"): 
                    export_filename = export_filename[:-4]
                np.savetxt("{}.csv".format(export_filename), paired_data, 
                           delimiter=',')
                print("Export from overview complete")
            except PermissionError:
                print("Error: unable to access export destination")
        
        return

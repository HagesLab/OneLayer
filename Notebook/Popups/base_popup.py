import matplotlib
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import tkinter as tk

# This lets us pass params to functions called by tkinter buttons
# but it is also the reason why we cant extract more of "popup"
# related methods out of notebook.py
from functools import partial 

from config import init_logging
logger = init_logging(__name__)

class Popup():
    
    def __init__(self, nb, grab=False):
        self.toplevel = tk.Toplevel(nb.root)
        
        self.toplevel.protocol("WM_DELETE_WINDOW", self.close)
        
        if grab:
            self.toplevel.grab_set()        

    def close(self):
        self.toplevel.destroy()
        return


## Functions to create popups and manage
def do_module_popup(nb):
    """ Popup for selecting the active module (e.g. Nanowire) """
    nb.select_module_popup = tk.Toplevel(nb.root)
    tk.Label(nb.select_module_popup, 
                text="The following TEDs modules were found; "
                "select one to continue: ").grid(row=0,column=0)
    
    nb.module_names = list(nb.module_list.keys())
    nb.module_listbox = tk.Listbox(nb.select_module_popup, width=40, height=10)
    nb.module_listbox.grid(row=1,column=0)
    nb.module_listbox.delete(0,tk.END)
    nb.module_listbox.insert(0,*(nb.module_names))
    
    tk.Button(
        nb.select_module_popup,
        text="Continue", 
        command=partial(nb.on_select_module_popup_close, True)
        ).grid(row=2,column=0)
    
    nb.select_module_popup.protocol(
        "WM_DELETE_WINDOW", 
        partial(nb.on_select_module_popup_close, continue_=False))
    
    nb.select_module_popup.attributes(
        "-topmost", True)
    nb.select_module_popup.after_idle(
        nb.select_module_popup.attributes,
        '-topmost', False)
    
    nb.select_module_popup.grab_set()

    return nb

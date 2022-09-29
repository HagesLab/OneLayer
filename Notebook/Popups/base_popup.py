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
        self.nb = nb
        
        if grab:
            self.toplevel.grab_set()        

    def close(self):
        self.toplevel.destroy()
        return
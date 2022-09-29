# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 18:29:19 2022

@author: cfai2
"""
import tkinter as tk

from functools import partial

from Notebook.Popups.base_popup import Popup

class ConfirmationPopup(Popup):
    """ General purpose popup for important operations (e.g. deleting something)
        which should require user confirmation."""
    
    def __init__(self, nb, text, hide_cancel=False):
        super().__init__(nb, grab=True)

        tk.Message(self.toplevel, text=text, 
                    width=(float(nb.root.winfo_screenwidth()) / 4)).grid(row=0,column=0, columnspan=2)
        
        self.toplevel.protocol("WM_DELETE_WINDOW", 
                               partial(self.close, continue_=False))
        if not hide_cancel:
            tk.Button(self.toplevel, text="Cancel", 
                        command=partial(self.close, 
                                        continue_=False)).grid(row=1,column=0)
        
        tk.Button(self.toplevel, text='Continue', 
                    command=partial(self.close, 
                                    continue_=True)).grid(row=1,column=1)
        return
    
    def close(self, continue_=False):
        """ Inform caller of do_confirmation_popup of whether user confirmation 
            was received 
        """
        self.nb.confirmed = continue_
        super().close()
        return
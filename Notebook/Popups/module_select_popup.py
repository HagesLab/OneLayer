# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:01:02 2022

@author: cfai2
"""
import tkinter as tk

from functools import partial

from Notebook.Popups.base_popup import Popup

class ModuleSelectPopup(Popup):
    """ Popup for selecting the active module (e.g. Nanowire) """
    
    def __init__(self, nb, logger):
        super().__init__(nb, grab=True)
        
        tk.Label(self.toplevel, 
                    text="The following TEDs modules were found; "
                    "select one to continue: ").grid(row=0,column=0)
        
        self.module_names = list(nb.module_list.keys())
        self.module_listbox = tk.Listbox(self.toplevel, width=40, height=10)
        self.module_listbox.grid(row=1,column=0)
        self.module_listbox.delete(0,tk.END)
        self.module_listbox.insert(0,*(self.module_names))
        
        self.toplevel.protocol("WM_DELETE_WINDOW", 
                               partial(self.close, logger, False))
        
        tk.Button(
            self.toplevel,
            text="Continue", 
            command=partial(self.close, logger, True)
            ).grid(row=2,column=0)
                
        self.toplevel.attributes(
            "-topmost", True)
        self.toplevel.after_idle(
            self.toplevel.attributes,
            '-topmost', False)
        
        return
    
    def close(self, logger=None, continue_=False):
        """ Do basic verification checks defined by OneD_Model.verify() 
            and inform tkinter of selected module 
        """
        try:
            if continue_:
                self.nb.verified=False
                self.nb.module = self.nb.module_list[self.module_names[self.module_listbox.curselection()[0]]]()
                self.nb.module.verify()
                self.nb.verified=True
                
            super().close()

        except IndexError:
            logger.error("No module selected: Select a module from the list")
        except AssertionError as oops:
            logger.error("Error: could not verify selected module")
            logger.error(str(oops))

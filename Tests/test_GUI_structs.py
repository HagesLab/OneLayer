# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:18:49 2022

@author: cfai2
"""

import unittest
import numpy as np
import tkinter as tk
from tkinter import ttk

from GUI_structs import Param_Rule
from GUI_structs import Flag
from GUI_structs import Batchable
from GUI_structs import Data_Set
from GUI_structs import Raw_Data_Set as RDS
from GUI_structs import Integrated_Data_Set as IDS

class TestUtils(unittest.TestCase):
    
    def test_Param_Rule_get(self):
        variable = "X"
        type_ = "invalid type"
        l_bound = 1
        
        with self.assertRaises(ValueError):
            p = Param_Rule(variable, type_, l_bound, r_bound=-1, l_boundval=-1, r_boundval=-1)
            p.get()
        
        type_ = "POINT"
        lv = 20
        p = Param_Rule(variable, type_, l_bound, r_bound=-1, l_boundval=lv, r_boundval=-1)
        self.assertEqual(p.get(), r"X: POINT at x=1.0000e+00 with value: 2.0000e+01")
        
        type_ = "FILL"
        r = 2
        p = Param_Rule(variable, type_, l_bound, r_bound=r, l_boundval=lv, r_boundval=-1)
        self.assertEqual(p.get(), r"X: FILL from x=1.0000e+00 to 2.0000e+00 with value: 2.0000e+01")
        
        type_ = "LINE"
        rv = 40
        p = Param_Rule(variable, type_, l_bound, r_bound=r, l_boundval=lv, r_boundval=rv)
        self.assertEqual(p.get(), r"X: LINE from x=1.0000e+00 to 2.0000e+00 "
                         "with left value: 2.0000e+01 and right value: 4.0000e+01")
        
        type_ = "EXP"
        p = Param_Rule(variable, type_, l_bound, r_bound=r, l_boundval=lv, r_boundval=rv)
        self.assertEqual(p.get(), r"X: EXP from x=1.0000e+00 to 2.0000e+00 "
                         "with left value: 2.0000e+01 and right value: 4.0000e+01")
        
        self.assertEqual(p.variable, variable)
        self.assertEqual(p.type, type_)
        self.assertEqual(p.l_bound, l_bound)
        self.assertEqual(p.r_bound, r)
        self.assertEqual(p.l_boundval, lv)
        self.assertEqual(p.r_boundval, rv)
        return
   
    def test_Flag(self):
        root = tk.Tk()
        name = "name"
        flag = Flag(root, name)
        flag.set(1)
        self.assertEqual(flag.value(), 1)
        return
    
    def test_Batchable(self):
        root = tk.Tk()
        name = "X"
        optionmenu = tk.ttk.OptionMenu(root, name, "", "")
        entry = tk.ttk.Entry(root, width=80)
        b = Batchable(optionmenu, entry, name)
        
        self.assertIsInstance(b.tk_optionmenu, tk.ttk.OptionMenu)
        self.assertIsInstance(b.tk_entrybox, tk.ttk.Entry)
        
        self.assertEqual(b.param_name, name)
        return
    
    def test_Data_Set(self):
        data = -1
        grid_x = -1
        params_dict = {}
        flags = {}
        type_ = "X"
        filename = "__file__"
        
        ds = Data_Set(data, grid_x, params_dict, flags, type_, filename)
        
        tag = ds.tag(for_matplotlib=False)
        self.assertEqual(tag, "__file___X")
        
        tag = ds.tag(for_matplotlib=True)
        self.assertEqual(tag, "file___X")
        
        self.assertEqual(ds.data, data)
        self.assertEqual(ds.grid_x, grid_x)
        self.assertEqual(ds.params_dict, params_dict)
        self.assertEqual(ds.flags, flags)
        self.assertEqual(ds.type, type_)
        self.assertEqual(ds.filename, filename)
        return
    
    def test_RDS(self):
        data = np.linspace(0,1,100)
        grid_x = np.linspace(0, 100, 100)
        node_x = np.linspace(0, 100, 100)
        total_time = 1
        dt = 0.025
        params_dict = {'a':1}
        flags = {'f':1}
        type_ = "X"
        filename = "__file__"
        show_index = 0
        rds = RDS(data, grid_x, node_x, total_time, dt, params_dict, flags, type_, filename, show_index)
        
        np.testing.assert_equal(rds.node_x, node_x)
        self.assertEqual(rds.show_index, show_index)
        self.assertEqual(rds.total_time, total_time)
        self.assertEqual(rds.dt, dt)
        self.assertEqual(rds.num_tsteps, total_time / dt)
        
        stac = rds.build()
        expected_stac = np.array([grid_x, data])
        np.testing.assert_equal(stac, expected_stac)
        return
    
    def test_IDS(self):
        data = np.linspace(0,1,100)
        grid_x = np.linspace(0, 100, 100)
        total_time = 1
        dt = 0.025
        params_dict = {'a':1}
        flags = {'f':1}
        type_ = "X"
        filename = "__file__"
        ids = IDS(data, grid_x, total_time, dt, params_dict, flags, type_, filename)
        self.assertEqual(ids.total_time, total_time)
        self.assertEqual(ids.dt, dt)
        return
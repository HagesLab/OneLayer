# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:18:49 2022

@author: cfai2
"""

import unittest
import numpy as np

from GUI_structs import Raw_Data_Set as RDS
from GUI_structs import Data_Group as DG
from GUI_structs import Raw_Data_Group as RDG
from GUI_structs import Integrated_Data_Group as IDG

class TestDG(unittest.TestCase):
    def setUp(self):
        self.dg = DG()
        self.dg.type = "X"
        self.dg.dt = 0.025
        self.dg.total_t = 1
        return
    
    def test_init(self):
        self.assertEqual(self.dg.type, "X")
        self.assertEqual(self.dg.datasets, {})
        self.assertEqual(self.dg.dt, 0.025)
        self.assertEqual(self.dg.total_t, 1)
        return
    
    def test_minmaxclear(self):
        dataset1 = np.linspace(0,1,100)
        dataset2 = np.linspace(1000,100,100)
        
        self.dg.datasets["1"] = dataset1
        self.dg.datasets["2"] = dataset2
        
        self.assertEqual(self.dg.get_minval(), 0)
        self.assertEqual(self.dg.get_maxval(), 1000)
        
        self.assertEqual(self.dg.size(), 2)
        self.dg.clear()
        self.assertEqual(self.dg.size(), 0)
        return
    
class TestRDG(unittest.TestCase):
    def setUp(self):
        self.rdg = RDG()
        
        self.data1 = np.linspace(0,100,101)
        self.grid_x1 = np.linspace(0,100,101)
        total_time = 1
        dt = 0.025
        params_dict = {}
        flags = {}
        type_ = "X"
        filename = "f1"
        ds1 = RDS(self.data1, self.grid_x1, self.grid_x1, total_time, dt, params_dict, flags, type_, filename, show_index=0)
        
        self.data2 = 10 * np.linspace(0,100,101)
        self.grid_x2 = 10 * np.linspace(0,100,101)
        total_time = 1
        dt = 0.025
        params_dict = {}
        flags = {}
        type_ = "X"
        filename = "f2"
        ds2 = RDS(self.data2, self.grid_x2, self.grid_x2, total_time, dt, params_dict, flags, type_, filename, show_index=0)
        
        data = 10 * np.linspace(0,100,101)
        grid_x = 10 * np.linspace(0,100,101)
        total_time = 1
        dt = 0.025
        params_dict = {}
        flags = {}
        type_ = "wrong_type"
        filename = "f3"
        ds3 = RDS(data, grid_x, grid_x, total_time, dt, params_dict, flags, type_, filename, show_index=0)
        
        self.rdg.add(ds1)
        self.rdg.add(ds2)
        self.rdg.add(ds3) # Should be denied
        return
    
    def test_init(self):
        self.assertEqual(self.rdg.dt, 0.025)
        self.assertEqual(self.rdg.total_t, 1)
        self.assertEqual(self.rdg.type, "X")
        self.assertEqual(len(self.rdg.datasets), 2)
        return
    
    def test_build(self):
        convert_out = {"X":0.1}
        out = self.rdg.build(convert_out)
        
        np.testing.assert_equal(out, [self.grid_x1, 0.1*self.data1, self.grid_x2, 0.1*self.data2])
        return
    
    def test_max_x(self):
        out = self.rdg.get_max_x(is_edge=True)
        self.assertEqual(out, 1000)
        out = self.rdg.get_max_x(is_edge=False) # Should add dx(=10) to the limit
        self.assertEqual(out, 1010)
        
    def test_max_time(self):
        out = self.rdg.get_maxtime()
        self.assertEqual(out, 1)
        
    def test_max_nt(self):
        out = self.rdg.get_maxnumtsteps()
        self.assertEqual(out, 1 / 0.025)
        
class TestIDG(unittest.TestCase):
    def setUp(self):
        self.idg = IDG()
        
        self.data1 = np.linspace(0,100,101)
        self.grid_x1 = np.linspace(0,100,101)
        total_time = 1
        dt = 0.025
        params_dict = {}
        flags = {}
        type_ = "X"
        filename = "f1"
        ds1 = RDS(self.data1, self.grid_x1, self.grid_x1, total_time, dt, params_dict, flags, type_, filename, show_index=0)
        
        self.data2 = 10 * np.linspace(0,100,101)
        self.grid_x2 = 10 * np.linspace(0,100,101)
        total_time = 1
        dt = 0.025
        params_dict = {}
        flags = {}
        type_ = "X"
        filename = "f2"
        ds2 = RDS(self.data2, self.grid_x2, self.grid_x2, total_time, dt, params_dict, flags, type_, filename, show_index=0)
        
        data = 10 * np.linspace(0,100,101)
        grid_x = 10 * np.linspace(0,100,101)
        total_time = 1
        dt = 0.025
        params_dict = {}
        flags = {}
        type_ = "wrong_type"
        filename = "f3"
        ds3 = RDS(data, grid_x, grid_x, total_time, dt, params_dict, flags, type_, filename, show_index=0)
        
        self.idg.add(ds1)
        self.idg.add(ds2)
        self.idg.add(ds3) # Should be denied
        return
    
    def test_init(self):
        self.assertEqual(self.idg.dt, 0.025)
        self.assertEqual(self.idg.total_t, 1)
        self.assertEqual(self.idg.type, "X")
        self.assertEqual(len(self.idg.datasets), 2)
        return

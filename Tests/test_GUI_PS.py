# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:18:49 2022

@author: cfai2
"""

import unittest
import numpy as np

from GUI_structs import Scalable_Plot_State as SPS
from GUI_structs import Integration_Plot_State as IPS
from GUI_structs import Integrated_Data_Group as IDG
from GUI_structs import Analysis_Plot_State as APS
from GUI_structs import Raw_Data_Group as RDG, Raw_Data_Set as RDS

class TestSPS(unittest.TestCase):
    
    def test_init(self):
        self.sps = SPS()
        self.assertTrue(hasattr(self.sps, "plot_obj"))
        self.assertTrue(hasattr(self.sps, "display_legend"))
        self.assertTrue(hasattr(self.sps, "do_freeze_axes"))
        
        self.assertTrue(self.sps.xaxis_type == "linear" or self.sps.xaxis_type == 'log')
        self.assertTrue(self.sps.yaxis_type == "linear" or self.sps.yaxis_type == 'log')
        
        self.assertIsInstance(self.sps.xlim, tuple)
        self.assertIsInstance(self.sps.ylim, tuple)
        return

class TestIPS(unittest.TestCase):
    
    def test_init(self):
        self.ips = IPS()
        self.assertIsInstance(self.ips, SPS)
        self.assertIsInstance(self.ips.mode, str)
        self.assertIsInstance(self.ips.x_param, str)
        self.assertIsInstance(self.ips.datagroup, IDG)
        return
    
class TestAPS(unittest.TestCase):
    def setUp(self):
        self.aps = APS()
        self.data1 = np.linspace(0,100,101)
        self.grid_x1 = np.linspace(0,100,101)
        total_time = 1
        dt = 0.025
        params_dict = {}
        flags = {}
        type_ = "X"
        filename = "f1"
        ds1 = RDS(self.data1, self.grid_x1, self.grid_x1, total_time, dt, params_dict, flags, type_, filename, current_time=0)
        self.aps.datagroup.add(ds1)
    
    def test_init(self):
        
        self.assertEqual(self.aps.data_filenames, [])
        self.assertIsInstance(self.aps.datagroup, RDG)
        
        self.aps.add_time(2000)
        self.assertEqual(self.aps.time, 1)
        
        self.aps.add_time(-2000)
        self.assertEqual(self.aps.time, 0)
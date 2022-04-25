# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:18:49 2022

@author: cfai2
"""

import unittest
import numpy as np

from helper_structs import Characteristic
from helper_structs import Parameter
from helper_structs import Output
from helper_structs import Layer

class TestC(unittest.TestCase):

    def test_init(self):
        c = Characteristic("cm s^-1", True)
        self.assertEqual(c.units, "cm s^-1")
        self.assertTrue(c.is_edge)
        return
    
class TestP(unittest.TestCase):
    
    def test_init(self):
        units = "cm"
        is_edge = False
        p = Parameter(units, is_edge)
        
        self.assertEqual(p.value, 0)
        self.assertEqual(p.valid_range, (-np.inf, np.inf))
        self.assertTrue(p.is_space_dependent)
        self.assertEqual(p.param_rules, [])
        return
    
class TestO(unittest.TestCase):
    
    def test_init(self):
        display_name = "MyOutput"
        units = "hamburgers/ftballfield"
        xlabel = "[nm]"
        xvar = "position"
        is_edge = True
        layer = "First layer"
        
        o = Output(display_name, units, xlabel, xvar, is_edge, layer, integrated_units=None, analysis_plotable=True, yscale='symlog', yfactors=(1,1))
        
        self.assertIsInstance(o.display_name, str)
        self.assertIsInstance(o.xlabel, str)
        self.assertIsInstance(o.xvar, str)
        self.assertIsInstance(o.layer, str)
        self.assertEqual(o.integrated_units, units)
        self.assertTrue(o.analysis_plotable)
        self.assertIsInstance(o.yscale, str)
        self.assertIsInstance(o.yfactors, tuple)
        
        o = Output(display_name, units, xlabel, xvar, is_edge, layer, integrated_units="new units", analysis_plotable=True, yscale='symlog', yfactors=(1,1))
        self.assertEqual(o.integrated_units, "new units")
        return
    
class TestL(unittest.TestCase):
    
    def test_init(self):
        params = {"Param1":Parameter("cm", False)}
        s_outputs = {"Output1":Output("MyOutput", "cm", "[nm]", "position", False, "Layer1")}
        c_outputs = {"Output2":Output("MyOutput", "cm", "[nm]", "position", False, "Layer1", integrated_units="new units")}
        length_unit = "nm"
        convert_in = {"Param1":2, "Output1":2, "Output2":2}
        iconvert_in = {"Output2":2}
        l = Layer(params, s_outputs, c_outputs, length_unit, convert_in, iconvert_in)
        
        self.assertEqual(l.params, params)
        self.assertEqual(l.param_count, len(params))
        self.assertEqual(l.s_outputs, s_outputs)
        self.assertEqual(l.c_outputs, c_outputs)
        self.assertIsInstance(l.outputs, dict)
        self.assertEqual(l.s_outputs_count, 1)
        self.assertEqual(l.c_outputs_count, 1)
        self.assertEqual(l.outputs_count, 2)
        self.assertEqual(l.total_length, -1)
        self.assertEqual(l.dx, -1)
        self.assertEqual(l.length_unit, length_unit)
        self.assertEqual(l.grid_x_nodes, -1)
        self.assertEqual(l.grid_x_edges, -1)
        self.assertEqual(l.spacegrid_is_set, False)
        self.assertEqual(l.convert_in, convert_in)
        self.assertEqual(l.iconvert_in, iconvert_in)
        
        # Since all conversions were 2
        convert_out = {"Param1":0.5, "Output1":0.5, "Output2":0.5}
        iconvert_out = {"Output2":0.5}
        self.assertEqual(l.convert_out, convert_out)
        self.assertEqual(l.iconvert_out, iconvert_out)
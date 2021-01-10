# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:49:49 2021

@author: cfai2
"""

class Characteristic:
    def __init__(self, units, is_edge):
        self.units = units
        self.is_edge = is_edge
        return

class Parameter(Characteristic):
    # Helper class to store info about each of a Nanowire's parameters and initial distributions
    def __init__(self, units, is_edge):
        super().__init__(units, is_edge)
        # self.value can be a number (i.e. the parameter value is constant across the length of the nanowire)
        # or an array (i.e. the parameter value is spatially dependent)
        self.value = 0
        self.param_rules = []
        return
    
class Output(Characteristic):
    
    def __init__(self, display_name, units, xlabel, xvar, is_edge, is_calculated=False, calc_func=None, is_integrated=False, yscale='log', yfactors=(1,1)):
        super().__init__(units, is_edge)
        self.display_name = display_name
        self.xlabel = xlabel
        self.xvar = xvar
        self.is_calculated = is_calculated
        self.is_integrated = is_integrated
        self.yscale = yscale
        self.yfactors = yfactors
        self.calc_func = calc_func
        return    
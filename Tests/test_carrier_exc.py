# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:18:49 2022

@author: cfai2
"""

import unittest
import numpy as np
from scipy.integrate import trapz

from carrier_excitations import pulse_laser_power_spotsize
from carrier_excitations import pulse_laser_powerdensity
from carrier_excitations import pulse_laser_maxgen
from carrier_excitations import pulse_laser_totalgen

class TestUtils(unittest.TestCase):
    
    def test_spotsize(self):
        # Base case: these should all cancel to 1
        hc = 1240 # eV/nm
        wavelength = 1240
        spotsize = 1
        freq = 2
        power = 2
        alpha = 10
        
        x_array = np.linspace(0,1,100)
        out = pulse_laser_power_spotsize(power, spotsize, freq, wavelength, 
                                         alpha, x_array, hc)
        
        np.testing.assert_equal(out, alpha * np.exp(-alpha*x_array))
        return
    
    def test_density(self):
        # Base case: these should all cancel to 1
        hc = 1240 # eV/nm
        wavelength = 1240
        freq = 2
        power_density = 2
        alpha = 10
        
        x_array = np.linspace(0,1,100)
        out = pulse_laser_powerdensity(power_density, freq, wavelength, 
                                       alpha, x_array, hc)
        
        np.testing.assert_equal(out, alpha * np.exp(-alpha*x_array))
        return
    
    def test_maxgen(self):
        # Simple exponential decay profile
        hc = 1240 # eV/nm
        max_gen = 2
        alpha = 10
        
        x_array = np.linspace(0,1,100)
        out = pulse_laser_maxgen(max_gen, alpha, x_array, hc)
        
        np.testing.assert_equal(out, max_gen * np.exp(-alpha*x_array))
        return
    
    def test_totalgen(self):
        # The total gen should set the integral of the carrier distribution
        hc = 1240 # eV/nm
        total_gen = 2
        total_length = 1
        alpha = 10
        
        x_array = np.linspace(0,1,20000)
        out = pulse_laser_totalgen(total_gen, total_length, alpha, x_array, hc)
        
        
        self.assertAlmostEqual(trapz(out, dx=x_array[1]-x_array[0]), 2)
        return
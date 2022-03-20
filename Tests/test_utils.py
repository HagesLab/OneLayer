import unittest
import numpy as np
from utils import to_index
from utils import to_pos
from utils import to_array
from utils import get_all_combinations
from utils import autoscale
from utils import correct_integral
from utils import new_integrate

class TestUtils(unittest.TestCase):
    
    def test_to_index(self):
        # Grid: x = [0.5, 1.5, ..., 9.5]
        upper_bound = 10
        dx = 1
        with self.assertRaises(ValueError):
            to_index(-1, dx, upper_bound, is_edge=False)
        with self.assertRaises(ValueError):
            to_index(1000, dx, upper_bound, is_edge=False)
            
        self.assertEqual(to_index(0, dx, upper_bound, is_edge=False), 0)
        self.assertEqual(to_index(1.4999, dx, upper_bound, is_edge=False), 0)
        self.assertEqual(to_index(1.5001, dx, upper_bound, is_edge=False), 1)
        self.assertEqual(to_index(9.4, dx, upper_bound, is_edge=False), 8)
        self.assertEqual(to_index(10, dx, upper_bound, is_edge=False), 9)
        
        # Grid: x = [0,1,...,10]
        with self.assertRaises(ValueError):
            to_index(-1, dx, upper_bound, is_edge=True)
        with self.assertRaises(ValueError):
            to_index(1000, dx, upper_bound, is_edge=True)
            
        self.assertEqual(to_index(0, dx, upper_bound, is_edge=True), 0)
        self.assertEqual(to_index(0.999, dx, upper_bound, is_edge=True), 0)
        self.assertEqual(to_index(1.0001, dx, upper_bound, is_edge=True), 1)
        self.assertEqual(to_index(9.999, dx, upper_bound, is_edge=True), 9)
        self.assertEqual(to_index(10, dx, upper_bound, is_edge=True), 10)
        
    def test_to_pos(self):
        # Grid: x = [1, 3, ...]
        dx = 2
        self.assertEqual(to_pos(0, dx, is_edge=False), 1)
        self.assertEqual(to_pos(62, dx, is_edge=False), 125)
        
        # Grid: x = [0, 2, ...]
        self.assertEqual(to_pos(0, dx, is_edge=True), 0)
        self.assertEqual(to_pos(23235, dx, is_edge=True), 46470)
        
    def test_to_array(self):
        size = 10
        x = 12
        expected = np.ones(size) * x
        np.testing.assert_equal(to_array(x, size, is_edge=False), expected)
        
        expected = np.ones(size+1) * x
        np.testing.assert_equal(to_array(x, size, is_edge=True), expected)
        
        already_an_array = expected
        np.testing.assert_equal(to_array(already_an_array, size, is_edge=True), already_an_array)
        
    def test_get_all_combinations(self):
        test_empty = {}
        expected = []
        self.assertEqual(get_all_combinations(test_empty), expected)
        
        test_one = {"Param1":[1,2,3]}
        expected = [{"Param1":1},{"Param1":2},{"Param1":3}]
        self.assertEqual(get_all_combinations(test_one), expected)
        
        test_two = {"Param1":[1,2,3], "Param2":[4,5]}
        expected = [{"Param1":1,"Param2":4},{"Param1":1,"Param2":5},
                    {"Param1":2,"Param2":4},{"Param1":2,"Param2":5},
                    {"Param1":3,"Param2":4},{"Param1":3,"Param2":5}] 
        self.assertEqual(get_all_combinations(test_two), expected)
        
    def test_autoscale(self):
        with self.assertRaises(AssertionError):
            autoscale(min_val=1, max_val=0)
        
        self.assertEqual(autoscale(min_val=0, max_val=0), 'linear')
        
        self.assertEqual(autoscale(min_val=0, max_val=1), 'linear')
        self.assertEqual(autoscale(min_val=-1, max_val=0), 'linear')
                         
        self.assertEqual(autoscale(min_val=1, max_val=100), 'symlog')
        self.assertEqual(autoscale(min_val=-100, max_val=-1), 'symlog')
                    
        self.assertEqual(autoscale(min_val=-100, max_val=1), 'symlog')
        self.assertEqual(autoscale(min_val=-1, max_val=100), 'symlog')
        
    def test_correct_integral(self):
        dx = 1
        thickness = 100
        nx = 100
        # x = [0.5, 1.5, ...99.5]
        x = np.linspace(dx/2, thickness-dx/2, nx)
        
        # y = [5,15,25,...995]
        integrand = 10*np.linspace(dx/2, thickness-dx/2, nx)
        
        # No correction needed if bounds match
        l_bound = 0.5
        u_bound = 99.5
        out = correct_integral(integrand, l_bound, u_bound, dx, thickness)
        self.assertEqual(out, 0.0)
        
        # Add one quarter of the first node
        l_bound = 0.25
        u_bound = 99.5
        out = correct_integral(integrand, l_bound, u_bound, dx, thickness)
        self.assertEqual(out, 1.25)
        
        # REMOVE one quarter of the first node
        l_bound = 0.75
        u_bound = 99.5
        out = correct_integral(integrand, l_bound, u_bound, dx, thickness)
        self.assertEqual(out, -1.25)
        
        # REMOVE one half of the first node and one quarter of the second node
        l_bound = 1.25 # maps to real pos -> 0.5
        u_bound = 99.5 # -> 9.5

        out = correct_integral(integrand, l_bound, u_bound, dx, thickness)
        self.assertEqual(out, -2.5 - 3.75)
        
        # Add half the last node
        l_bound = 0.5
        u_bound = 100
        out = correct_integral(integrand, l_bound, u_bound, dx, thickness)
        self.assertEqual(out, 497.5)
        
        # Add half the 50th node and a quarter of the 51st node
        l_bound = 0.5
        u_bound = 50.25 # -> 49.5
        out = correct_integral(integrand, l_bound, u_bound, dx, thickness)
        self.assertEqual(out, 0.5*495 + 0.25*505)
        
        # Integrate over nominal thickness - add half the area of the first and last nodes
        l_bound = 0
        u_bound = 100
        out = correct_integral(integrand, l_bound, u_bound, dx, thickness)
        self.assertEqual(out, 500)
        
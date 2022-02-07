import unittest
import numpy as np
from utils import to_index
from utils import to_pos
from utils import to_array

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
import unittest
import tables
import os
import numpy as np

from io_utils import get_split_and_clean_line
from io_utils import check_valid_filename
from io_utils import extract_values
from io_utils import u_read

class TestUtils(unittest.TestCase):
    
    def test_get_split_and_clean_line(self):
        line_1 = "System_class: MAPI_Rubrene"
        split_line_1 = get_split_and_clean_line(line_1)
        self.assertEqual(len(split_line_1), 2)
        self.assertEqual(split_line_1[0], "System_class")
        self.assertEqual(split_line_1[1], "MAPI_Rubrene")

        line_2 = "Power: 0.22999999999999998"
        split_line_2 = get_split_and_clean_line(line_2)
        self.assertEqual(len(split_line_2), 2)
        self.assertEqual(split_line_2[0], 'Power')
        self.assertEqual(split_line_2[1], '0.22999999999999998')
        
        # TODO: adding an assert len()=2 to the method itself
        # to raise AssertionError (and handle it somewhere else)
        # when none or more than 1 colon ":" is present

        line_3 = "  Random Test Line        "
        split_line_3 = get_split_and_clean_line(line_3)
        self.assertEqual(len(split_line_3), 1)
        self.assertEqual(split_line_3[0], 'Random Test Line')

        line_4 = " 2:colon:line"
        split_line_4 = get_split_and_clean_line(line_4)
        self.assertEqual(len(split_line_4), 3)
        self.assertEqual(split_line_4[0], '2')
        self.assertEqual(split_line_4[1], 'colon')
        self.assertEqual(split_line_4[2], 'line')

    def test_check_valid_filename(self):
        filename_1 = "a_filename_without_weird_characters"
        self.assertEqual(check_valid_filename(filename_1), True)

        filename_2 = "?a<filename>with/weird*characters."
        self.assertEqual(check_valid_filename(filename_2), False)

    def test_extract_values(self):
        string = "100;200;300"
        delimiter = ";"
        extracted_values = extract_values(string, delimiter)
        self.assertEqual(extracted_values[0], 100)
        self.assertEqual(extracted_values[1], 200)
        self.assertEqual(extracted_values[2], 300)

        string = "100.01,200.02,300.03"
        delimiter = ","
        extracted_values = extract_values(string, delimiter)
        self.assertEqual(extracted_values[0], 100.01)
        self.assertEqual(extracted_values[1], 200.02)
        self.assertEqual(extracted_values[2], 300.03)

        # The following fails due to the comma:
        # ValueError: could not convert string to float: '100,01'
        #string = "100,01;200,02;300,03"
        #delimiter = ";"
        
    def test_u_read(self):
        np.random.seed(42)
        test_array = np.random.normal(size=(5,5))
        path = os.path.join("u_read_test1.h5")
        with tables.open_file(path, mode='w') as ofstream:

            # Important - "data" must be used as the array name here, as pytables will use the string "data" 
            # to name the attribute earray.data, which is then used to access the array
            earray = ofstream.create_array(ofstream.root, "data", test_array)

        test_out = u_read(path)
        
        # Read full array
        np.testing.assert_equal(test_array, test_out)
        
        # Read one time step
        # Force_1D will coerce this into a 1D array (for plotting) by default
        t0 = 2
        test_out = u_read(path, t0=t0, single_tstep=True)
        
        np.testing.assert_equal(test_array[t0], test_out)
        
        # Read one time step, but an integral wants it to stay as a 2D array
        test_out = u_read(path, t0=t0, single_tstep=True, force_1D=False)
        
        np.testing.assert_equal(test_array[t0, np.newaxis], test_out)
        
        # Read a few space steps
        l = 1
        r = 3
        test_out = u_read(path, l=l, r=r)
        
        np.testing.assert_equal(test_array[:,l:r], test_out)
        
        # "I need an extra space node for this integral"
        test_out = u_read(path, l=l, r=r, need_extra_node=True)
        
        np.testing.assert_equal(test_array[:,l:r+1], test_out)
        
        
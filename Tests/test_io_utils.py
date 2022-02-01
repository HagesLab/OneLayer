import unittest
from io_utils import get_split_and_clean_line
from io_utils import check_valid_filename
from io_utils import extract_values

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
        
        
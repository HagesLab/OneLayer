import numpy as np
from utils import to_array


class PN_Junction_Initial_Conditions():
    def __init__(self, ntype, buffer, ptype):
        """Taking information from MAPI and Rubrene 'Layer' types
        we generate and return initial conditions in a dictionary."""

        self.ntype = ntype
        self.buffer = buffer
        self.ptype = ptype
        
        self.calc_inits_from_layers()
        
        self.uniformize_inits_to_arrays()


    def calc_inits_from_layers(self):
        """ Physical calculations of initial distributions."""

        self.ntype_init_N = (
            self.ntype.params["N0"].value + self.ntype.params["delta_N"].value
            ) * self.ntype.convert_in["N"]

        self.ntype_init_P = (
            self.ntype.params["P0"].value + self.ntype.params["delta_P"].value
            ) * self.ntype.convert_in["P"]
        
        self.buffer_init_N = (
            self.buffer.params["N0"].value + self.buffer.params["delta_N"].value
            ) * self.buffer.convert_in["N"]

        self.buffer_init_P = (
            self.buffer.params["P0"].value + self.buffer.params["delta_P"].value
            ) * self.buffer.convert_in["P"]
        
        self.ptype_init_N = (
            self.ptype.params["N0"].value + self.ptype.params["delta_N"].value
            ) * self.ptype.convert_in["N"]

        self.ptype_init_P = (
            self.ptype.params["P0"].value + self.ptype.params["delta_P"].value
            ) * self.ptype.convert_in["P"]
    

    def uniformize_inits_to_arrays(self):
        """ 'Typecast' single values to uniform arrays. """

        def generate_array_if_not_yet(array_to_check, array_length):
            if not isinstance(array_to_check, np.ndarray):
                array_to_check = to_array(array_to_check, array_length, False)

            return array_to_check

        ntype_length = len(self.ntype.grid_x_nodes)
        buffer_length = len(self.buffer.grid_x_nodes)
        ptype_length = len(self.ptype.grid_x_nodes)

        self.ntype_init_N = generate_array_if_not_yet(self.ntype_init_N, ntype_length)
        self.ntype_init_P = generate_array_if_not_yet(self.ntype_init_P, ntype_length)
        self.buffer_init_N = generate_array_if_not_yet(self.buffer_init_N, buffer_length)
        self.buffer_init_P = generate_array_if_not_yet(self.buffer_init_P, buffer_length)
        self.ptype_init_N = generate_array_if_not_yet(self.ptype_init_N, ptype_length)
        self.ptype_init_P = generate_array_if_not_yet(self.ptype_init_P, ptype_length)


    def format_inits_to_dict(self):
        """Returning the calculated initial arrays
        organized in a dictionary as per contract."""
        return {
            "N": np.hstack([self.ntype_init_N, self.buffer_init_N, self.ptype_init_N]),
            "P": np.hstack([self.ntype_init_P, self.buffer_init_P, self.ptype_init_P]),
        }
